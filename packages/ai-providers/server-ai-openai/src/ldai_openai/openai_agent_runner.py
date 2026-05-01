from typing import Any, Dict, List, Optional

from ldai import log
from ldai.providers import RunnerResult, ToolRegistry
from ldai.providers.runner import Runner
from ldai.providers.types import LDAIMetrics

from ldai_openai.openai_helper import (
    get_ai_usage_from_response,
    get_tool_calls_from_run_items,
    is_agent_tool_instance,
    registry_value_to_agent_tool,
)


class OpenAIAgentRunner(Runner):
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use.
    It may change or be removed without notice and is not subject to backwards
    compatibility guarantees.

    Runner implementation for a single OpenAI agent.

    Executes a single agent using the OpenAI Agents SDK (``openai-agents``).
    Tool calling and the agentic loop are handled internally by ``Runner.run``.
    Returned by ``OpenAIRunnerFactory.create_agent(config, tools)``.

    Implements the unified :class:`~ldai.providers.runner.Runner` protocol.
    Requires ``openai-agents`` to be installed.
    """

    def __init__(
        self,
        model_name: str,
        parameters: Dict[str, Any],
        instructions: str,
        tool_definitions: List[Dict[str, Any]],
        tools: ToolRegistry,
    ):
        self._model_name = model_name
        self._parameters = parameters
        self._instructions = instructions
        self._tool_definitions = tool_definitions
        self._tools = tools
        self._tool_name_map: Dict[str, str] = {}

    async def run(
        self,
        input: Any,
        output_type: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Run the agent with the given input.

        Delegates to the OpenAI Agents SDK ``Runner.run``, which handles the
        tool-calling loop internally.

        :param input: The user prompt or input to the agent
        :param output_type: Reserved for future structured output support;
            currently ignored.
        :return: :class:`RunnerResult` with ``content``, ``raw`` response, and
            metrics including aggregated token usage and observed ``tool_calls``.
        """
        try:
            from agents import Agent
            from agents import Runner as _Runner
        except ImportError:
            log.warning(
                "openai-agents is required for OpenAIAgentRunner. "
                "Install it with: pip install openai-agents"
            )
            return RunnerResult(
                content="",
                metrics=LDAIMetrics(success=False, usage=None),
            )

        try:
            agent_tools = self._build_agent_tools()
            model_settings = self._build_model_settings()

            agent = Agent(
                name="ldai-agent",
                instructions=self._instructions or None,
                model=self._model_name,
                tools=agent_tools,
                model_settings=model_settings,
            )

            result = await _Runner.run(agent, str(input), max_turns=25)

            tool_calls = [
                ld_name
                for _agent_name, tool_fn_name in get_tool_calls_from_run_items(result.new_items)
                for ld_name in [self._tool_name_map.get(tool_fn_name)]
                if ld_name is not None
            ]

            return RunnerResult(
                content=str(result.final_output),
                metrics=LDAIMetrics(
                    success=True,
                    usage=get_ai_usage_from_response(result),
                    tool_calls=tool_calls if tool_calls else None,
                ),
                raw=result,
            )
        except Exception as error:
            log.warning(f"OpenAI agent run failed: {error}")
            return RunnerResult(
                content="",
                metrics=LDAIMetrics(success=False, usage=None),
            )

    def _build_agent_tools(self) -> List[Any]:
        """Build tool instances from LD tool definitions and registry.

        Also populates ``self._tool_name_map`` so observed tool-call names
        from the runtime can be translated back to their LD config keys for
        metric reporting.
        """
        tools = []
        self._tool_name_map = {}
        for td in self._tool_definitions:
            if not isinstance(td, dict):
                continue
            name = td.get("name", "")
            if not name:
                continue

            tool_fn = self._tools.get(name)
            if tool_fn:
                # Map runtime tool name → LD config key for metrics (function __name__
                # for callables; identity for native tool instances — see get_tool_calls_from_run_items).
                if is_agent_tool_instance(tool_fn):
                    self._tool_name_map[tool_fn.name] = name
                else:
                    fn_name = getattr(tool_fn, '__name__', None)
                    if fn_name:
                        self._tool_name_map[fn_name] = name
                tools.append(registry_value_to_agent_tool(tool_fn))
                continue

            log.warning(
                f"Tool '{name}' is defined in the AI config but was not found in "
                "the tool registry; skipping."
            )
        return tools

    def _build_model_settings(self) -> Any:
        """Map LD model parameters to an openai-agents ModelSettings instance."""
        from agents import ModelSettings

        known = {
            "temperature", "top_p", "max_tokens",
            "frequency_penalty", "presence_penalty",
        }
        kwargs = {k: v for k, v in self._parameters.items() if k in known}
        return ModelSettings(**kwargs) if kwargs else None
