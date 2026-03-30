"""OpenAI agent runner for LaunchDarkly AI SDK."""

import json
from typing import Any, Dict, List

from ldai import log
from ldai.providers import AgentResult, AgentRunner, ToolRegistry
from ldai.providers.types import LDAIMetrics

from ldai_openai.openai_helper import (
    NATIVE_OPENAI_TOOLS,
    get_ai_usage_from_response,
)


class OpenAIAgentRunner(AgentRunner):
    """
    AgentRunner implementation for OpenAI.

    Executes a single agent using the OpenAI Agents SDK (``openai-agents``).
    Tool calling and the agentic loop are handled internally by ``Runner.run``.
    Returned by OpenAIRunnerFactory.create_agent(config, tools).

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

    async def run(self, input: Any) -> AgentResult:
        """
        Run the agent with the given input string.

        Delegates to the OpenAI Agents SDK ``Runner.run``, which handles the
        tool-calling loop internally.

        :param input: The user prompt or input to the agent
        :return: AgentResult with output, raw response, and aggregated metrics
        """
        try:
            from agents import Agent, Runner
        except ImportError:
            log.warning(
                "openai-agents is required for OpenAIAgentRunner. "
                "Install it with: pip install openai-agents"
            )
            return AgentResult(output="", raw=None, metrics=LDAIMetrics(success=False, usage=None))

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

            result = await Runner.run(agent, str(input), max_turns=25)

            return AgentResult(
                output=str(result.final_output),
                raw=result,
                metrics=LDAIMetrics(
                    success=True,
                    usage=get_ai_usage_from_response(result),
                ),
            )
        except Exception as error:
            log.warning(f"OpenAI agent run failed: {error}")
            return AgentResult(output="", raw=None, metrics=LDAIMetrics(success=False, usage=None))

    def _build_agent_tools(self) -> List[Any]:
        """Build tool instances from LD tool definitions and registry."""
        from agents import FunctionTool
        from agents.tool_context import ToolContext

        tools = []
        for td in self._tool_definitions:
            if not isinstance(td, dict):
                continue
            name = td.get("name", "")

            # Native OpenAI tools run on OpenAI's infrastructure — no local fn required.
            if name and name in NATIVE_OPENAI_TOOLS:
                tools.append(NATIVE_OPENAI_TOOLS[name](td))
                continue

            tool_type = td.get("type")
            if tool_type and tool_type != "function":
                log.warning(
                    f"Built-in tool '{tool_type}' is not supported and will be skipped. "
                    "Use the OpenAIAgentGraphRunner for built-in provider tools."
                )
                continue

            if not name:
                continue

            tool_fn = self._tools.get(name)
            if not tool_fn:
                log.warning(
                    f"Tool '{name}' is defined in the AI config but was not found in "
                    "the tool registry; skipping."
                )
                continue

            def _make_invoker(fn: Any, tool_name: str) -> Any:
                async def on_invoke_tool(tool_ctx: ToolContext, args_json: str) -> str:
                    try:
                        args = json.loads(args_json) if args_json else {}
                    except Exception:
                        args = {}
                    try:
                        res = fn(**args)
                        if hasattr(res, "__await__"):
                            res = await res
                        return str(res)
                    except Exception as e:
                        log.warning(f"Tool '{tool_name}' execution failed: {e}")
                        return f"Tool execution failed: {e}"
                return on_invoke_tool

            tools.append(FunctionTool(
                name=name,
                description=td.get("description", ""),
                params_json_schema=td.get("parameters", {}),
                on_invoke_tool=_make_invoker(tool_fn, name),
            ))
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
