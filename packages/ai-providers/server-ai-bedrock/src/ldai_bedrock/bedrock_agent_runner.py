from typing import Any, Dict, List, Optional

from ldai import log
from ldai.providers import RunnerResult, ToolRegistry
from ldai.providers.runner import Runner
from ldai.providers.types import LDAIMetrics
from ldai.tracker import TokenUsage


class BedrockAgentRunner(Runner):
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use.
    It may change or be removed without notice and is not subject to backwards
    compatibility guarantees.

    Runner implementation for a single agent running on Amazon Bedrock via
    the Strands Agents SDK (``strands-agents``).

    Tool calling and the agentic loop are handled internally by the Strands
    ``Agent`` class.  Returned by ``BedrockRunnerFactory.create_agent(config, tools)``.

    Implements the unified :class:`~ldai.providers.runner.Runner` protocol.
    Requires ``strands-agents`` to be installed:

    ``pip install launchdarkly-server-sdk-ai-bedrock[agents]``
    """

    def __init__(
        self,
        model_id: str,
        parameters: Dict[str, Any],
        instructions: str,
        tool_definitions: List[Dict[str, Any]],
        tools: ToolRegistry,
    ):
        self._model_id = model_id
        self._parameters = parameters
        self._instructions = instructions
        self._tool_definitions = tool_definitions
        self._tools = tools

    async def run(
        self,
        input: str,
        output_type: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Run the agent with the given input.

        Delegates to Strands' ``Agent.invoke_async``, which handles the
        tool-calling loop internally.

        :param input: The user prompt string to the agent
        :param output_type: Reserved for future structured output support;
            currently ignored.
        :return: :class:`RunnerResult` with ``content``, ``raw`` response, and
            metrics including aggregated token usage and observed ``tool_calls``.
        """
        try:
            from strands import Agent
        except ImportError:
            log.warning(
                "strands-agents is required for BedrockAgentRunner. "
                "Install it with: pip install launchdarkly-server-sdk-ai-bedrock[agents]"
            )
            return RunnerResult(
                content='',
                metrics=LDAIMetrics(success=False, tokens=None),
            )

        try:
            agent_tools = self._build_agent_tools()
            agent_kwargs: Dict[str, Any] = {
                'model': self._model_id,
                'tools': agent_tools,
            }
            if self._instructions:
                agent_kwargs['system_prompt'] = self._instructions

            agent = Agent(**agent_kwargs)
            result = await agent.invoke_async(str(input))

            return RunnerResult(
                content=_extract_strands_content(result),
                metrics=LDAIMetrics(
                    success=True,
                    tokens=_extract_strands_usage(result),
                    tool_calls=_extract_strands_tool_calls(result) or None,
                ),
                raw=result,
            )
        except Exception as error:
            log.warning(f'Bedrock agent run failed: {error}')
            return RunnerResult(
                content='',
                metrics=LDAIMetrics(success=False, tokens=None),
            )

    def _build_agent_tools(self) -> List[Any]:
        """Build the tool list passed to ``Agent(tools=...)``.

        Plain callables are wrapped with Strands' ``@tool`` decorator; values
        that are already Strands tool objects are returned unchanged.
        """
        try:
            from strands import tool as strands_tool
        except ImportError as exc:
            raise ImportError(
                "strands-agents is required for agent tools. "
                "Install it with: pip install launchdarkly-server-sdk-ai-bedrock[agents]"
            ) from exc

        tools: List[Any] = []
        for td in self._tool_definitions:
            if not isinstance(td, dict):
                continue
            name = td.get('name', '')
            if not name:
                continue
            tool_fn = self._tools.get(name)
            if tool_fn is None:
                log.warning(
                    f"Tool '{name}' is defined in the AI config but was not found in "
                    "the tool registry; skipping."
                )
                continue
            if callable(tool_fn) and not _is_strands_tool(tool_fn):
                tools.append(strands_tool(tool_fn))
            else:
                tools.append(tool_fn)
        return tools


def _is_strands_tool(value: Any) -> bool:
    """Heuristically detect a value already decorated as a Strands tool."""
    return hasattr(value, 'tool_name') or hasattr(value, 'tool_spec')


def _extract_strands_content(result: Any) -> str:
    """Best-effort extraction of the final text output from a Strands AgentResult."""
    if result is None:
        return ''
    # Strands' AgentResult is convertible to str and exposes ``message`` with
    # Bedrock-style content blocks.  Prefer the structured form so we never
    # accidentally surface a debug repr.
    message = getattr(result, 'message', None)
    if isinstance(message, dict):
        content = message.get('content')
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get('text')
                    if isinstance(text, str) and text:
                        parts.append(text)
            if parts:
                return ''.join(parts)
    return str(result)


def _extract_strands_usage(result: Any) -> Optional[TokenUsage]:
    """Extract aggregate token usage from a Strands AgentResult, if available."""
    if result is None:
        return None
    metrics = getattr(result, 'metrics', None)
    if metrics is None:
        return None
    usage = getattr(metrics, 'accumulated_usage', None) or getattr(metrics, 'usage', None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        total = usage.get('totalTokens') or usage.get('total_tokens') or 0
        inp = usage.get('inputTokens') or usage.get('input_tokens') or 0
        out = usage.get('outputTokens') or usage.get('output_tokens') or 0
    else:
        total = getattr(usage, 'totalTokens', None) or getattr(usage, 'total_tokens', 0) or 0
        inp = getattr(usage, 'inputTokens', None) or getattr(usage, 'input_tokens', 0) or 0
        out = getattr(usage, 'outputTokens', None) or getattr(usage, 'output_tokens', 0) or 0
    if not (total or inp or out):
        return None
    return TokenUsage(total=total, input=inp, output=out)


def _extract_strands_tool_calls(result: Any) -> List[str]:
    """Return the names of any tool calls recorded on a Strands AgentResult."""
    if result is None:
        return []
    metrics = getattr(result, 'metrics', None)
    if metrics is None:
        return []
    tool_metrics = getattr(metrics, 'tool_metrics', None)
    if isinstance(tool_metrics, dict):
        names: List[str] = []
        for name, entry in tool_metrics.items():
            call_count = getattr(entry, 'call_count', None)
            if call_count is None and isinstance(entry, dict):
                call_count = entry.get('call_count')
            count = int(call_count or 0) if call_count is not None else 1
            names.extend([str(name)] * max(count, 1))
        return names
    return []
