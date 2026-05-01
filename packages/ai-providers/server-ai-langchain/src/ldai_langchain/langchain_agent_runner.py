from typing import Any, Dict, Optional

from ldai import log
from ldai.providers.runner import Runner
from ldai.providers.types import LDAIMetrics, RunnerResult

from ldai_langchain.langchain_helper import (
    extract_last_message_content,
    sum_token_usage_from_messages,
)


class LangChainAgentRunner(Runner):
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use.
    It may change or be removed without notice and is not subject to backwards
    compatibility guarantees.

    Runner implementation for LangChain agents.

    Wraps a compiled LangChain agent graph (from ``langchain.agents.create_agent``)
    and delegates execution to it. Tool calling and loop management are handled
    internally by the graph.
    Returned by LangChainRunnerFactory.create_agent(config, tools).

    Implements the unified :class:`~ldai.providers.runner.Runner` protocol via
    :meth:`run`.
    """

    def __init__(self, agent: Any):
        self._agent = agent

    async def run(
        self,
        input: Any,
        output_type: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Run the agent with the given input string.

        Delegates to the compiled LangChain agent, which handles
        the tool-calling loop internally.

        :param input: The user prompt or input to the agent
        :param output_type: Reserved for future structured output support;
            currently ignored.
        :return: :class:`RunnerResult` with ``content``, ``raw`` response, and
            aggregated metrics.
        """
        try:
            result = await self._agent.ainvoke({
                "messages": [{"role": "user", "content": str(input)}]
            })
            messages = result.get("messages", [])
            output = extract_last_message_content(messages)
            tool_calls = _extract_tool_calls(messages)
            return RunnerResult(
                content=output,
                metrics=LDAIMetrics(
                    success=True,
                    usage=sum_token_usage_from_messages(messages),
                    tool_calls=tool_calls if tool_calls else None,
                ),
                raw=result,
            )
        except Exception as error:
            log.warning(f"LangChain agent run failed: {error}")
            return RunnerResult(
                content="",
                metrics=LDAIMetrics(success=False, usage=None),
            )

    def get_agent(self) -> Any:
        """Return the underlying compiled LangChain agent."""
        return self._agent


def _extract_tool_calls(messages: Any) -> list:
    """
    Extract tool-call names from a LangChain agent's message list.

    LangChain's ``AIMessage`` exposes ``.tool_calls`` as a list of dicts
    (``{'name': ..., 'args': ..., 'id': ...}``). Some providers emit
    OpenAI-style objects with ``.function.name`` instead; handle both shapes.
    """
    tool_calls: list = []
    for msg in messages or []:
        msg_tool_calls = getattr(msg, 'tool_calls', None)
        if not msg_tool_calls:
            continue
        for tc in msg_tool_calls:
            if isinstance(tc, dict) and 'name' in tc:
                tool_calls.append(tc['name'])
            else:
                fn = getattr(tc, 'function', None)
                name = getattr(fn, 'name', None) if fn is not None else None
                if name:
                    tool_calls.append(name)
    return tool_calls
