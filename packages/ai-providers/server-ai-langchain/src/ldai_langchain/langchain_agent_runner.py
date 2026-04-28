from typing import Any, Dict, List, Optional

from ldai import log
from ldai.providers.types import LDAIMetrics, RunnerResult

from ldai_langchain.langchain_helper import (
    extract_last_message_content,
    get_tool_calls_from_response,
    sum_token_usage_from_messages,
)


class LangChainAgentRunner:
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use.
    It may change or be removed without notice and is not subject to backwards
    compatibility guarantees.

    Runner implementation for a single LangChain agent.

    Wraps a compiled LangChain agent graph (from ``langchain.agents.create_agent``)
    and delegates execution to it. Tool calling and loop management are handled
    internally by the graph.
    Returned by ``LangChainRunnerFactory.create_agent(config, tools)``.

    Implements the unified :class:`~ldai.providers.runner.Runner` protocol.
    """

    def __init__(self, agent: Any):
        self._agent = agent

    async def run(
        self,
        input: Any,
        output_type: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Run the agent with the given input.

        Delegates to the compiled LangChain agent, which handles
        the tool-calling loop internally.

        :param input: The user prompt or input to the agent
        :param output_type: Reserved for future structured output support;
            currently ignored.
        :return: :class:`RunnerResult` with ``content``, ``raw`` response, and
            metrics including aggregated token usage and observed ``tool_calls``.
        """
        try:
            result = await self._agent.ainvoke({
                "messages": [{"role": "user", "content": str(input)}]
            })
            messages: List[Any] = result.get("messages", [])
            content = extract_last_message_content(messages)
            tool_calls = self._extract_tool_calls(messages)
            return RunnerResult(
                content=content,
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

    @staticmethod
    def _extract_tool_calls(messages: List[Any]) -> List[str]:
        """Collect tool call names from all messages in the agent output."""
        names: List[str] = []
        for msg in messages:
            names.extend(get_tool_calls_from_response(msg))
        return names

    def get_agent(self) -> Any:
        """Return the underlying compiled LangChain agent."""
        return self._agent
