from typing import Any

from ldai import log
from ldai.providers import AgentResult, AgentRunner
from ldai.providers.types import LDAIMetrics

from ldai_langchain.langchain_helper import (
    extract_last_message_content,
    sum_token_usage_from_messages,
)


class LangChainAgentRunner(AgentRunner):
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use. 
    It may change or be removed without notice and is not subject to backwards 
    compatibility guarantees.

    AgentRunner implementation for LangChain.

    Wraps a compiled LangChain agent graph (from ``langchain.agents.create_agent``)
    and delegates execution to it. Tool calling and loop management are handled
    internally by the graph.
    Returned by LangChainRunnerFactory.create_agent(config, tools).
    """

    def __init__(self, agent: Any):
        self._agent = agent

    async def run(self, input: Any) -> AgentResult:
        """
        Run the agent with the given input string.

        Delegates to the compiled LangChain agent, which handles
        the tool-calling loop internally.

        :param input: The user prompt or input to the agent
        :return: AgentResult with output, raw response, and aggregated metrics
        """
        try:
            result = await self._agent.ainvoke({
                "messages": [{"role": "user", "content": str(input)}]
            })
            messages = result.get("messages", [])
            output = extract_last_message_content(messages)
            return AgentResult(
                output=output,
                raw=result,
                metrics=LDAIMetrics(
                    success=True,
                    usage=sum_token_usage_from_messages(messages),
                ),
            )
        except Exception as error:
            log.warning(f"LangChain agent run failed: {error}")
            return AgentResult(
                output="",
                raw=None,
                metrics=LDAIMetrics(success=False, usage=None),
            )

    def get_agent(self) -> Any:
        """Return the underlying compiled LangChain agent."""
        return self._agent
