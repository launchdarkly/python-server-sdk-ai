from typing import Protocol, runtime_checkable

from ldai.providers.types import AgentGraphRunnerResult


@runtime_checkable
class AgentGraphRunner(Protocol):
    """
    CAUTION:
    This feature is experimental and should NOT be considered ready for production use.
    It may change or be removed without notice and is not subject to backwards
    compatibility guarantees.

    Runtime capability interface for multi-agent graph execution.

    An AgentGraphRunner is a focused, configured object returned by
    AIProvider.create_agent_graph(). It holds all provider wiring internally —
    the caller just passes input.
    """

    async def run(self, input: str) -> AgentGraphRunnerResult:
        """
        Run the agent graph with the given input.

        :param input: The string prompt to send to the agent graph
        :return: AgentGraphRunnerResult containing the content, raw response, and AIGraphMetrics
        """
        ...
