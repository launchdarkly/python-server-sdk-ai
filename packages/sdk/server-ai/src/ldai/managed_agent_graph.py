"""ManagedAgentGraph — LaunchDarkly managed wrapper for agent graph execution."""

from typing import Any

from ldai.providers import AgentGraphResult, AgentGraphRunner


class ManagedAgentGraph:
    """
    LaunchDarkly managed wrapper for AI agent graph execution.

    Holds an AgentGraphRunner. Auto-tracking of path,
    tool calls, handoffs, latency, and invocation success/failure is handled
    by the runner implementation.

    Obtain an instance via ``LDAIClient.create_agent_graph()``.
    """

    def __init__(
        self,
        runner: AgentGraphRunner,
    ):
        """
        Initialize ManagedAgentGraph.

        :param runner: The AgentGraphRunner to delegate execution to
        """
        self._runner = runner

    async def run(self, input: Any) -> AgentGraphResult:
        """
        Run the agent graph with the given input.

        Delegates to the underlying AgentGraphRunner, which handles
        execution and all auto-tracking internally.

        :param input: The input prompt or structured input for the graph
        :return: AgentGraphResult containing the output, raw response, and metrics
        """
        return await self._runner.run(input)

    def get_agent_graph_runner(self) -> AgentGraphRunner:
        """
        Return the underlying AgentGraphRunner for advanced use.

        :return: The AgentGraphRunner instance
        """
        return self._runner
