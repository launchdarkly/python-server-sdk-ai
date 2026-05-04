"""ManagedAgentGraph — LaunchDarkly managed wrapper for agent graph execution."""

from typing import Any, Dict

from ldai.agent_graph import AgentGraphDefinition
from ldai.providers import AgentGraphRunner
from ldai.providers.types import (
    AgentGraphRunnerResult,
    LDAIMetrics,
    ManagedGraphResult,
)
from ldai.tracker import LDAIMetricSummary


class ManagedAgentGraph:
    """
    LaunchDarkly managed wrapper for AI agent graph execution.

    Holds an AgentGraphRunner and an AgentGraphDefinition. Delegates execution
    to the runner, then drives all graph-level and per-node tracking from the
    returned :class:`~ldai.providers.types.AgentGraphRunnerResult`.

    Obtain an instance via ``LDAIClient.create_agent_graph()``.
    """

    def __init__(
        self,
        graph: AgentGraphDefinition,
        runner: AgentGraphRunner,
    ):
        """
        Initialize ManagedAgentGraph.

        :param graph: The AgentGraphDefinition used to drive graph-level and
            per-node tracking from the runner result metrics.
        :param runner: The AgentGraphRunner to delegate execution to
        """
        self._graph = graph
        self._runner = runner

    async def run(self, input: Any) -> ManagedGraphResult:
        """
        Run the agent graph with the given input.

        Delegates to the underlying AgentGraphRunner, then drives all
        LaunchDarkly tracking from ``result.metrics``.

        :param input: The input prompt or structured input for the graph
        :return: ManagedGraphResult containing the content, metric summary,
            and raw response.
        """
        graph_tracker = self._graph.create_tracker()
        result = await graph_tracker.track_graph_metrics_of_async(
            lambda r: r.metrics,
            lambda: self._runner.run(input),
        )

        summary = graph_tracker.get_summary()
        summary.node_metrics = self._track_node_metrics(result.metrics.node_metrics)

        return ManagedGraphResult(
            content=result.content,
            metrics=summary,
            raw=result.raw,
            evaluations=None,
        )

    def _track_node_metrics(
        self, node_metrics: Dict[str, LDAIMetrics]
    ) -> Dict[str, LDAIMetricSummary]:
        """
        Drive per-node LaunchDarkly tracking events and collect node metric summaries.

        For each node key present in ``node_metrics``, obtains the node's
        config tracker via the graph definition, fires tracking events, and
        returns a map of node key to the tracker's metric summary.
        """
        node_summaries: Dict[str, LDAIMetricSummary] = {}
        for node_key, node_ldai_metrics in node_metrics.items():
            node = self._graph.get_node(node_key)
            if node is None:
                continue
            node_tracker = node.get_config().create_tracker()

            if node_ldai_metrics.usage is not None:
                node_tracker.track_tokens(node_ldai_metrics.usage)
            if node_ldai_metrics.duration_ms is not None:
                node_tracker.track_duration(node_ldai_metrics.duration_ms)
            if node_ldai_metrics.tool_calls:
                node_tracker.track_tool_calls(node_ldai_metrics.tool_calls)
            if node_ldai_metrics.success:
                node_tracker.track_success()
            else:
                node_tracker.track_error()

            node_summaries[node_key] = node_tracker.get_summary()
        return node_summaries

    def get_agent_graph_runner(self) -> AgentGraphRunner:
        """
        Return the underlying AgentGraphRunner for advanced use.

        :return: The AgentGraphRunner instance
        """
        return self._runner
