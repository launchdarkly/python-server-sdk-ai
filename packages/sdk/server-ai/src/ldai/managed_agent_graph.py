"""ManagedAgentGraph — LaunchDarkly managed wrapper for agent graph execution."""

from typing import Any, Optional

from ldai.providers import AgentGraphRunner
from ldai.providers.types import (
    AgentGraphRunnerResult,
    GraphMetricSummary,
    ManagedGraphResult,
)


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
        runner: AgentGraphRunner,
        graph: Optional[Any] = None,
    ):
        """
        Initialize ManagedAgentGraph.

        :param runner: The AgentGraphRunner to delegate execution to
        :param graph: Optional AgentGraphDefinition used to drive graph-level and
            per-node tracking from the runner result metrics.
        """
        self._runner = runner
        self._graph = graph

    async def run(self, input: Any) -> ManagedGraphResult:
        """
        Run the agent graph with the given input.

        Delegates to the underlying AgentGraphRunner, then drives all
        LaunchDarkly tracking from ``result.metrics``:

        - Graph-level events (path, duration, success/failure, total tokens) via
          the graph tracker obtained from the graph definition.
        - Per-node events (tokens, duration, tool calls, success) via per-node
          trackers for each key present in ``result.metrics.node_metrics``.

        :param input: The input prompt or structured input for the graph
        :return: ManagedGraphResult containing the content, metric summary,
            and raw response.
        """
        result = await self._runner.run(input)

        summary = self._build_summary_from_runner_result(result)

        if self._graph is not None:
            graph_tracker = self._graph.create_tracker()
            self._flush_graph_tracking(result, graph_tracker)
            self._flush_node_tracking(result)

        return ManagedGraphResult(
            content=result.content,
            metrics=summary,
            raw=result.raw,
            evaluations=None,
        )

    def _build_summary_from_runner_result(
        self,
        result: AgentGraphRunnerResult,
    ) -> GraphMetricSummary:
        """Build a GraphMetricSummary from an AgentGraphRunnerResult."""
        m = result.metrics
        return GraphMetricSummary(
            success=m.success,
            path=list(m.path),
            duration_ms=m.duration_ms,
            usage=m.usage,
            node_metrics=dict(m.node_metrics),
        )

    def _flush_graph_tracking(self, result: AgentGraphRunnerResult, tracker: Any) -> None:
        """
        Drive graph-level LaunchDarkly tracking events from runner result metrics.
        """
        m = result.metrics
        if m.path:
            tracker.track_path(m.path)
        if m.duration_ms is not None:
            tracker.track_duration(m.duration_ms)
        if m.success:
            tracker.track_invocation_success()
        else:
            tracker.track_invocation_failure()
        if m.usage is not None:
            tracker.track_total_tokens(m.usage)

    def _flush_node_tracking(self, result: AgentGraphRunnerResult) -> None:
        """
        Drive per-node LaunchDarkly tracking events from ``result.metrics.node_metrics``.

        For each node key present in ``node_metrics``, obtains the node's
        config tracker via the graph definition and fires token, duration,
        tool call, and success/error events.
        """
        if self._graph is None:
            return

        for node_key, node_ldai_metrics in result.metrics.node_metrics.items():
            node = self._graph.get_node(node_key)
            if node is None:
                continue
            node_tracker = node.get_config().create_tracker()
            if node_tracker is None:
                continue

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

    def get_agent_graph_runner(self) -> AgentGraphRunner:
        """
        Return the underlying AgentGraphRunner for advanced use.

        :return: The AgentGraphRunner instance
        """
        return self._runner
