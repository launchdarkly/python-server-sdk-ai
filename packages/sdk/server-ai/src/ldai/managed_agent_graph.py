"""ManagedAgentGraph — LaunchDarkly managed wrapper for agent graph execution."""

from typing import Any, Optional

from ldai.providers import AgentGraphResult, AgentGraphRunner
from ldai.providers.types import (
    AgentGraphRunnerResult,
    GraphMetricSummary,
    LDAIMetrics,
    ManagedGraphResult,
)


class ManagedAgentGraph:
    """
    LaunchDarkly managed wrapper for AI agent graph execution.

    Holds an AgentGraphRunner and an optional AgentGraphDefinition. Wraps the
    runner result in a :class:`~ldai.providers.types.ManagedGraphResult` and
    builds a :class:`~ldai.providers.types.GraphMetricSummary` from the runner's
    metrics.

    When the runner returns an :class:`~ldai.providers.types.AgentGraphRunnerResult`
    (new shape), the managed layer drives all graph-level tracking from
    ``result.metrics``.  When the runner returns the legacy
    :class:`~ldai.providers.AgentGraphResult`, tracking has already been performed
    inside the runner; the managed layer simply wraps the result.  This detection
    branch exists as a deliberate bridge: once PR 11-openai and PR 11-langchain
    migrate both runners to return ``AgentGraphRunnerResult``, the legacy branch
    becomes dead code and will be removed in PR 11-langchain's final cleanup commit.

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
        :param graph: Optional AgentGraphDefinition used to create the
            graph-level tracker when the runner returns an
            :class:`AgentGraphRunnerResult` (new shape).  Not needed for
            legacy runners that still return :class:`AgentGraphResult`.
        """
        self._runner = runner
        self._graph = graph

    async def run(self, input: Any) -> ManagedGraphResult:
        """
        Run the agent graph with the given input.

        Delegates to the underlying AgentGraphRunner.  The returned type
        determines which tracking path is taken:

        - :class:`AgentGraphRunnerResult` (new shape): the managed layer drives
          graph-level tracking from ``result.metrics`` via the graph tracker.
          Per-node tracking from ``result.metrics.node_metrics`` will be wired
          in a follow-up commit once the runners populate ``node_metrics``.
        - :class:`AgentGraphResult` (legacy shape): tracking already occurred
          inside the runner; the managed layer wraps the result without
          additional tracking.

        :param input: The input prompt or structured input for the graph
        :return: ManagedGraphResult containing the content, metric summary,
            raw response, and an optional evaluations task (always ``None``
            for now — per-graph evaluations will be added in a future PR).
        """
        raw_result = await self._runner.run(input)

        if isinstance(raw_result, AgentGraphRunnerResult):
            # New shape: managed layer drives all tracking.
            summary = self._build_summary_from_runner_result(raw_result)
            if self._graph is not None:
                self._flush_graph_tracking(raw_result, self._graph.create_tracker())
            return ManagedGraphResult(
                content=raw_result.content,
                metrics=summary,
                raw=raw_result.raw,
                evaluations=None,
            )

        # Legacy shape (AgentGraphResult): tracking already happened in the runner.
        # Build a GraphMetricSummary from the runner result's LDAIMetrics.
        # path and node_metrics will be populated once graph runners are migrated
        # to return AgentGraphRunnerResult with GraphMetrics (PR 11-openai/langchain).
        metrics: LDAIMetrics = raw_result.metrics
        summary = GraphMetricSummary(
            success=metrics.success,
            usage=metrics.usage,
            duration_ms=getattr(metrics, 'duration_ms', None),
        )
        return ManagedGraphResult(
            content=raw_result.output,
            metrics=summary,
            raw=raw_result.raw,
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

        Called only when the runner returns the new ``AgentGraphRunnerResult``
        shape.  Node-level tracking (from ``result.metrics.node_metrics``) will
        be wired once the runners start populating that field.
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

    def get_agent_graph_runner(self) -> AgentGraphRunner:
        """
        Return the underlying AgentGraphRunner for advanced use.

        :return: The AgentGraphRunner instance
        """
        return self._runner
