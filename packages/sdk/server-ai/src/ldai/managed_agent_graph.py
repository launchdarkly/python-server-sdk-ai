"""ManagedAgentGraph — LaunchDarkly managed wrapper for agent graph execution."""

import asyncio
from typing import Any, List

from ldai.providers import AgentGraphResult, AgentGraphRunner
from ldai.providers.types import GraphMetricSummary, JudgeResult, ManagedGraphResult


class ManagedAgentGraph:
    """
    LaunchDarkly managed wrapper for AI agent graph execution.

    Holds an AgentGraphRunner. Wraps the runner result in a
    :class:`~ldai.providers.types.ManagedGraphResult` and builds a
    :class:`~ldai.providers.types.GraphMetricSummary` from the runner's metrics.

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

    async def run(self, input: Any) -> ManagedGraphResult:
        """
        Run the agent graph with the given input.

        :param input: The input prompt or structured input for the graph
        :return: ManagedGraphResult containing the content, metric summary, raw response,
            and an optional evaluations task (currently always ``None`` for graphs —
            per-graph evaluations will be added in a future PR).
        """
        result: AgentGraphResult = await self._runner.run(input)

        # Build a GraphMetricSummary from the runner result's LDAIMetrics.
        # path and node_metrics will be populated once graph runners are migrated
        # to return AgentGraphRunnerResult with GraphMetrics (PR 11).
        metrics = result.metrics
        summary = GraphMetricSummary(
            success=metrics.success,
            usage=metrics.usage,
            duration_ms=getattr(metrics, 'duration_ms', None),
        )

        return ManagedGraphResult(
            content=result.output,
            metrics=summary,
            raw=result.raw,
            evaluations=None,
        )

    def get_agent_graph_runner(self) -> AgentGraphRunner:
        """
        Return the underlying AgentGraphRunner for advanced use.

        :return: The AgentGraphRunner instance
        """
        return self._runner
