"""Types for AI provider responses."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ldai.tracker import LDAIMetricSummary, TokenUsage

# Type alias for a registry of tools available to an agent.
# Keys are tool names; values are the callable implementations.
ToolRegistry = Dict[str, Callable]


@dataclass
class LDAIMetrics:
    """Contains metrics for a single AI invocation."""

    success: bool
    """Whether the invocation succeeded."""

    usage: Optional[TokenUsage] = None
    """Optional token usage information."""

    tool_calls: Optional[List[str]] = None
    """Ordered list of tool-call names observed during the invocation."""

    duration_ms: Optional[int] = None
    """Wall-clock duration of the runner invocation in milliseconds."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the metrics as a dictionary object.
        """
        result: Dict[str, Any] = {
            'success': self.success,
        }
        if self.usage is not None:
            result['usage'] = {
                'total': self.usage.total,
                'input': self.usage.input,
                'output': self.usage.output,
            }
        if self.tool_calls is not None:
            result['toolCalls'] = self.tool_calls
        if self.duration_ms is not None:
            result['durationMs'] = self.duration_ms
        return result


@dataclass
class RunnerResult:
    """Contains the result of a single AI model invocation."""

    content: str
    """The text content returned by the model."""

    metrics: LDAIMetrics
    """Metrics for this invocation."""

    raw: Optional[Any] = None
    """Optional provider-native response object for advanced consumers."""

    parsed: Optional[Dict[str, Any]] = None
    """Optional parsed structured output, populated when ``output_type`` was supplied."""


@dataclass
class ManagedResult:
    """Contains the result of a managed AI invocation, including metrics and optional judge evaluations."""

    content: str
    """The text content returned by the model."""

    metrics: LDAIMetricSummary
    """Aggregated metric summary from the tracker for this invocation."""

    raw: Optional[Any] = None
    """Optional provider-native response object for advanced consumers."""

    parsed: Optional[Dict[str, Any]] = None
    """Optional parsed structured output, populated when ``output_type`` was supplied."""

    evaluations: Optional[asyncio.Task[List[JudgeResult]]] = None
    """Optional asyncio Task that resolves to the list of :class:`JudgeResult` instances when awaited."""


@dataclass
class GraphMetrics:
    """Contains raw metrics from a single agent graph run."""

    success: bool
    """Whether the graph run succeeded."""

    path: List[str] = field(default_factory=list)
    """Ordered list of node keys visited during the run."""

    duration_ms: Optional[int] = None
    """Wall-clock duration of the graph run in milliseconds."""

    usage: Optional[TokenUsage] = None
    """Optional aggregate token usage information across all nodes in the graph run."""

    node_metrics: Dict[str, LDAIMetrics] = field(default_factory=dict)
    """Per-node metrics keyed by node key."""


@dataclass
class GraphMetricSummary:
    """Contains a summary of metrics for an agent graph run."""

    success: bool
    """Whether the graph run succeeded."""

    path: List[str] = field(default_factory=list)
    """Ordered list of node keys visited during the run."""

    duration_ms: Optional[int] = None
    """Wall-clock duration of the graph run in milliseconds."""

    usage: Optional[TokenUsage] = None
    """Optional aggregate token usage information across all nodes in the graph run."""

    node_metrics: Dict[str, LDAIMetrics] = field(default_factory=dict)
    """Per-node metrics keyed by node key."""

    resumption_token: Optional[str] = None
    """Optional resumption token from the graph tracker for cross-process resumption."""


@dataclass
class ManagedGraphResult:
    """Contains the result of a managed agent graph run, including metrics and optional judge evaluations."""

    content: str
    """The graph's final output content."""

    metrics: GraphMetricSummary
    """Aggregated metric summary from the graph tracker for this run."""

    raw: Optional[Any] = None
    """Optional provider-native response object for advanced consumers."""

    evaluations: Optional[asyncio.Task[List[JudgeResult]]] = None
    """Optional asyncio Task that resolves to the list of :class:`JudgeResult` instances when awaited."""


@dataclass
class AgentGraphRunnerResult:
    """Contains the result of an agent graph runner invocation."""

    content: str
    """The graph's final output content."""

    metrics: GraphMetrics
    """Metrics from the graph run."""

    raw: Optional[Any] = None
    """Optional provider-native response object for advanced consumers."""


@dataclass
class JudgeResult:
    """Contains the result of a single judge evaluation."""

    judge_config_key: Optional[str] = None
    """The configuration key of the judge that produced this result."""

    success: bool = False
    """Whether the judge evaluation completed successfully."""

    error_message: Optional[str] = None
    """Error message describing why the evaluation failed, if any."""

    sampled: bool = False
    """True when the evaluation was sampled and run."""

    metric_key: Optional[str] = None
    """The metric key under which this judge's score is reported."""

    score: Optional[float] = None
    """The numeric score (0-1) returned by the judge."""

    reasoning: Optional[str] = None
    """The judge's reasoning text accompanying the score."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the judge result as a dictionary object.
        """
        result: Dict[str, Any] = {
            'success': self.success,
            'sampled': self.sampled,
        }
        if self.score is not None:
            result['score'] = self.score
        if self.reasoning is not None:
            result['reasoning'] = self.reasoning
        if self.metric_key is not None:
            result['metricKey'] = self.metric_key
        if self.judge_config_key is not None:
            result['judgeConfigKey'] = self.judge_config_key
        if self.error_message is not None:
            result['errorMessage'] = self.error_message
        return result


@dataclass
class AgentGraphResult:
    """Contains the result of an agent graph run."""

    output: str
    """The agent graph's final output content."""

    raw: Any
    """The provider-native response object from the graph run."""

    metrics: LDAIMetrics
    """Metrics recorded during the graph run."""

    evaluations: Optional[List[JudgeResult]] = None
    """Optional list of judge evaluation results produced for the graph run."""
