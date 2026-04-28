"""Types for AI provider responses."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ldai.models import LDMessage
from ldai.tracker import LDAIMetricSummary, TokenUsage

# Type alias for a registry of tools available to an agent.
# Keys are tool names; values are the callable implementations.
ToolRegistry = Dict[str, Callable]


@dataclass
class LDAIMetrics:
    """
    Metrics information for AI operations that includes success status, token
    usage, and optional enrichment fields populated by runners.

    ``tool_calls`` is a list of tool-call names observed during the invocation
    (populated by agent runners that execute tool loops).

    ``duration_ms`` is the wall-clock duration of the runner invocation in
    milliseconds, when measured by the runner itself rather than externally.
    When set, the tracker uses this value directly instead of measuring elapsed
    time.
    """
    success: bool
    usage: Optional[TokenUsage] = None
    tool_calls: Optional[List[str]] = None
    duration_ms: Optional[int] = None

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
    """
    Result returned by a :class:`~ldai.providers.runner.Runner` from a single
    invocation.

    This is the unified return type for all Runner implementations.
    ``evaluations`` is intentionally absent — judge evaluations are dispatched
    by the managed layer and live on :class:`ManagedResult`.
    """
    content: str
    metrics: LDAIMetrics
    raw: Optional[Any] = None
    parsed: Optional[Dict[str, Any]] = None


@dataclass
class ManagedResult:
    """
    Result returned by the managed layer (:class:`~ldai.ManagedModel` /
    :class:`~ldai.ManagedAgent`) after a single invocation.

    ``metrics`` is an :class:`~ldai.tracker.LDAIMetricSummary` (from
    ``tracker.get_summary()``) rather than a raw :class:`LDAIMetrics`.
    ``evaluations`` is an optional asyncio Task that resolves to a list of
    :class:`JudgeResult` instances when awaited.
    """
    content: str
    metrics: LDAIMetricSummary
    raw: Optional[Any] = None
    parsed: Optional[Dict[str, Any]] = None
    evaluations: Optional[asyncio.Task[List[JudgeResult]]] = None


@dataclass
class ModelResponse:
    """
    Response from a model invocation.

    .. deprecated::
        Use :class:`RunnerResult` (from a runner) and :class:`ManagedResult`
        (from the managed layer) instead.
    """
    message: LDMessage
    metrics: LDAIMetrics
    evaluations: Optional[asyncio.Task[List[JudgeResult]]] = None


@dataclass
class StructuredResponse:
    """
    Structured response from AI models.

    .. deprecated::
        Structured output is now represented by :attr:`RunnerResult.parsed`.
    """
    data: Dict[str, Any]
    raw_response: str
    metrics: LDAIMetrics


@dataclass
class JudgeResult:
    """
    Result from a judge evaluation.
    """
    judge_config_key: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None
    sampled: bool = False  # True when the evaluation was sampled and run
    metric_key: Optional[str] = None
    score: Optional[float] = None
    reasoning: Optional[str] = None

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
class AgentResult:
    """
    Result from a single-agent run.

    .. deprecated::
        Use :class:`ManagedResult` (managed layer) or :class:`RunnerResult`
        (runner layer) instead.
    """
    output: str
    raw: Any
    metrics: LDAIMetrics


@dataclass
class AgentGraphResult:
    """
    Result from an agent graph run.
    """
    output: str
    raw: Any
    metrics: LDAIMetrics
    evaluations: Optional[List[JudgeResult]] = None
