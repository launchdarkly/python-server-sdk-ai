"""Types for AI provider responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ldai.models import LDMessage
from ldai.tracker import TokenUsage

# Type alias for a registry of tools available to an agent.
# Keys are tool names; values are the callable implementations.
ToolRegistry = Dict[str, Callable]


@dataclass
class LDAIMetrics:
    """
    Metrics information for AI operations that includes success status and token usage.
    """
    success: bool
    usage: Optional[TokenUsage] = None

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
        return result


@dataclass
class ModelResponse:
    """
    Response from a model invocation.
    """
    message: LDMessage
    metrics: LDAIMetrics
    evaluations: Optional[List[JudgeResult]] = None


@dataclass
class StructuredResponse:
    """
    Structured response from AI models.
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
    score: Optional[float] = None
    reasoning: Optional[str] = None
    metric_key: Optional[str] = None

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
