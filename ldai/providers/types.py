"""Types for AI provider responses."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ldai.models import LDMessage
from ldai.tracker import TokenUsage


@dataclass
class LDAIMetrics:
    """
    Metrics information for AI operations that includes success status and token usage.
    """
    success: bool
    usage: Optional[TokenUsage] = None


@dataclass
class ChatResponse:
    """
    Chat response structure.
    """
    message: LDMessage
    metrics: LDAIMetrics
    evaluations: Optional[List[Any]] = None  # List of JudgeResponse, will be populated later


@dataclass
class StructuredResponse:
    """
    Structured response from AI models.
    """
    data: Dict[str, Any]
    raw_response: str
    metrics: LDAIMetrics


@dataclass
class EvalScore:
    """
    Score and reasoning for a single evaluation metric.
    """
    score: float  # Score between 0.0 and 1.0
    reasoning: str  # Reasoning behind the provided score


@dataclass
class JudgeResponse:
    """
    Response from a judge evaluation containing scores and reasoning for multiple metrics.
    """
    evals: Dict[str, EvalScore]  # Dictionary where keys are metric names and values contain score and reasoning
    success: bool  # Whether the evaluation completed successfully
    error: Optional[str] = None  # Error message if evaluation failed

