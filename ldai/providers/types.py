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

