"""Types for chat responses."""

from dataclasses import dataclass
from typing import Any, List, Optional

from ldai.metrics import LDAIMetrics
from ldai.models import LDMessage


@dataclass
class ChatResponse:
    """
    Chat response structure.
    """
    message: LDMessage
    metrics: LDAIMetrics
    evaluations: Optional[List[Any]] = None  # List of JudgeResponse, will be populated later
