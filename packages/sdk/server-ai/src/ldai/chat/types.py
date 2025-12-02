"""Types for chat responses."""

from dataclasses import dataclass
from typing import Any, List, Optional

from ldai.config.types import LDMessage
from ldai.metrics import LDAIMetrics


@dataclass
class ChatResponse:
    """
    Chat response structure.
    """
    message: LDMessage
    metrics: LDAIMetrics
    evaluations: Optional[List[Any]] = None  # List of JudgeResponse, will be populated later
