"""Metrics tracking for AI operations."""

from typing import Any, Dict, Optional

from ldai.metrics.feedback_kind import FeedbackKind
from ldai.metrics.token_usage import TokenUsage


class LDAIMetricSummary:
    """
    Summary of metrics which have been tracked.
    """

    def __init__(self):
        self._duration = None
        self._success = None
        self._feedback = None
        self._usage = None
        self._time_to_first_token = None

    @property
    def duration(self) -> Optional[int]:
        return self._duration

    @property
    def success(self) -> Optional[bool]:
        return self._success

    @property
    def feedback(self) -> Optional[Dict[str, FeedbackKind]]:
        return self._feedback

    @property
    def usage(self) -> Optional[TokenUsage]:
        return self._usage

    @property
    def time_to_first_token(self) -> Optional[int]:
        return self._time_to_first_token


class LDAIMetrics:
    """
    Metrics information for AI operations that includes success status and token usage.
    """

    def __init__(self, success: bool, usage: Optional[TokenUsage] = None):
        """
        Initialize metrics.

        :param success: Whether the operation was successful.
        :param usage: Optional token usage information.
        """
        self.success = success
        self.usage = usage

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
