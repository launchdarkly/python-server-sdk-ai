"""Feedback kind enumeration for AI operations."""

from enum import Enum


class FeedbackKind(Enum):
    """
    Types of feedback that can be provided for AI operations.
    """

    Positive = "positive"
    Negative = "negative"
