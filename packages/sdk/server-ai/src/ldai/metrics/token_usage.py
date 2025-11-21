"""Token usage tracking for AI operations."""

from dataclasses import dataclass


@dataclass
class TokenUsage:
    """
    Tracks token usage for AI operations.

    :param total: Total number of tokens used.
    :param input: Number of tokens in the prompt.
    :param output: Number of tokens in the completion.
    """

    total: int
    input: int
    output: int
