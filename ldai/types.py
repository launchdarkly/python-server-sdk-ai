from enum import Enum
from typing import TypedDict

class FeedbackKind(Enum):
    Positive = "positive"
    Negative = "negative"

class TokenUsage(TypedDict):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

class UnderscoreTokenUsage(TypedDict):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

class BedrockTokenUsage(TypedDict):
    totalTokens: int
    inputTokens: int
    outputTokens: int

class TokenMetrics(TypedDict):
    total: int
    input: int
    output: int # type: ignore