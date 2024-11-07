from enum import Enum
from typing import Callable
from dataclasses import dataclass

@dataclass
class TokenMetrics():
    total: int
    input: int
    output: int # type: ignore

@dataclass
class AIConfigData():
    config: dict
    prompt: any
    _ldMeta: dict

class AITracker():
    track_duration: Callable[..., None]
    track_tokens: Callable[..., None]
    track_error: Callable[..., None]
    track_generation: Callable[..., None]
    track_feedback: Callable[..., None]

class AIConfig():
    def __init__(self, config: AIConfigData, tracker: AITracker, enabled: bool):
        self.config = config
        self.tracker = tracker
        self.enabled = enabled

@dataclass
class FeedbackKind(Enum):
    Positive = "positive"
    Negative = "negative"

@dataclass
class TokenUsage():
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

    def to_metrics(self):
        return {
            'total': self['total_tokens'],
            'input': self['prompt_tokens'],
            'output': self['completion_tokens'],
        }

@dataclass
class OpenAITokenUsage:
    def __init__(self, data: any):
        self.total_tokens = data.total_tokens
        self.prompt_tokens = data.prompt_tokens
        self.completion_tokens = data.completion_tokens

    def to_metrics(self) -> TokenMetrics:
        return {
            'total': self.total_tokens,
            'input': self.prompt_tokens,
            'output': self.completion_tokens,
        }
 
class BedrockTokenUsage:
    def __init__(self, data: dict):
        self.totalTokens = data.get('totalTokens', 0)
        self.inputTokens = data.get('inputTokens', 0)
        self.outputTokens = data.get('outputTokens', 0)

    def to_metrics(self) -> TokenMetrics:
        return {
            'total': self.totalTokens,
            'input': self.inputTokens,
            'output': self.outputTokens,
        }