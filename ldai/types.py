from enum import Enum
from typing import Any, Callable, List, Literal, Optional
from dataclasses import dataclass

from ldai.tracker import LDAIConfigTracker

@dataclass
class TokenMetrics():
    total: int
    input: int
    output: int # type: ignore

@dataclass
class LDMessage():
    role: Literal['system', 'user', 'assistant']
    content: str

@dataclass
class AIConfigData():
    model: Optional[dict]
    prompt: Optional[List[LDMessage]]
    _ldMeta: dict

class AIConfig():
    def __init__(self, config: AIConfigData, tracker: LDAIConfigTracker, enabled: bool):
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
class LDOpenAIUsage():
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

@dataclass
class OpenAITokenUsage:
    def __init__(self, data: LDOpenAIUsage):
        self.total_tokens = data.total_tokens
        self.prompt_tokens = data.prompt_tokens
        self.completion_tokens = data.completion_tokens

    def to_metrics(self) -> TokenMetrics:
        return TokenMetrics(
            total=self.total_tokens,
            input=self.prompt_tokens,
            output=self.completion_tokens,
        )
 
@dataclass
class BedrockTokenUsage:
    def __init__(self, data: dict):
        self.totalTokens = data.get('totalTokens', 0)
        self.inputTokens = data.get('inputTokens', 0)
        self.outputTokens = data.get('outputTokens', 0)

    def to_metrics(self) -> TokenMetrics:
        return TokenMetrics(
            total=self.totalTokens,
            input=self.inputTokens,
            output=self.outputTokens,
        )