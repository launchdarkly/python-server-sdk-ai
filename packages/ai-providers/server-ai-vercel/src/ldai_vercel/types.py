"""Types for Vercel AI provider."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ldai import LDMessage

# Type alias for provider function
VercelProviderFunction = Callable[[str], Any]


@dataclass
class VercelModelParameters:
    """
    Vercel/LiteLLM model parameters.

    These are the parameters that can be passed to LiteLLM methods.
    """
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result: Dict[str, Any] = {}
        if self.max_tokens is not None:
            result['max_tokens'] = self.max_tokens
        if self.temperature is not None:
            result['temperature'] = self.temperature
        if self.top_p is not None:
            result['top_p'] = self.top_p
        if self.top_k is not None:
            result['top_k'] = self.top_k
        if self.presence_penalty is not None:
            result['presence_penalty'] = self.presence_penalty
        if self.frequency_penalty is not None:
            result['frequency_penalty'] = self.frequency_penalty
        if self.stop is not None:
            result['stop'] = self.stop
        if self.seed is not None:
            result['seed'] = self.seed
        return result


@dataclass
class VercelSDKMapOptions:
    """Options for mapping to Vercel/LiteLLM SDK configuration."""
    non_interpolated_messages: Optional[List[LDMessage]] = None


@dataclass
class VercelSDKConfig:
    """Configuration format compatible with LiteLLM's completion methods."""
    model: str
    messages: Optional[List[LDMessage]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result: Dict[str, Any] = {'model': self.model}
        if self.messages is not None:
            result['messages'] = [{'role': m.role, 'content': m.content} for m in self.messages]
        if self.max_tokens is not None:
            result['max_tokens'] = self.max_tokens
        if self.temperature is not None:
            result['temperature'] = self.temperature
        if self.top_p is not None:
            result['top_p'] = self.top_p
        if self.top_k is not None:
            result['top_k'] = self.top_k
        if self.presence_penalty is not None:
            result['presence_penalty'] = self.presence_penalty
        if self.frequency_penalty is not None:
            result['frequency_penalty'] = self.frequency_penalty
        if self.stop is not None:
            result['stop'] = self.stop
        if self.seed is not None:
            result['seed'] = self.seed
        return result


@dataclass
class ModelUsageTokens:
    """
    Token usage information from LiteLLM operations.
    """
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class TextResponse:
    """Response type for non-streaming LiteLLM operations."""
    finish_reason: Optional[str] = None
    usage: Optional[ModelUsageTokens] = None


@dataclass
class StreamResponse:
    """Response type for streaming LiteLLM operations."""
    # Note: In async streaming, these would be resolved after the stream completes
    finish_reason: Optional[str] = None
    usage: Optional[ModelUsageTokens] = None
