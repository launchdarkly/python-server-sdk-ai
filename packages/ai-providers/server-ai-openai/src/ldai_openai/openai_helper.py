from typing import Any, Iterable, List, Optional, cast

from ldai import LDMessage
from ldai.providers.types import LDAIMetrics
from ldai.tracker import TokenUsage
from openai.types.chat import ChatCompletionMessageParam


def convert_messages_to_openai(messages: List[LDMessage]) -> Iterable[ChatCompletionMessageParam]:
    """
    Convert LaunchDarkly messages to OpenAI chat completion message format.

    :param messages: List of LDMessage objects
    :return: Iterable of OpenAI ChatCompletionMessageParam dicts
    """
    return cast(
        Iterable[ChatCompletionMessageParam],
        [{'role': msg.role, 'content': msg.content} for msg in messages],
    )


def get_ai_usage_from_response(response: Any) -> Optional[TokenUsage]:
    """
    Extract token usage from an OpenAI response.

    :param response: The response from the OpenAI chat completions API
    :return: TokenUsage or None if unavailable
    """
    if hasattr(response, 'usage') and response.usage:
        u = response.usage
        return TokenUsage(
            total=getattr(u, 'total_tokens', None) or 0,
            input=getattr(u, 'prompt_tokens', None) or 0,
            output=getattr(u, 'completion_tokens', None) or 0,
        )
    return None


def get_ai_metrics_from_response(response: Any) -> LDAIMetrics:
    """
    Extract LaunchDarkly AI metrics from an OpenAI response.

    :param response: The response from the OpenAI chat completions API
    :return: LDAIMetrics with success status and token usage
    """
    return LDAIMetrics(success=True, usage=get_ai_usage_from_response(response))
