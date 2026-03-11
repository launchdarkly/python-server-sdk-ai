"""Shared OpenAI utilities for the LaunchDarkly AI SDK."""

from typing import Any, Iterable, List, Optional, cast

from ldai import LDMessage
from ldai.providers.types import LDAIMetrics
from ldai.tracker import TokenUsage
from openai.types.chat import ChatCompletionMessageParam


class OpenAIHelper:
    """
    Shared utilities for OpenAI-based runners (model, agent, agent graph).

    All methods are static — this class is a namespace, not meant to be instantiated.
    """

    @staticmethod
    def convert_messages(messages: List[LDMessage]) -> Iterable[ChatCompletionMessageParam]:
        """
        Convert LaunchDarkly messages to OpenAI chat completion message format.

        :param messages: List of LDMessage objects
        :return: Iterable of OpenAI ChatCompletionMessageParam dicts
        """
        return cast(
            Iterable[ChatCompletionMessageParam],
            [{'role': msg.role, 'content': msg.content} for msg in messages],
        )

    @staticmethod
    def get_ai_metrics_from_response(response: Any) -> LDAIMetrics:
        """
        Extract LaunchDarkly AI metrics from an OpenAI response.

        :param response: The response from the OpenAI chat completions API
        :return: LDAIMetrics with success status and token usage
        """
        usage: Optional[TokenUsage] = None
        if hasattr(response, 'usage') and response.usage:
            usage = TokenUsage(
                total=response.usage.total_tokens or 0,
                input=response.usage.prompt_tokens or 0,
                output=response.usage.completion_tokens or 0,
            )
        return LDAIMetrics(success=True, usage=usage)
