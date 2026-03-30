from typing import Any, Dict, Iterable, List, Optional, cast

from ldai import LDMessage
from ldai.providers.types import LDAIMetrics
from ldai.tracker import TokenUsage
from openai.types.chat import ChatCompletionMessageParam


def _build_native_tool_map() -> Dict[str, Any]:
    try:
        from agents import (
            CodeInterpreterTool,
            FileSearchTool,
            ImageGenerationTool,
            WebSearchTool,
        )
        return {
            'web_search_tool': lambda _: WebSearchTool(),
            'file_search_tool': lambda _: FileSearchTool(),
            'code_interpreter': lambda _: CodeInterpreterTool(),
            'image_generation': lambda _: ImageGenerationTool(),
        }
    except ImportError:
        return {}


NATIVE_OPENAI_TOOLS: Dict[str, Any] = _build_native_tool_map()


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

    Handles both chat completions responses (``response.usage``) and
    openai-agents ``RunResult`` objects (``response.context_wrapper.usage``).
    Field names are normalised: agents SDK uses ``input_tokens``/``output_tokens``
    while the chat completions API uses ``prompt_tokens``/``completion_tokens``.

    :param response: An OpenAI chat completions response or openai-agents RunResult
    :return: TokenUsage or None if unavailable
    """
    usage = None
    try:
        usage = response.context_wrapper.usage
    except Exception:
        pass

    if usage is None and hasattr(response, 'usage') and response.usage:
        usage = response.usage

    if usage is None:
        return None

    total = getattr(usage, 'total_tokens', None) or 0
    inp = getattr(usage, 'input_tokens', None) or getattr(usage, 'prompt_tokens', None) or 0
    out = getattr(usage, 'output_tokens', None) or getattr(usage, 'completion_tokens', None) or 0

    if total or inp or out:
        return TokenUsage(total=total, input=inp, output=out)
    return None


def get_ai_metrics_from_response(response: Any) -> LDAIMetrics:
    """
    Extract LaunchDarkly AI metrics from an OpenAI response.

    :param response: An OpenAI chat completions response or openai-agents RunResult
    :return: LDAIMetrics with success status and token usage
    """
    return LDAIMetrics(success=True, usage=get_ai_usage_from_response(response))
