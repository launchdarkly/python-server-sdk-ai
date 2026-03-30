from typing import Any, Iterable, List, Optional, Tuple, cast

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


# Native tool raw_item type names don't always match the LD config key convention.
_NATIVE_TOOL_TYPE_TO_CONFIG_KEY = {
    'web_search': 'web_search_tool',
    'file_search': 'file_search_tool',
}


def get_tool_calls_from_run_items(new_items: List[Any]) -> List[Tuple[str, str]]:
    """
    Extract (agent_name, tool_name) pairs from RunResult.new_items.

    Covers both custom FunctionTools (tracked by their config key) and native
    hosted tools (web search, file search, code interpreter, image generation).

    :param new_items: The new_items list from a RunResult
    :return: List of (agent_name, tool_name) tuples
    """
    try:
        from agents.items import ToolCallItem
        from openai.types.responses import ResponseFunctionToolCall
    except ImportError:
        return []

    result = []
    for item in new_items:
        if not isinstance(item, ToolCallItem):
            continue
        agent_name = getattr(item.agent, 'name', None)
        if not agent_name:
            continue
        raw = item.raw_item
        if isinstance(raw, ResponseFunctionToolCall):
            # Custom FunctionTools are registered as 'tool_{config_key}'
            tool_name = raw.name.removeprefix('tool_')
        else:
            raw_type = getattr(raw, 'type', None) or (raw.get('type') if isinstance(raw, dict) else None)
            if not raw_type:
                continue
            tool_name = _NATIVE_TOOL_TYPE_TO_CONFIG_KEY.get(raw_type, raw_type)
        if tool_name:
            result.append((agent_name, tool_name))
    return result
