from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

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

    Handles both chat completions responses (``response.usage``) and
    openai-agents ``RunResult`` objects (``response.context_wrapper.usage``).

    :param response: An OpenAI chat completions response or openai-agents RunResult
    :return: TokenUsage or None if unavailable
    """
    try:
        usage = response.context_wrapper.usage
        if usage is not None:
            total = getattr(usage, 'total_tokens', None) or 0
            inp = getattr(usage, 'input_tokens', None) or 0
            out = getattr(usage, 'output_tokens', None) or 0
            if total or inp or out:
                return TokenUsage(total=total, input=inp, output=out)
    except Exception:
        pass

    usage = getattr(response, 'usage', None)
    if usage is not None:
        total = getattr(usage, 'total_tokens', None) or 0
        inp = getattr(usage, 'prompt_tokens', None) or 0
        out = getattr(usage, 'completion_tokens', None) or 0
        if total or inp or out:
            return TokenUsage(total=total, input=inp, output=out)

    return None


def extract_usage_from_request_entry(entry: Any) -> Optional[TokenUsage]:
    """
    Extract token usage from a single request_usage_entry in an openai-agents RunResult.

    :param entry: A request_usage_entry from context_wrapper.usage.request_usage_entries
    :return: TokenUsage or None if extraction fails
    """
    try:
        return TokenUsage(
            total=entry.total_tokens,
            input=entry.input_tokens,
            output=entry.output_tokens,
        )
    except Exception:
        return None


def get_ai_metrics_from_response(response: Any) -> LDAIMetrics:
    """
    Extract LaunchDarkly AI metrics from an OpenAI response.

    :param response: An OpenAI chat completions response or openai-agents RunResult
    :return: LDAIMetrics with success status and token usage
    """
    return LDAIMetrics(success=True, usage=get_ai_usage_from_response(response))


# Tool names that require their own API type in the Chat Completions API.
# LD stores all tools as type="function"; these are converted to their correct type.
_NATIVE_API_TOOL_NAMES = frozenset({
    'web_search_tool',
    'file_search',
    'computer_use_preview',
})


def normalize_tool_types(tool_definitions: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert LD tool definitions to Chat Completions API format.

    LD emits all tools as ``type="function"`` with a flat structure. This helper
    wraps regular function tools in the nested ``function`` key the API requires,
    and converts known native tool names to their correct API type without a schema.

    :param tool_definitions: Tool definitions from the LD AI config
    :return: Tool list ready to pass to ``chat.completions.create``
    """
    result = []
    for td in tool_definitions:
        if not isinstance(td, dict):
            continue
        name = td.get('name', '')
        result.append({**td, 'type': name} if name in _NATIVE_API_TOOL_NAMES else td)
    return result


# Native tool raw_item type names don't always match the LD config key convention.
_NATIVE_TOOL_TYPE_TO_CONFIG_KEY = {
    'web_search': 'web_search_tool',
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
            tool_name = raw.name
        else:
            raw_type = getattr(raw, 'type', None) or (raw.get('type') if isinstance(raw, dict) else None)
            if not raw_type:
                continue
            tool_name = _NATIVE_TOOL_TYPE_TO_CONFIG_KEY.get(raw_type, raw_type)
        if tool_name:
            result.append((agent_name, tool_name))
    return result
