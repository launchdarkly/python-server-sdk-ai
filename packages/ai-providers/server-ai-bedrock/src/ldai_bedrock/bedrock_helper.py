from typing import Any, Dict, List, Optional

from ldai import LDMessage
from ldai.providers.types import LDAIMetrics
from ldai.tracker import TokenUsage


def map_provider(ld_provider_name: str) -> str:
    """
    Map a LaunchDarkly provider name to its Bedrock-equivalent identifier.

    Amazon Bedrock is routed by ``modelId`` rather than a provider string,
    so this is effectively an identity for ``bedrock``.  The helper exists
    to keep parity with the LangChain and OpenAI providers, which both
    expose a similar mapping function.

    :param ld_provider_name: LaunchDarkly provider name
    :return: The Bedrock provider identifier (always ``"bedrock"``)
    """
    lowercased_name = ld_provider_name.lower()
    if lowercased_name == 'bedrock' or lowercased_name.startswith('bedrock:'):
        return 'bedrock'
    return lowercased_name


def convert_messages_to_bedrock(messages: List[LDMessage]) -> Dict[str, Any]:
    """
    Convert LaunchDarkly messages into the Bedrock Converse API shape.

    The Converse API splits the conversation into two top-level fields:
    a ``system`` list aggregating all system prompts, and a ``messages``
    list containing user/assistant turns.  Each non-system message has
    a ``role`` plus a ``content`` array of typed content blocks
    (``{"text": "..."}``).

    :param messages: List of LDMessage objects
    :return: Dict with ``system`` and ``messages`` keys ready to pass to
        ``bedrock-runtime.converse(...)``.  The ``system`` key is absent
        when no system prompts are present.
    :raises ValueError: When a message has an unsupported role.
    """
    system_blocks: List[Dict[str, Any]] = []
    bedrock_messages: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.role == 'system':
            system_blocks.append({'text': msg.content})
        elif msg.role in ('user', 'assistant'):
            bedrock_messages.append({
                'role': msg.role,
                'content': [{'text': msg.content}],
            })
        else:
            raise ValueError(f'Unsupported message role: {msg.role}')

    result: Dict[str, Any] = {'messages': bedrock_messages}
    if system_blocks:
        result['system'] = system_blocks
    return result


def convert_tools_to_bedrock(tool_definitions: List[Any]) -> Optional[Dict[str, Any]]:
    """
    Convert LaunchDarkly tool definitions into Bedrock Converse ``toolConfig``.

    Bedrock expects ``toolConfig`` to be a dict containing a ``tools`` list
    where each entry is wrapped in a ``toolSpec`` envelope.  Each ``toolSpec``
    has ``name``, ``description``, and ``inputSchema.json`` keys.  The LD
    config layout matches this closely so most fields map through directly.

    :param tool_definitions: Tool definitions from the LD AI config
    :return: A ``toolConfig`` dict ready to pass to ``converse(...)``, or
        ``None`` when no usable tools are provided.
    """
    tool_specs: List[Dict[str, Any]] = []
    for td in tool_definitions:
        if not isinstance(td, dict):
            continue
        name = td.get('name')
        if not name:
            continue
        spec: Dict[str, Any] = {'name': name}
        description = td.get('description')
        if description is not None:
            spec['description'] = description
        parameters = td.get('parameters')
        if parameters is not None:
            spec['inputSchema'] = {'json': parameters}
        tool_specs.append({'toolSpec': spec})

    if not tool_specs:
        return None
    return {'tools': tool_specs}


def get_ai_usage_from_response(response: Any) -> Optional[TokenUsage]:
    """
    Extract token usage from a Bedrock Converse response.

    :param response: Response dict returned by ``bedrock-runtime.converse(...)``
    :return: TokenUsage or None if usage data is unavailable
    """
    if not isinstance(response, dict):
        return None
    usage = response.get('usage')
    if not isinstance(usage, dict):
        return None
    total = usage.get('totalTokens') or 0
    inp = usage.get('inputTokens') or 0
    out = usage.get('outputTokens') or 0
    if not (total or inp or out):
        return None
    return TokenUsage(total=total, input=inp, output=out)


def get_ai_metrics_from_response(response: Any) -> LDAIMetrics:
    """
    Extract LaunchDarkly AI metrics from a Bedrock Converse response.

    ``success`` is derived from the HTTP status in ``ResponseMetadata``: a
    200 response is treated as success, anything else as failure.  Token
    usage, request duration, and observed tool-call names are populated
    when present in the response.

    :param response: Response dict returned by ``bedrock-runtime.converse(...)``
    :return: LDAIMetrics with success, tokens, duration_ms, and tool_calls
    """
    if not isinstance(response, dict):
        return LDAIMetrics(success=False)

    status_code = (response.get('ResponseMetadata') or {}).get('HTTPStatusCode')
    success = status_code == 200

    tokens = get_ai_usage_from_response(response)

    duration_ms: Optional[int] = None
    metrics_block = response.get('metrics')
    if isinstance(metrics_block, dict):
        latency = metrics_block.get('latencyMs')
        if isinstance(latency, (int, float)):
            duration_ms = int(latency)

    tool_calls = _extract_tool_calls(response)

    return LDAIMetrics(
        success=success,
        tokens=tokens,
        tool_calls=tool_calls if tool_calls else None,
        duration_ms=duration_ms,
    )


def _extract_tool_calls(response: Dict[str, Any]) -> List[str]:
    """Return the names of any ``toolUse`` content blocks in the response."""
    output = response.get('output')
    if not isinstance(output, dict):
        return []
    message = output.get('message')
    if not isinstance(message, dict):
        return []
    content = message.get('content')
    if not isinstance(content, list):
        return []
    names: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        tool_use = block.get('toolUse')
        if isinstance(tool_use, dict):
            name = tool_use.get('name')
            if isinstance(name, str):
                names.append(name)
    return names


def extract_content_from_response(response: Any) -> str:
    """
    Pull the first textual content block out of a Bedrock Converse response.

    :param response: Response dict returned by ``bedrock-runtime.converse(...)``
    :return: The concatenated text from the first message's text blocks, or
        an empty string when no text content is available.
    """
    if not isinstance(response, dict):
        return ''
    output = response.get('output')
    if not isinstance(output, dict):
        return ''
    message = output.get('message')
    if not isinstance(message, dict):
        return ''
    content = message.get('content')
    if not isinstance(content, list):
        return ''
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get('text')
        if isinstance(text, str) and text:
            parts.append(text)
    return ''.join(parts)
