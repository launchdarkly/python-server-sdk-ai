import inspect
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ldai import LDMessage, log
from ldai.models import AIConfigKind
from ldai.providers import ToolRegistry
from ldai.providers.types import LDAIMetrics
from ldai.tracker import TokenUsage


def map_provider(ld_provider_name: str) -> str:
    """
    Map a LaunchDarkly provider name to its LangChain equivalent.

    :param ld_provider_name: LaunchDarkly provider name
    :return: LangChain-compatible provider name
    """
    lowercased_name = ld_provider_name.lower()
    # Bedrock is the only provider that uses "provider:model_family" (e.g. Bedrock:Anthropic).
    if lowercased_name.startswith('bedrock:'):
        return 'bedrock_converse'

    mapping: Dict[str, str] = {
        'gemini': 'google-genai',
        'bedrock': 'bedrock_converse',
    }
    return mapping.get(lowercased_name, lowercased_name)


def convert_messages_to_langchain(
    messages: List[LDMessage],
) -> List[Union[HumanMessage, SystemMessage, AIMessage]]:
    """
    Convert LaunchDarkly messages to LangChain message objects.

    :param messages: List of LDMessage objects
    :return: List of LangChain message objects
    :raises ValueError: If an unsupported message role is encountered
    """
    result: List[Union[HumanMessage, SystemMessage, AIMessage]] = []
    for msg in messages:
        if msg.role == 'system':
            result.append(SystemMessage(content=msg.content))
        elif msg.role == 'user':
            result.append(HumanMessage(content=msg.content))
        elif msg.role == 'assistant':
            result.append(AIMessage(content=msg.content))
        else:
            raise ValueError(f'Unsupported message role: {msg.role}')
    return result


def create_langchain_model(ai_config: AIConfigKind) -> BaseChatModel:
    """
    Create a LangChain BaseChatModel from a LaunchDarkly AI configuration.

    :param ai_config: The LaunchDarkly AI configuration
    :return: A configured LangChain BaseChatModel
    """
    from langchain.chat_models import init_chat_model

    config_dict = ai_config.to_dict()
    model_dict = config_dict.get('model') or {}
    provider_dict = config_dict.get('provider') or {}

    model_name = model_dict.get('name', '')
    provider = provider_dict.get('name', '')
    parameters = dict(model_dict.get('parameters') or {})
    parameters.pop('tools', None)
    mapped_provider = map_provider(provider)

    # Bedrock requires the foundation provider (e.g. Bedrock:Anthropic) passed in
    # parameters separately from model_provider, which is used for LangChain routing.
    if mapped_provider == 'bedrock_converse' and 'provider' not in parameters:
        parameters['provider'] = provider.removeprefix('bedrock:')

    return init_chat_model(
        model_name,
        model_provider=mapped_provider,
        **parameters,
    )


def _iter_valid_tools(
    tool_definitions: List[Dict[str, Any]],
    tool_registry: ToolRegistry,
) -> List[tuple]:
    """
    Filter LD tool definitions against a registry, returning (name, td) pairs for each
    valid function tool that has a callable implementation. Built-in provider tools and
    tools missing from the registry are skipped with a warning.
    """
    valid = []
    for td in tool_definitions:
        if not isinstance(td, dict):
            continue

        tool_type = td.get('type')
        if tool_type and tool_type != 'function':
            log.warning(
                f"Built-in tool '{tool_type}' is not reliably supported via LangChain and will be skipped. "
                "Use a provider-specific runner to use built-in provider tools."
            )
            continue

        name = td.get('name')
        if not name:
            continue

        if name not in tool_registry:
            log.warning(f"Tool '{name}' is defined in the AI config but was not found in the tool registry; skipping.")
            continue

        valid.append((name, td))

    return valid


def build_tools(ai_config: AIConfigKind, tool_registry: ToolRegistry) -> List[Any]:
    """
    Return callables from the registry for each tool defined in the AI config.

    Tools not found in the registry are skipped with a warning. The returned
    callables can be passed directly to bind_tools or langchain.agents.create_agent.
    Functions should have type-annotated parameters so LangChain can infer the schema.

    :param ai_config: The LaunchDarkly AI configuration
    :param tool_registry: Registry mapping tool names to callable implementations
    :return: List of callables ready to pass to bind_tools or create_agent
    """
    config_dict = ai_config.to_dict()
    model_dict = config_dict.get('model') or {}
    parameters = dict(model_dict.get('parameters') or {})
    tool_definitions = parameters.pop('tools', []) or []

    tools = []
    for td in tool_definitions:
        if not isinstance(td, dict):
            continue
        name = td.get('name')
        if not name:
            continue
        fn = tool_registry.get(name)
        if fn is None:
            log.warning(f"Tool '{name}' is defined in the AI config but was not found in the tool registry; skipping.")
            continue
        tools.append(fn)
    return tools


def build_structured_tools(ai_config: AIConfigKind, tool_registry: ToolRegistry) -> List[Any]:
    """
    Build a list of LangChain StructuredTool instances from LD tool definitions and a registry.

    Tools found in the registry are wrapped as StructuredTool using the LD config key as the
    tool name so the model's tool calls match ToolNode lookup. Async callables use ``coroutine=``
    so LangGraph invokes them correctly. Built-in provider tools and tools missing from the
    registry are skipped with a warning.

    :param ai_config: The LaunchDarkly AI configuration
    :param tool_registry: Registry mapping tool names to callable implementations
    :return: List of StructuredTool instances ready to pass to langchain.agents.create_agent
    """
    from langchain_core.tools import StructuredTool

    config_dict = ai_config.to_dict()
    model_dict = config_dict.get('model') or {}
    parameters = dict(model_dict.get('parameters') or {})
    tool_definitions = parameters.pop('tools', []) or []

    tools = []
    for name, td in _iter_valid_tools(tool_definitions, tool_registry):
        fn = tool_registry[name]
        raw_desc = td.get('description') if isinstance(td.get('description'), str) else ''
        description = raw_desc.strip() or (getattr(fn, '__doc__', None) or '').strip() or f'Tool {name}'
        if inspect.iscoroutinefunction(fn):
            tool = StructuredTool.from_function(coroutine=fn, name=name, description=description)
        else:
            tool = StructuredTool.from_function(fn, name=name, description=description)
        tools.append(tool)
    return tools


def get_ai_usage_from_response(response: Any) -> Optional[TokenUsage]:
    """
    Extract token usage from a LangChain response.

    :param response: The response from a LangChain model (BaseMessage or similar)
    :return: TokenUsage or None if unavailable
    """
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        total = response.usage_metadata.get('total_tokens', 0)
        inp = response.usage_metadata.get('input_tokens', 0)
        out = response.usage_metadata.get('output_tokens', 0)
        if total or inp or out:
            return TokenUsage(total=total, input=inp, output=out)
    if hasattr(response, 'response_metadata') and response.response_metadata:
        token_usage = (
            response.response_metadata.get('tokenUsage')
            or response.response_metadata.get('token_usage')
        )
        if token_usage:
            return TokenUsage(
                total=token_usage.get('totalTokens', 0) or token_usage.get('total_tokens', 0),
                input=token_usage.get('promptTokens', 0) or token_usage.get('prompt_tokens', 0),
                output=token_usage.get('completionTokens', 0) or token_usage.get('completion_tokens', 0),
            )
    return None


def get_ai_metrics_from_response(response: Any) -> LDAIMetrics:
    """
    Extract LaunchDarkly AI metrics from a LangChain response.

    :param response: The response from a LangChain model (BaseMessage or similar)
    :return: LDAIMetrics with success status and token usage
    """
    return LDAIMetrics(success=True, usage=get_ai_usage_from_response(response))


def get_tool_calls_from_response(response: Any) -> List[str]:
    """
    Get tool call names from a LangChain provider response.

    :param response: The response from the LangChain model
    :return: List of tool names in order, or empty list if none
    """
    names: List[str] = []
    if hasattr(response, 'tool_calls') and isinstance(response.tool_calls, list):
        for tc in response.tool_calls:
            n = tc.get('name')
            if n:
                names.append(str(n))
    return names


def extract_last_message_content(messages: List[Any]) -> str:
    """
    Extract the string content of the last message in a list.

    :param messages: List of LangChain message objects
    :return: String content of the last message, or empty string if none or content is not a str
    """
    if messages:
        last = messages[-1]
        if hasattr(last, 'content') and isinstance(last.content, str):
            return last.content
    return ''


def sum_token_usage_from_messages(messages: List[Any]) -> Optional[TokenUsage]:
    """
    Sum token usage across LangChain messages using get_ai_usage_from_response per message.

    :param messages: List of message objects (e.g. from a graph state)
    :return: Aggregated TokenUsage, or None if no usage on any message
    """
    in_sum = 0
    out_sum = 0
    total_sum = 0
    for m in messages:
        u = get_ai_usage_from_response(m)
        if u is None:
            continue
        in_sum += u.input
        out_sum += u.output
        total_sum += u.total
    if in_sum == 0 and out_sum == 0 and total_sum == 0:
        return None
    return TokenUsage(total=total_sum, input=in_sum, output=out_sum)
