from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from ldai import LDMessage
from ldai.models import AIConfigKind
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
