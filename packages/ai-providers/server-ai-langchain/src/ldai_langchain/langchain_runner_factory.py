"""LangChain connector for LaunchDarkly AI SDK."""

from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from ldai import LDMessage, log
from ldai.models import AIConfigKind
from ldai.providers import AIProvider
from ldai.providers.types import ChatResponse, LDAIMetrics, StructuredResponse
from ldai.tracker import TokenUsage


class LangChainRunnerFactory(AIProvider):
    """
    LangChain connector for the LaunchDarkly AI SDK.

    Can be used in two ways:
    - Transparently via ExecutorFactory (pass ``default_ai_provider='langchain'`` to
      ``create_model()`` / ``create_chat()``).
    - Directly for full control: instantiate with a ``BaseChatModel``, then call
      ``invoke_model()`` yourself and use the static convenience methods
      (``get_ai_metrics_from_response``, ``convert_messages_to_langchain``,
      ``map_provider``, ``create_langchain_model``).
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the LangChain connector.

        When called with no arguments the connector acts as a per-provider factory
        — call ``create_model(config)`` to obtain a configured instance.

        When called with an explicit ``llm`` the connector is ready to invoke
        the model immediately.

        :param llm: A LangChain BaseChatModel instance (optional)
        """
        self._llm = llm

    # --- AIProvider factory methods ---

    def create_model(self, config: AIConfigKind) -> 'LangChainRunnerFactory':
        """
        Create a configured LangChain model connector for the given AI config.

        :param config: The LaunchDarkly AI configuration
        :return: Configured LangChainRunnerFactory ready to invoke the model
        """
        llm = LangChainRunnerFactory.create_langchain_model(config)
        return LangChainRunnerFactory(llm)

    # --- Model invocation ---

    async def invoke_model(self, messages: List[LDMessage]) -> ChatResponse:
        """
        Invoke the LangChain model with an array of messages.

        :param messages: Array of LDMessage objects representing the conversation
        :return: ChatResponse containing the model's response and metrics
        """
        try:
            langchain_messages = LangChainRunnerFactory.convert_messages_to_langchain(messages)
            response: BaseMessage = await self._llm.ainvoke(langchain_messages)
            metrics = LangChainRunnerFactory.get_ai_metrics_from_response(response)

            content: str = ''
            if isinstance(response.content, str):
                content = response.content
            else:
                log.warning(
                    f'Multimodal response not supported, expecting a string. '
                    f'Content type: {type(response.content)}, Content: {response.content}'
                )
                metrics = LDAIMetrics(success=False, usage=metrics.usage)

            return ChatResponse(
                message=LDMessage(role='assistant', content=content),
                metrics=metrics,
            )
        except Exception as error:
            log.warning(f'LangChain model invocation failed: {error}')

            return ChatResponse(
                message=LDMessage(role='assistant', content=''),
                metrics=LDAIMetrics(success=False, usage=None),
            )

    async def invoke_structured_model(
        self,
        messages: List[LDMessage],
        response_structure: Dict[str, Any],
    ) -> StructuredResponse:
        """
        Invoke the LangChain model with structured output support.

        :param messages: Array of LDMessage objects representing the conversation
        :param response_structure: Dictionary defining the output structure
        :return: StructuredResponse containing the structured data
        """
        try:
            langchain_messages = LangChainRunnerFactory.convert_messages_to_langchain(messages)
            structured_llm = self._llm.with_structured_output(response_structure)
            response = await structured_llm.ainvoke(langchain_messages)

            if not isinstance(response, dict):
                log.warning(
                    f'Structured output did not return a dict. '
                    f'Got: {type(response)}'
                )
                return StructuredResponse(
                    data={},
                    raw_response='',
                    metrics=LDAIMetrics(
                        success=False,
                        usage=TokenUsage(total=0, input=0, output=0),
                    ),
                )

            return StructuredResponse(
                data=response,
                raw_response=str(response),
                metrics=LDAIMetrics(
                    success=True,
                    usage=TokenUsage(total=0, input=0, output=0),
                ),
            )
        except Exception as error:
            log.warning(f'LangChain structured model invocation failed: {error}')

            return StructuredResponse(
                data={},
                raw_response='',
                metrics=LDAIMetrics(
                    success=False,
                    usage=TokenUsage(total=0, input=0, output=0),
                ),
            )

    # --- Convenience accessors ---

    def get_chat_model(self) -> Optional[BaseChatModel]:
        """
        Get the underlying LangChain model instance.

        :return: The underlying BaseChatModel, or None if not yet configured
        """
        return self._llm

    @staticmethod
    def map_provider(ld_provider_name: str) -> str:
        """
        Map LaunchDarkly provider names to LangChain provider names.

        :param ld_provider_name: LaunchDarkly provider name
        :return: LangChain-compatible provider name
        """
        lowercased_name = ld_provider_name.lower()

        mapping: Dict[str, str] = {
            'gemini': 'google-genai',
        }

        return mapping.get(lowercased_name, lowercased_name)

    @staticmethod
    def get_ai_metrics_from_response(response: BaseMessage) -> LDAIMetrics:
        """
        Extract LaunchDarkly AI metrics from a LangChain response.

        :param response: The response from the LangChain model
        :return: LDAIMetrics with success status and token usage

        Example::

            response = await tracker.track_metrics_of(
                lambda: llm.ainvoke(messages),
                LangChainRunnerFactory.get_ai_metrics_from_response
            )
        """
        usage: Optional[TokenUsage] = None
        if hasattr(response, 'response_metadata') and response.response_metadata:
            token_usage = (
                response.response_metadata.get('tokenUsage')
                or response.response_metadata.get('token_usage')
            )
            if token_usage:
                usage = TokenUsage(
                    total=token_usage.get('totalTokens', 0) or token_usage.get('total_tokens', 0),
                    input=token_usage.get('promptTokens', 0) or token_usage.get('prompt_tokens', 0),
                    output=token_usage.get('completionTokens', 0) or token_usage.get('completion_tokens', 0),
                )

        return LDAIMetrics(success=True, usage=usage)

    @staticmethod
    def convert_messages_to_langchain(
        messages: List[LDMessage],
    ) -> List[Union[HumanMessage, SystemMessage, AIMessage]]:
        """
        Convert LaunchDarkly messages to LangChain messages.

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

    @staticmethod
    def create_langchain_model(ai_config: AIConfigKind) -> BaseChatModel:
        """
        Create a LangChain model from a LaunchDarkly AI configuration.

        :param ai_config: The LaunchDarkly AI configuration
        :return: A configured LangChain BaseChatModel
        """
        from langchain.chat_models import init_chat_model

        config_dict = ai_config.to_dict()
        model_dict = config_dict.get('model') or {}
        provider_dict = config_dict.get('provider') or {}

        model_name = model_dict.get('name', '')
        provider = provider_dict.get('name', '')
        parameters = model_dict.get('parameters') or {}

        return init_chat_model(
            model_name,
            model_provider=LangChainRunnerFactory.map_provider(provider),
            **parameters,
        )

