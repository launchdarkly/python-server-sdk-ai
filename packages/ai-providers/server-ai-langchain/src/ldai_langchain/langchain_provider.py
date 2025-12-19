"""LangChain implementation of AIProvider for LaunchDarkly AI SDK."""

from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from ldai.models import AIConfigKind, LDMessage
from ldai.providers.ai_provider import AIProvider
from ldai.providers.types import ChatResponse, LDAIMetrics, StructuredResponse
from ldai.tracker import TokenUsage


class LangChainProvider(AIProvider):
    """
    LangChain implementation of AIProvider.
    
    This provider integrates LangChain models with LaunchDarkly's tracking capabilities.
    """

    def __init__(self, llm: BaseChatModel, logger: Optional[Any] = None):
        """
        Initialize the LangChain provider.

        :param llm: A LangChain BaseChatModel instance
        :param logger: Optional logger for logging provider operations
        """
        super().__init__(logger)
        self._llm = llm


    @staticmethod
    async def create(ai_config: AIConfigKind, logger: Optional[Any] = None) -> 'LangChainProvider':
        """
        Static factory method to create a LangChain AIProvider from an AI configuration.

        :param ai_config: The LaunchDarkly AI configuration
        :param logger: Optional logger for the provider
        :return: Configured LangChainProvider instance
        """
        llm = await LangChainProvider.create_langchain_model(ai_config)
        return LangChainProvider(llm, logger)

    async def invoke_model(self, messages: List[LDMessage]) -> ChatResponse:
        """
        Invoke the LangChain model with an array of messages.

        :param messages: Array of LDMessage objects representing the conversation
        :return: ChatResponse containing the model's response and metrics
        """
        try:
            # Convert LDMessage[] to LangChain messages
            langchain_messages = LangChainProvider.convert_messages_to_langchain(messages)

            # Get the LangChain response
            response: BaseMessage = await self._llm.ainvoke(langchain_messages)

            # Generate metrics early (assumes success by default)
            metrics = LangChainProvider.get_ai_metrics_from_response(response)

            # Extract text content from the response
            content: str = ''
            if isinstance(response.content, str):
                content = response.content
            else:
                # Log warning for non-string content (likely multimodal)
                if self.logger:
                    self.logger.warn(
                        f'Multimodal response not supported, expecting a string. '
                        f'Content type: {type(response.content)}, Content: {response.content}'
                    )
                # Update metrics to reflect content loss
                metrics = LDAIMetrics(success=False, usage=metrics.usage)

            # Create the assistant message
            assistant_message = LDMessage(role='assistant', content=content)

            return ChatResponse(
                message=assistant_message,
                metrics=metrics,
            )
        except Exception as error:
            if self.logger:
                self.logger.warn(f'LangChain model invocation failed: {error}')

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
            # Convert LDMessage[] to LangChain messages
            langchain_messages = LangChainProvider.convert_messages_to_langchain(messages)

            # Get the LangChain response with structured output
            structured_llm = self._llm.with_structured_output(response_structure)
            response = await structured_llm.ainvoke(langchain_messages)

            # Using structured output doesn't support metrics
            metrics = LDAIMetrics(
                success=True,
                usage=TokenUsage(total=0, input=0, output=0),
            )

            # Handle response serialization
            if isinstance(response, dict):
                raw_response = str(response)
            else:
                import json
                try:
                    raw_response = json.dumps(response)
                except (TypeError, ValueError):
                    raw_response = str(response)

            return StructuredResponse(
                data=response if isinstance(response, dict) else {'result': response},
                raw_response=raw_response,
                metrics=metrics,
            )
        except Exception as error:
            if self.logger:
                self.logger.warn(f'LangChain structured model invocation failed: {error}')

            return StructuredResponse(
                data={},
                raw_response='',
                metrics=LDAIMetrics(
                    success=False,
                    usage=TokenUsage(total=0, input=0, output=0),
                ),
            )

    def get_chat_model(self) -> BaseChatModel:
        """
        Get the underlying LangChain model instance.

        :return: The underlying BaseChatModel
        """
        return self._llm

    # =============================================================================
    # STATIC UTILITY METHODS
    # =============================================================================

    @staticmethod
    def map_provider(ld_provider_name: str) -> str:
        """
        Map LaunchDarkly provider names to LangChain provider names.
        
        This method enables seamless integration between LaunchDarkly's standardized
        provider naming and LangChain's naming conventions.

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
        Get AI metrics from a LangChain provider response.
        
        This method extracts token usage information and success status from LangChain responses
        and returns a LaunchDarkly AIMetrics object.

        :param response: The response from the LangChain model
        :return: LDAIMetrics with success status and token usage

        Example:
            # Use with tracker.track_metrics_of for automatic tracking
            response = await tracker.track_metrics_of(
                lambda: llm.ainvoke(messages),
                LangChainProvider.get_ai_metrics_from_response
            )
        """
        # Extract token usage if available
        usage: Optional[TokenUsage] = None
        if hasattr(response, 'response_metadata') and response.response_metadata:
            token_usage = response.response_metadata.get('tokenUsage') or response.response_metadata.get('token_usage')
            if token_usage:
                usage = TokenUsage(
                    total=token_usage.get('totalTokens', 0) or token_usage.get('total_tokens', 0),
                    input=token_usage.get('promptTokens', 0) or token_usage.get('prompt_tokens', 0),
                    output=token_usage.get('completionTokens', 0) or token_usage.get('completion_tokens', 0),
                )

        # LangChain responses that complete successfully are considered successful by default
        return LDAIMetrics(success=True, usage=usage)

    @staticmethod
    def create_ai_metrics(langchain_response: BaseMessage) -> LDAIMetrics:
        """
        Create AI metrics information from a LangChain provider response.
        
        .. deprecated::
            Use `get_ai_metrics_from_response()` instead.

        :param langchain_response: The response from the LangChain model
        :return: LDAIMetrics with success status and token usage
        """
        return LangChainProvider.get_ai_metrics_from_response(langchain_response)

    @staticmethod
    def convert_messages_to_langchain(
        messages: List[LDMessage],
    ) -> List[Union[HumanMessage, SystemMessage, AIMessage]]:
        """
        Convert LaunchDarkly messages to LangChain messages.
        
        This helper method enables developers to work directly with LangChain message types
        while maintaining compatibility with LaunchDarkly's standardized message format.

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
    async def create_langchain_model(ai_config: AIConfigKind) -> BaseChatModel:
        """
        Create a LangChain model from an AI configuration.
        
        This public helper method enables developers to initialize their own LangChain models
        using LaunchDarkly AI configurations.

        :param ai_config: The LaunchDarkly AI configuration
        :return: A configured LangChain BaseChatModel
        """
        from langchain.chat_models import init_chat_model

        model_name = ai_config.model.name if ai_config.model else ''
        provider = ai_config.provider.name if ai_config.provider else ''
        parameters = {}
        
        if ai_config.model and hasattr(ai_config.model, '_parameters') and ai_config.model._parameters:
            parameters = ai_config.model._parameters.copy()

        # Use LangChain's universal init_chat_model to support multiple providers
        return init_chat_model(
            model_name,
            model_provider=LangChainProvider.map_provider(provider),
            **parameters,
        )
