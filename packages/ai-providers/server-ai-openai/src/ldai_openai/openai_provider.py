"""OpenAI implementation of AIProvider for LaunchDarkly AI SDK."""

import json
import os
from typing import Any, Dict, Iterable, List, Optional, cast

from ldai import LDMessage
from ldai.models import AIConfigKind
from ldai.providers import AIProvider
from ldai.providers.types import ChatResponse, LDAIMetrics, StructuredResponse
from ldai.tracker import TokenUsage
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam


class OpenAIProvider(AIProvider):
    """
    OpenAI implementation of AIProvider.

    This provider integrates OpenAI's chat completions API with LaunchDarkly's tracking capabilities.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        parameters: Dict[str, Any],
        logger: Optional[Any] = None
    ):
        """
        Initialize the OpenAI provider.

        :param client: An AsyncOpenAI client instance
        :param model_name: The name of the model to use
        :param parameters: Additional model parameters
        :param logger: Optional logger for logging provider operations
        """
        super().__init__(logger)
        self._client = client
        self._model_name = model_name
        self._parameters = parameters

    # =============================================================================
    # MAIN FACTORY METHOD
    # =============================================================================

    @staticmethod
    async def create(ai_config: AIConfigKind, logger: Optional[Any] = None) -> 'OpenAIProvider':
        """
        Static factory method to create an OpenAI AIProvider from an AI configuration.

        :param ai_config: The LaunchDarkly AI configuration
        :param logger: Optional logger for the provider
        :return: Configured OpenAIProvider instance
        """
        client = AsyncOpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
        )

        config_dict = ai_config.to_dict()
        model_dict = config_dict.get('model') or {}
        model_name = model_dict.get('name', '')
        parameters = model_dict.get('parameters') or {}

        return OpenAIProvider(client, model_name, parameters, logger)

    # =============================================================================
    # INSTANCE METHODS (AIProvider Implementation)
    # =============================================================================

    async def invoke_model(self, messages: List[LDMessage]) -> ChatResponse:
        """
        Invoke the OpenAI model with an array of messages.

        :param messages: Array of LDMessage objects representing the conversation
        :return: ChatResponse containing the model's response and metrics
        """
        try:
            # Convert LDMessage to OpenAI message format
            openai_messages: Iterable[ChatCompletionMessageParam] = cast(
                Iterable[ChatCompletionMessageParam],
                [{'role': msg.role, 'content': msg.content} for msg in messages]
            )

            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                **self._parameters,
            )

            # Generate metrics early (assumes success by default)
            metrics = OpenAIProvider.get_ai_metrics_from_response(response)

            # Safely extract the first choice content
            content = ''
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content

            if not content:
                if self.logger:
                    self.logger.warn('OpenAI response has no content available')
                metrics = LDAIMetrics(success=False, usage=metrics.usage)

            return ChatResponse(
                message=LDMessage(role='assistant', content=content),
                metrics=metrics,
            )
        except Exception as error:
            if self.logger:
                self.logger.warn(f'OpenAI model invocation failed: {error}')

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
        Invoke the OpenAI model with structured output support.

        :param messages: Array of LDMessage objects representing the conversation
        :param response_structure: Dictionary defining the JSON schema for output structure
        :return: StructuredResponse containing the structured data
        """
        try:
            # Convert LDMessage to OpenAI message format
            openai_messages: Iterable[ChatCompletionMessageParam] = cast(
                Iterable[ChatCompletionMessageParam],
                [{'role': msg.role, 'content': msg.content} for msg in messages]
            )

            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                response_format={  # type: ignore[arg-type]
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'structured_output',
                        'schema': response_structure,
                        'strict': True,
                    },
                },
                **self._parameters,
            )

            # Generate metrics early (assumes success by default)
            metrics = OpenAIProvider.get_ai_metrics_from_response(response)

            # Safely extract the first choice content
            content = ''
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content

            if not content:
                if self.logger:
                    self.logger.warn('OpenAI structured response has no content available')
                metrics = LDAIMetrics(success=False, usage=metrics.usage)
                return StructuredResponse(
                    data={},
                    raw_response='',
                    metrics=metrics,
                )

            try:
                data = json.loads(content)
                return StructuredResponse(
                    data=data,
                    raw_response=content,
                    metrics=metrics,
                )
            except json.JSONDecodeError as parse_error:
                if self.logger:
                    self.logger.warn(f'OpenAI structured response contains invalid JSON: {parse_error}')
                metrics = LDAIMetrics(success=False, usage=metrics.usage)
                return StructuredResponse(
                    data={},
                    raw_response=content,
                    metrics=metrics,
                )
        except Exception as error:
            if self.logger:
                self.logger.warn(f'OpenAI structured model invocation failed: {error}')

            return StructuredResponse(
                data={},
                raw_response='',
                metrics=LDAIMetrics(success=False, usage=None),
            )

    def get_client(self) -> AsyncOpenAI:
        """
        Get the underlying OpenAI client instance.

        :return: The underlying AsyncOpenAI client
        """
        return self._client

    # =============================================================================
    # STATIC UTILITY METHODS
    # =============================================================================

    @staticmethod
    def get_ai_metrics_from_response(response: Any) -> LDAIMetrics:
        """
        Get AI metrics from an OpenAI response.

        This method extracts token usage information and success status from OpenAI responses
        and returns a LaunchDarkly AIMetrics object.

        :param response: The response from OpenAI chat completions API
        :return: LDAIMetrics with success status and token usage

        Example:
            response = await tracker.track_metrics_of(
                lambda: client.chat.completions.create(config),
                OpenAIProvider.get_ai_metrics_from_response
            )
        """
        # Extract token usage if available
        usage: Optional[TokenUsage] = None
        if hasattr(response, 'usage') and response.usage:
            usage = TokenUsage(
                total=response.usage.total_tokens or 0,
                input=response.usage.prompt_tokens or 0,
                output=response.usage.completion_tokens or 0,
            )

        # OpenAI responses that complete successfully are considered successful by default
        return LDAIMetrics(success=True, usage=usage)
