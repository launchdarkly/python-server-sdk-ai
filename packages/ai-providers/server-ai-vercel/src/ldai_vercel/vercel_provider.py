"""Vercel AI implementation of AIProvider for LaunchDarkly AI SDK using LiteLLM."""

import json
from typing import Any, Callable, Dict, List, Optional, Union

import litellm
from litellm import acompletion

from ldai import LDMessage
from ldai.models import AIConfigKind
from ldai.providers import AIProvider
from ldai.providers.types import ChatResponse, LDAIMetrics, StructuredResponse
from ldai.tracker import TokenUsage

from ldai_vercel.types import (
    ModelUsageTokens,
    TextResponse,
    VercelModelParameters,
    VercelProviderFunction,
    VercelSDKConfig,
    VercelSDKMapOptions,
)


class VercelProvider(AIProvider):
    """
    Vercel AI implementation of AIProvider using LiteLLM.

    This provider integrates multiple AI providers (OpenAI, Anthropic, Google, etc.)
    with LaunchDarkly's tracking capabilities through LiteLLM.
    """

    def __init__(
        self,
        model_name: str,
        parameters: VercelModelParameters,
        logger: Optional[Any] = None
    ):
        """
        Initialize the Vercel provider.

        :param model_name: The full model name in LiteLLM format (e.g., 'openai/gpt-4', 'anthropic/claude-3-opus')
        :param parameters: Model parameters
        :param logger: Optional logger for logging provider operations
        """
        super().__init__(logger)
        self._model_name = model_name
        self._parameters = parameters

    # =============================================================================
    # MAIN FACTORY METHODS
    # =============================================================================

    @staticmethod
    async def create(ai_config: AIConfigKind, logger: Optional[Any] = None) -> 'VercelProvider':
        """
        Static factory method to create a Vercel AIProvider from an AI configuration.
        This method auto-detects the provider and creates the model.

        :param ai_config: The LaunchDarkly AI configuration
        :param logger: Optional logger
        :return: A configured VercelProvider
        """
        model_name = VercelProvider.create_model_name(ai_config)
        parameters = VercelProvider.map_parameters(ai_config.to_dict().get('model', {}).get('parameters'))
        return VercelProvider(model_name, parameters, logger)

    # =============================================================================
    # INSTANCE METHODS (AIProvider Implementation)
    # =============================================================================

    async def invoke_model(self, messages: List[LDMessage]) -> ChatResponse:
        """
        Invoke the AI model with an array of messages.

        :param messages: Array of LDMessage objects representing the conversation
        :return: ChatResponse containing the model's response and metrics
        """
        try:
            # Convert LDMessage to LiteLLM message format
            litellm_messages = [
                {'role': msg.role, 'content': msg.content}
                for msg in messages
            ]

            # Call LiteLLM acompletion
            response = await acompletion(
                model=self._model_name,
                messages=litellm_messages,
                **self._parameters.to_dict(),
            )

            # Extract metrics including token usage and success status
            metrics = VercelProvider.get_ai_metrics_from_response(response)

            # Create the assistant message
            content = ''
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content

            return ChatResponse(
                message=LDMessage(role='assistant', content=content),
                metrics=metrics,
            )
        except Exception as error:
            if self.logger:
                self.logger.warn(f'Vercel AI model invocation failed: {error}')

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
        Invoke the AI model with structured output support.

        :param messages: Array of LDMessage objects representing the conversation
        :param response_structure: Dictionary defining the JSON schema for output structure
        :return: StructuredResponse containing the structured data
        """
        try:
            # Convert LDMessage to LiteLLM message format
            litellm_messages = [
                {'role': msg.role, 'content': msg.content}
                for msg in messages
            ]

            # Call LiteLLM acompletion with JSON response format
            response = await acompletion(
                model=self._model_name,
                messages=litellm_messages,
                response_format={'type': 'json_object'},
                **self._parameters.to_dict(),
            )

            # Extract metrics
            metrics = VercelProvider.get_ai_metrics_from_response(response)

            # Safely extract the content
            content = ''
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content

            if not content:
                if self.logger:
                    self.logger.warn('Vercel AI structured response has no content available')
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
                    self.logger.warn(f'Vercel AI structured response contains invalid JSON: {parse_error}')
                metrics = LDAIMetrics(success=False, usage=metrics.usage)
                return StructuredResponse(
                    data={},
                    raw_response=content,
                    metrics=metrics,
                )
        except Exception as error:
            if self.logger:
                self.logger.warn(f'Vercel AI structured model invocation failed: {error}')

            return StructuredResponse(
                data={},
                raw_response='',
                metrics=LDAIMetrics(success=False, usage=None),
            )

    def get_model_name(self) -> str:
        """
        Get the model name.

        :return: The model name
        """
        return self._model_name

    # =============================================================================
    # STATIC UTILITY METHODS
    # =============================================================================

    @staticmethod
    def map_provider(ld_provider_name: str) -> str:
        """
        Map LaunchDarkly provider names to LiteLLM provider prefixes.

        This method enables seamless integration between LaunchDarkly's standardized
        provider naming and LiteLLM's naming conventions.

        :param ld_provider_name: LaunchDarkly provider name
        :return: LiteLLM-compatible provider prefix
        """
        lowercased_name = ld_provider_name.lower()

        mapping: Dict[str, str] = {
            'gemini': 'gemini',
            'google': 'gemini',
            'openai': 'openai',
            'anthropic': 'anthropic',
            'cohere': 'cohere',
            'mistral': 'mistral',
            'azure': 'azure',
            'bedrock': 'bedrock',
        }

        return mapping.get(lowercased_name, lowercased_name)

    @staticmethod
    def map_usage_data_to_ld_token_usage(usage_data: Any) -> TokenUsage:
        """
        Map LiteLLM usage data to LaunchDarkly token usage.

        :param usage_data: Usage data from LiteLLM
        :return: TokenUsage
        """
        if not usage_data:
            return TokenUsage(total=0, input=0, output=0)

        total_tokens = getattr(usage_data, 'total_tokens', None) or 0
        prompt_tokens = getattr(usage_data, 'prompt_tokens', None) or 0
        completion_tokens = getattr(usage_data, 'completion_tokens', None) or 0

        return TokenUsage(
            total=total_tokens,
            input=prompt_tokens,
            output=completion_tokens,
        )

    @staticmethod
    def get_ai_metrics_from_response(response: Any) -> LDAIMetrics:
        """
        Get AI metrics from a LiteLLM response.

        This method extracts token usage information and success status from LiteLLM responses
        and returns a LaunchDarkly AIMetrics object.

        :param response: The response from LiteLLM
        :return: LDAIMetrics with success status and token usage

        Example:
            response = await tracker.track_metrics_of(
                lambda: acompletion(config),
                VercelProvider.get_ai_metrics_from_response
            )
        """
        # Check finish reason for error
        finish_reason = 'unknown'
        if response and hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'finish_reason'):
                finish_reason = choice.finish_reason or 'unknown'

        # Extract token usage if available
        usage: Optional[TokenUsage] = None
        if hasattr(response, 'usage') and response.usage:
            usage = VercelProvider.map_usage_data_to_ld_token_usage(response.usage)

        success = finish_reason != 'error'

        return LDAIMetrics(success=success, usage=usage)

    @staticmethod
    def create_ai_metrics(response: Any) -> LDAIMetrics:
        """
        Create AI metrics information from a LiteLLM response.

        :deprecated: Use `get_ai_metrics_from_response()` instead.
        :param response: The response from LiteLLM
        :return: LDAIMetrics with success status and token usage
        """
        return VercelProvider.get_ai_metrics_from_response(response)

    @staticmethod
    def map_parameters(parameters: Optional[Dict[str, Any]]) -> VercelModelParameters:
        """
        Map LaunchDarkly model parameters to LiteLLM parameters.

        Parameter mappings:
        - max_tokens → max_tokens
        - max_completion_tokens → max_tokens
        - temperature → temperature
        - top_p → top_p
        - top_k → top_k
        - presence_penalty → presence_penalty
        - frequency_penalty → frequency_penalty
        - stop → stop
        - seed → seed

        :param parameters: The LaunchDarkly model parameters to map
        :return: VercelModelParameters
        """
        if not parameters:
            return VercelModelParameters()

        return VercelModelParameters(
            max_tokens=parameters.get('max_tokens') or parameters.get('max_completion_tokens'),
            temperature=parameters.get('temperature'),
            top_p=parameters.get('top_p'),
            top_k=parameters.get('top_k'),
            presence_penalty=parameters.get('presence_penalty'),
            frequency_penalty=parameters.get('frequency_penalty'),
            stop=parameters.get('stop'),
            seed=parameters.get('seed'),
        )

    @staticmethod
    def to_litellm_config(
        ai_config: AIConfigKind,
        options: Optional[VercelSDKMapOptions] = None,
    ) -> VercelSDKConfig:
        """
        Convert an AI configuration to LiteLLM configuration.

        :param ai_config: The LaunchDarkly AI configuration
        :param options: Optional mapping options
        :return: A configuration directly usable in LiteLLM
        """
        config_dict = ai_config.to_dict()
        model_dict = config_dict.get('model') or {}
        provider_dict = config_dict.get('provider') or {}

        # Build full model name
        provider_name = VercelProvider.map_provider(provider_dict.get('name', ''))
        model_name = model_dict.get('name', '')

        full_model_name = f'{provider_name}/{model_name}' if provider_name else model_name

        # Merge messages from config and options
        messages: Optional[List[LDMessage]] = None
        config_messages = config_dict.get('messages')
        if config_messages or (options and options.non_interpolated_messages):
            messages = []
            if config_messages:
                for msg in config_messages:
                    messages.append(LDMessage(role=msg['role'], content=msg['content']))
            if options and options.non_interpolated_messages:
                messages.extend(options.non_interpolated_messages)

        # Map parameters using the shared mapping method
        params = VercelProvider.map_parameters(model_dict.get('parameters'))

        # Build and return the LiteLLM configuration
        return VercelSDKConfig(
            model=full_model_name,
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            stop=params.stop,
            seed=params.seed,
        )

    @staticmethod
    def create_model_name(ai_config: AIConfigKind) -> str:
        """
        Create a LiteLLM model name from an AI configuration.

        :param ai_config: The LaunchDarkly AI configuration
        :return: A LiteLLM-compatible model name
        """
        config_dict = ai_config.to_dict()
        model_dict = config_dict.get('model') or {}
        provider_dict = config_dict.get('provider') or {}

        provider_name = VercelProvider.map_provider(provider_dict.get('name', ''))
        model_name = model_dict.get('name', '')

        # LiteLLM uses provider/model format
        if provider_name:
            return f'{provider_name}/{model_name}'
        return model_name

