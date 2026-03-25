import json
import os
from typing import Any, Dict, Iterable, List, Optional, cast

from ldai import LDMessage, log
from ldai.models import AIConfigKind
from ldai.providers import AIProvider
from ldai.providers.types import ChatResponse, LDAIMetrics, StructuredResponse
from ldai.tracker import TokenUsage
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam


class OpenAIRunnerFactory(AIProvider):
    """
    OpenAI provider for the LaunchDarkly AI SDK.

    Can be used in two ways:
    - Transparently via RunnerFactory (pass ``default_ai_provider='openai'`` to
      ``create_model()`` / ``create_chat()``).
    - Directly for full control: instantiate with an ``AsyncOpenAI`` client,
      model name, and parameters, then call ``invoke_model()`` yourself.
    """

    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        model_name: str = '',
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OpenAI provider.

        When called with no arguments the provider reads credentials from the
        environment (``OPENAI_API_KEY``) and acts as a per-provider factory —
        call ``create_model(config)`` to obtain a configured instance.

        When called with explicit arguments the provider is ready to invoke
        the model immediately.

        :param client: An AsyncOpenAI client instance (created from env if omitted)
        :param model_name: The name of the model to use
        :param parameters: Additional model parameters
        """
        self._client = client if client is not None else AsyncOpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),
        )
        self._model_name = model_name
        self._parameters = parameters or {}

    # --- AIProvider factory methods ---

    def create_model(self, config: AIConfigKind) -> 'OpenAIRunnerFactory':
        """
        Create a configured OpenAI model provider for the given AI config.

        Reuses the underlying AsyncOpenAI client so that connection pooling is
        preserved across calls.

        :param config: The LaunchDarkly AI configuration
        :return: Configured OpenAIRunnerFactory ready to invoke the model
        """
        config_dict = config.to_dict()
        model_dict = config_dict.get('model') or {}
        model_name = model_dict.get('name', '')
        parameters = model_dict.get('parameters') or {}
        return OpenAIRunnerFactory(self._client, model_name, parameters)

    # --- Model invocation ---

    async def invoke_model(self, messages: List[LDMessage]) -> ChatResponse:
        """
        Invoke the OpenAI model with an array of messages.

        :param messages: Array of LDMessage objects representing the conversation
        :return: ChatResponse containing the model's response and metrics
        """
        try:
            openai_messages: Iterable[ChatCompletionMessageParam] = cast(
                Iterable[ChatCompletionMessageParam],
                [{'role': msg.role, 'content': msg.content} for msg in messages]
            )

            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                **self._parameters,
            )

            metrics = OpenAIRunnerFactory.get_ai_metrics_from_response(response)

            content = ''
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content

            if not content:
                log.warning('OpenAI response has no content available')
                metrics = LDAIMetrics(success=False, usage=metrics.usage)

            return ChatResponse(
                message=LDMessage(role='assistant', content=content),
                metrics=metrics,
            )
        except Exception as error:
            log.warning(f'OpenAI model invocation failed: {error}')

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

            metrics = OpenAIRunnerFactory.get_ai_metrics_from_response(response)

            content = ''
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content

            if not content:
                log.warning('OpenAI structured response has no content available')
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
                log.warning(f'OpenAI structured response contains invalid JSON: {parse_error}')
                metrics = LDAIMetrics(success=False, usage=metrics.usage)
                return StructuredResponse(
                    data={},
                    raw_response=content,
                    metrics=metrics,
                )
        except Exception as error:
            log.warning(f'OpenAI structured model invocation failed: {error}')

            return StructuredResponse(
                data={},
                raw_response='',
                metrics=LDAIMetrics(success=False, usage=None),
            )

    # --- Convenience accessors ---

    def get_client(self) -> AsyncOpenAI:
        """
        Get the underlying OpenAI client instance.

        :return: The underlying AsyncOpenAI client
        """
        return self._client

    @staticmethod
    def get_ai_metrics_from_response(response: Any) -> LDAIMetrics:
        """
        Extract LaunchDarkly AI metrics from an OpenAI response.

        :param response: The response from OpenAI chat completions API
        :return: LDAIMetrics with success status and token usage

        Example::

            response = await tracker.track_metrics_of(
                lambda: client.chat.completions.create(config),
                OpenAIRunnerFactory.get_ai_metrics_from_response
            )
        """
        usage: Optional[TokenUsage] = None
        if hasattr(response, 'usage') and response.usage:
            usage = TokenUsage(
                total=response.usage.total_tokens or 0,
                input=response.usage.prompt_tokens or 0,
                output=response.usage.completion_tokens or 0,
            )

        return LDAIMetrics(success=True, usage=usage)
