import json
from typing import Any, Dict, List

from ldai import LDMessage, log
from ldai.providers.model_runner import ModelRunner
from ldai.providers.types import LDAIMetrics, ModelResponse, StructuredResponse
from ldai.tracker import TokenUsage
from openai import AsyncOpenAI

from ldai_openai.openai_helper import OpenAIHelper


class OpenAIModelRunner(ModelRunner):
    """
    ModelRunner implementation for OpenAI.

    Holds a fully-configured AsyncOpenAI client, model name, and parameters.
    Returned by OpenAIConnector.create_model(config).
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        parameters: Dict[str, Any],
    ):
        self._client = client
        self._model_name = model_name
        self._parameters = parameters

    async def invoke_model(self, messages: List[LDMessage]) -> ModelResponse:
        """
        Invoke the OpenAI model with an array of messages.

        :param messages: Array of LDMessage objects representing the conversation
        :return: ModelResponse containing the model's response and metrics
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=OpenAIHelper.convert_messages(messages),
                **self._parameters,
            )

            metrics = OpenAIHelper.get_ai_metrics_from_response(response)

            content = ''
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content

            if not content:
                log.warning('OpenAI response has no content available')
                metrics = LDAIMetrics(success=False, usage=metrics.usage)

            return ModelResponse(
                message=LDMessage(role='assistant', content=content),
                metrics=metrics,
            )
        except Exception as error:
            log.warning(f'OpenAI model invocation failed: {error}')
            return ModelResponse(
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
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=OpenAIHelper.convert_messages(messages),
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

            metrics = OpenAIHelper.get_ai_metrics_from_response(response)

            content = ''
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content

            if not content:
                log.warning('OpenAI structured response has no content available')
                return StructuredResponse(
                    data={},
                    raw_response='',
                    metrics=LDAIMetrics(success=False, usage=metrics.usage),
                )

            try:
                data = json.loads(content)
                return StructuredResponse(data=data, raw_response=content, metrics=metrics)
            except json.JSONDecodeError as parse_error:
                log.warning(f'OpenAI structured response contains invalid JSON: {parse_error}')
                return StructuredResponse(
                    data={},
                    raw_response=content,
                    metrics=LDAIMetrics(success=False, usage=metrics.usage),
                )
        except Exception as error:
            log.warning(f'OpenAI structured model invocation failed: {error}')
            return StructuredResponse(
                data={},
                raw_response='',
                metrics=LDAIMetrics(success=False, usage=None),
            )
