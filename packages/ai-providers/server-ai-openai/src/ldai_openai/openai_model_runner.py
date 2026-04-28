import json
from typing import Any, Dict, List, Optional

from ldai import LDMessage, log
from ldai.providers.model_runner import ModelRunner
from ldai.providers.types import LDAIMetrics, ModelResponse, RunnerResult, StructuredResponse
from openai import AsyncOpenAI

from ldai_openai.openai_helper import (
    convert_messages_to_openai,
    get_ai_metrics_from_response,
)


class OpenAIModelRunner(ModelRunner):
    """
    Runner implementation for OpenAI chat completions.

    Holds a fully-configured AsyncOpenAI client, model name, and parameters.
    Returned by ``OpenAIRunnerFactory.create_model(config)``.

    Implements the unified :class:`~ldai.providers.runner.Runner` protocol via
    :meth:`run`. The legacy :meth:`invoke_model` and :meth:`invoke_structured_model`
    methods are preserved for backward compatibility with the managed layer until
    its migration to the unified protocol is complete.
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

    async def run(
        self,
        input: Any,
        output_type: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Run the OpenAI model with the given input.

        :param input: A string prompt or a list of :class:`LDMessage` objects
        :param output_type: Optional JSON schema dict requesting structured output.
            When provided, ``parsed`` on the returned :class:`RunnerResult` is
            populated with the parsed JSON document.
        :return: :class:`RunnerResult` containing ``content``, ``metrics``,
            ``raw`` and (when ``output_type`` is set) ``parsed``.
        """
        messages = self._coerce_input(input)

        if output_type is not None:
            return await self._run_structured(messages, output_type)
        return await self._run_completion(messages)

    @staticmethod
    def _coerce_input(input: Any) -> List[LDMessage]:
        if isinstance(input, str):
            return [LDMessage(role='user', content=input)]
        if isinstance(input, list):
            return input
        raise TypeError(
            f"Unsupported input type for OpenAIModelRunner.run: {type(input).__name__}"
        )

    async def _run_completion(self, messages: List[LDMessage]) -> RunnerResult:
        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=convert_messages_to_openai(messages),
                **self._parameters,
            )

            metrics = get_ai_metrics_from_response(response)
            content = self._extract_content(response)

            if not content:
                log.warning('OpenAI response has no content available')
                return RunnerResult(
                    content='',
                    metrics=LDAIMetrics(success=False, usage=metrics.usage),
                    raw=response,
                )

            return RunnerResult(content=content, metrics=metrics, raw=response)
        except Exception as error:
            log.warning(f'OpenAI model invocation failed: {error}')
            return RunnerResult(
                content='',
                metrics=LDAIMetrics(success=False, usage=None),
            )

    async def _run_structured(
        self,
        messages: List[LDMessage],
        output_type: Dict[str, Any],
    ) -> RunnerResult:
        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=convert_messages_to_openai(messages),
                response_format={  # type: ignore[arg-type]
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'structured_output',
                        'schema': output_type,
                        'strict': True,
                    },
                },
                **self._parameters,
            )

            metrics = get_ai_metrics_from_response(response)
            content = self._extract_content(response)

            if not content:
                log.warning('OpenAI structured response has no content available')
                return RunnerResult(
                    content='',
                    metrics=LDAIMetrics(success=False, usage=metrics.usage),
                    raw=response,
                )

            try:
                parsed = json.loads(content)
                return RunnerResult(
                    content=content,
                    metrics=metrics,
                    raw=response,
                    parsed=parsed,
                )
            except json.JSONDecodeError as parse_error:
                log.warning(f'OpenAI structured response contains invalid JSON: {parse_error}')
                return RunnerResult(
                    content=content,
                    metrics=LDAIMetrics(success=False, usage=metrics.usage),
                    raw=response,
                )
        except Exception as error:
            log.warning(f'OpenAI structured model invocation failed: {error}')
            return RunnerResult(
                content='',
                metrics=LDAIMetrics(success=False, usage=None),
            )

    @staticmethod
    def _extract_content(response: Any) -> str:
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            if message and message.content:
                return message.content
        return ''

    async def invoke_model(self, messages: List[LDMessage]) -> ModelResponse:
        """
        Invoke the OpenAI model with an array of messages.

        .. deprecated::
            Use :meth:`run` instead. This method delegates to :meth:`run` and
            adapts the result to the legacy :class:`ModelResponse` shape so
            existing callers in the managed layer continue to function.
        """
        result = await self._run_completion(messages)
        return ModelResponse(
            message=LDMessage(role='assistant', content=result.content),
            metrics=result.metrics,
        )

    async def invoke_structured_model(
        self,
        messages: List[LDMessage],
        response_structure: Dict[str, Any],
    ) -> StructuredResponse:
        """
        Invoke the OpenAI model with structured output support.

        .. deprecated::
            Use :meth:`run` with the ``output_type`` argument instead. This
            method delegates to :meth:`run` and adapts the result to the
            legacy :class:`StructuredResponse` shape.
        """
        result = await self._run_structured(messages, response_structure)
        return StructuredResponse(
            data=result.parsed or {},
            raw_response=result.content,
            metrics=result.metrics,
        )
