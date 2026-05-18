import asyncio
import json
from typing import Any, Dict, List, Optional

from ldai import LDMessage, log
from ldai.providers.runner import Runner
from ldai.providers.types import LDAIMetrics, RunnerResult

from ldai_bedrock.bedrock_helper import (
    convert_messages_to_bedrock,
    convert_tools_to_bedrock,
    extract_content_from_response,
    get_ai_metrics_from_response,
)


class BedrockModelRunner(Runner):
    """
    Runner implementation for Amazon Bedrock chat completions via the
    Converse API.

    Holds a fully-configured ``bedrock-runtime`` client, model id, and
    parameters.  Returned by ``BedrockRunnerFactory.create_model(config)``.

    The Converse API is synchronous (boto3 has no async transport), so
    each invocation dispatches to a worker thread via ``asyncio.to_thread``
    to avoid blocking the event loop.

    Implements the unified :class:`~ldai.providers.runner.Runner` protocol
    via :meth:`run`.
    """

    # Inference parameters recognised by the Converse API's ``inferenceConfig``.
    # All other parameters are forwarded as ``additionalModelRequestFields``.
    _INFERENCE_KEYS = frozenset({
        'maxTokens',
        'temperature',
        'topP',
        'stopSequences',
    })

    def __init__(
        self,
        client: Any,
        model_id: str,
        parameters: Dict[str, Any],
        config_messages: Optional[List[LDMessage]] = None,
        multi_turn: bool = True,
    ):
        self._client = client
        self._model_id = model_id
        self._parameters = parameters
        self._history: List[LDMessage] = list(config_messages or [])
        self._multi_turn = multi_turn

    async def run(
        self,
        input: str,
        output_type: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Run the Bedrock model with the given input.

        :param input: A string prompt
        :param output_type: Optional JSON schema dict requesting structured output.
            When provided, ``parsed`` on the returned :class:`RunnerResult` is
            populated with the parsed JSON document.  Bedrock does not have a
            native structured-output mode comparable to OpenAI's ``response_format``,
            so the schema is appended as a system instruction and the model's
            response is parsed as JSON.
        :return: :class:`RunnerResult` containing ``content``, ``metrics``,
            ``raw`` and (when ``output_type`` is set) ``parsed``.
        """
        user_message = LDMessage(role='user', content=input)
        messages = self._history + [user_message]

        if output_type is not None:
            result = await self._run_structured(messages, output_type)
        else:
            result = await self._run_completion(messages)

        if result.metrics.success and result.content and self._multi_turn:
            self._history.append(user_message)
            self._history.append(LDMessage(role='assistant', content=result.content))

        return result

    async def _run_completion(self, messages: List[LDMessage]) -> RunnerResult:
        try:
            response = await self._invoke_converse(messages)

            metrics = get_ai_metrics_from_response(response)
            content = extract_content_from_response(response)

            if not content:
                log.warning('Bedrock response has no content available')
                return RunnerResult(
                    content='',
                    metrics=LDAIMetrics(
                        success=False,
                        tokens=metrics.tokens,
                        duration_ms=metrics.duration_ms,
                    ),
                    raw=response,
                )

            return RunnerResult(content=content, metrics=metrics, raw=response)
        except Exception as error:
            log.warning(f'Bedrock model invocation failed: {error}')
            return RunnerResult(
                content='',
                metrics=LDAIMetrics(success=False, tokens=None),
            )

    async def _run_structured(
        self,
        messages: List[LDMessage],
        output_type: Dict[str, Any],
    ) -> RunnerResult:
        # Bedrock has no first-class JSON-schema response mode comparable to
        # OpenAI's ``response_format``.  Inject the schema as a system
        # instruction and parse the response text as JSON.
        schema_instruction = LDMessage(
            role='system',
            content=(
                'Respond with a JSON document that conforms exactly to the '
                'following JSON schema. Output JSON only, with no surrounding '
                f'prose or code fences.\nSchema: {json.dumps(output_type)}'
            ),
        )
        augmented = [schema_instruction, *messages]

        try:
            response = await self._invoke_converse(augmented)

            metrics = get_ai_metrics_from_response(response)
            content = extract_content_from_response(response)

            if not content:
                log.warning('Bedrock structured response has no content available')
                return RunnerResult(
                    content='',
                    metrics=LDAIMetrics(
                        success=False,
                        tokens=metrics.tokens,
                        duration_ms=metrics.duration_ms,
                    ),
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
                log.warning(f'Bedrock structured response contains invalid JSON: {parse_error}')
                return RunnerResult(
                    content=content,
                    metrics=LDAIMetrics(
                        success=False,
                        tokens=metrics.tokens,
                        duration_ms=metrics.duration_ms,
                    ),
                    raw=response,
                )
        except Exception as error:
            log.warning(f'Bedrock structured model invocation failed: {error}')
            return RunnerResult(
                content='',
                metrics=LDAIMetrics(success=False, tokens=None),
            )

    async def _invoke_converse(self, messages: List[LDMessage]) -> Dict[str, Any]:
        request = self._build_request(messages)
        return await asyncio.to_thread(self._client.converse, **request)

    def _build_request(self, messages: List[LDMessage]) -> Dict[str, Any]:
        """Assemble the kwargs passed to ``bedrock-runtime.converse(...)``."""
        request: Dict[str, Any] = {'modelId': self._model_id}
        request.update(convert_messages_to_bedrock(messages))

        inference_config: Dict[str, Any] = {}
        additional_fields: Dict[str, Any] = {}
        for key, value in self._parameters.items():
            if key == 'tools':
                continue
            if key in self._INFERENCE_KEYS:
                inference_config[key] = value
            else:
                additional_fields[key] = value

        if inference_config:
            request['inferenceConfig'] = inference_config
        if additional_fields:
            request['additionalModelRequestFields'] = additional_fields

        tool_config = convert_tools_to_bedrock(self._parameters.get('tools') or [])
        if tool_config:
            request['toolConfig'] = tool_config

        return request
