from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from ldai import LDMessage, log
from ldai.providers.types import LDAIMetrics, RunnerResult

from ldai_langchain.langchain_helper import (
    convert_messages_to_langchain,
    get_ai_metrics_from_response,
    get_ai_usage_from_response,
)


class LangChainModelRunner:
    """
    Runner implementation for LangChain chat models.

    Holds a fully-configured BaseChatModel.
    Returned by ``LangChainRunnerFactory.create_model(config)``.

    Implements the unified :class:`~ldai.providers.runner.Runner` protocol via
    :meth:`run`.
    """

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def get_llm(self) -> BaseChatModel:
        """
        Return the underlying LangChain BaseChatModel.

        :return: The BaseChatModel instance
        """
        return self._llm

    async def run(
        self,
        input: Any,
        output_type: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Run the LangChain model with the given input.

        :param input: A string prompt or a list of :class:`LDMessage` objects
        :param output_type: Optional JSON schema dict requesting structured output.
            When provided, ``parsed`` on the returned :class:`RunnerResult` is
            populated with the structured data.
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
            f"Unsupported input type for LangChainModelRunner.run: {type(input).__name__}"
        )

    async def _run_completion(self, messages: List[LDMessage]) -> RunnerResult:
        try:
            langchain_messages = convert_messages_to_langchain(messages)
            response: BaseMessage = await self._llm.ainvoke(langchain_messages)
            metrics = get_ai_metrics_from_response(response)

            content: str = ''
            if isinstance(response.content, str):
                content = response.content
            else:
                log.warning(
                    f'Multimodal response not supported, expecting a string. '
                    f'Content type: {type(response.content)}, Content: {response.content}'
                )
                return RunnerResult(
                    content='',
                    metrics=LDAIMetrics(success=False, usage=metrics.usage),
                    raw=response,
                )

            return RunnerResult(content=content, metrics=metrics, raw=response)
        except Exception as error:
            log.warning(f'LangChain model invocation failed: {error}')
            return RunnerResult(
                content='',
                metrics=LDAIMetrics(success=False, usage=None),
            )

    async def _run_structured(
        self, messages: List[LDMessage], response_structure: Dict[str, Any]
    ) -> RunnerResult:
        try:
            langchain_messages = convert_messages_to_langchain(messages)
            structured_llm = self._llm.with_structured_output(response_structure, include_raw=True)
            response = await structured_llm.ainvoke(langchain_messages)

            if not isinstance(response, dict):
                log.warning(f'Structured output did not return a dict. Got: {type(response)}')
                return RunnerResult(
                    content='',
                    metrics=LDAIMetrics(success=False, usage=None),
                )

            raw_response = response.get('raw')
            usage = get_ai_usage_from_response(raw_response) if raw_response is not None else None
            raw_content = raw_response.content if raw_response is not None and hasattr(raw_response, 'content') else ''

            if response.get('parsing_error'):
                log.warning('LangChain structured model invocation had a parsing error')
                return RunnerResult(
                    content=raw_content,
                    metrics=LDAIMetrics(success=False, usage=usage),
                    raw=raw_response,
                )

            parsed = response.get('parsed') or {}
            return RunnerResult(
                content=raw_content,
                metrics=LDAIMetrics(success=True, usage=usage),
                raw=raw_response,
                parsed=parsed,
            )
        except Exception as error:
            log.warning(f'LangChain structured model invocation failed: {error}')
            return RunnerResult(
                content='',
                metrics=LDAIMetrics(success=False, usage=None),
            )

