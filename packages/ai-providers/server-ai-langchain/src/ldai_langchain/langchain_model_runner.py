from typing import Any, Dict, List, Optional, cast

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from ldai import LDMessage, log
from ldai.providers.runner import Runner
from ldai.providers.types import LDAIMetrics, RunnerResult

from ldai_langchain.langchain_helper import (
    convert_messages_to_langchain,
    get_ai_metrics_from_response,
    get_ai_usage_from_response,
)


class LangChainModelRunner(Runner):
    """
    Runner implementation for LangChain chat models.

    Holds a fully-configured BaseChatModel.
    Returned by LangChainRunnerFactory.create_model(config).

    Implements the unified :class:`~ldai.providers.runner.Runner` protocol via
    :meth:`run`.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        config_messages: Optional[List[LDMessage]] = None,
        multi_turn: bool = True,
    ):
        self._llm = llm
        self._chat_history = InMemoryChatMessageHistory(
            messages=cast(List[BaseMessage], convert_messages_to_langchain(config_messages or []))
        )
        self._multi_turn = multi_turn

    def get_llm(self) -> BaseChatModel:
        """
        Return the underlying LangChain BaseChatModel.

        :return: The BaseChatModel instance
        """
        return self._llm

    async def run(
        self,
        input: str,
        output_type: Optional[Dict[str, Any]] = None,
    ) -> RunnerResult:
        """
        Run the LangChain model with the given input.

        :param input: A string prompt
        :param output_type: Optional JSON schema dict requesting structured output.
            When provided, ``parsed`` on the returned :class:`RunnerResult` is
            populated with the parsed JSON document.
        :return: :class:`RunnerResult` containing ``content``, ``metrics``,
            ``raw`` and (when ``output_type`` is set) ``parsed``.
        """
        langchain_messages = self._chat_history.messages + [HumanMessage(content=input)]

        if output_type is not None:
            result = await self._run_structured(langchain_messages, output_type)
        else:
            result = await self._run_completion(langchain_messages)

        if result.metrics.success and result.content and self._multi_turn:
            self._chat_history.add_user_message(input)
            self._chat_history.add_ai_message(result.content)

        return result

    async def _run_completion(self, messages: List[BaseMessage]) -> RunnerResult:
        try:
            response: BaseMessage = await self._llm.ainvoke(messages)
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
                    metrics=LDAIMetrics(success=False, tokens=metrics.tokens),
                    raw=response,
                )

            return RunnerResult(content=content, metrics=metrics, raw=response)
        except Exception as error:
            log.warning(f'LangChain model invocation failed: {error}')
            return RunnerResult(
                content='',
                metrics=LDAIMetrics(success=False, tokens=None),
            )

    async def _run_structured(
        self,
        messages: List[BaseMessage],
        output_type: Dict[str, Any],
    ) -> RunnerResult:
        try:
            structured_llm = self._llm.with_structured_output(output_type, include_raw=True)
            response = await structured_llm.ainvoke(messages)

            if not isinstance(response, dict):
                log.warning(f'Structured output did not return a dict. Got: {type(response)}')
                return RunnerResult(
                    content='',
                    metrics=LDAIMetrics(success=False, tokens=None),
                )

            raw_response = response.get('raw')
            tokens = None
            raw_content = ''
            if raw_response is not None:
                if hasattr(raw_response, 'content'):
                    if isinstance(raw_response.content, str):
                        raw_content = raw_response.content
                    else:
                        log.warning(
                            f'Multimodal response not supported in structured mode. '
                            f'Content type: {type(raw_response.content)}, Content: {raw_response.content}'
                        )
                tokens = get_ai_usage_from_response(raw_response)

            if response.get('parsing_error'):
                log.warning('LangChain structured model invocation had a parsing error')
                return RunnerResult(
                    content=raw_content,
                    metrics=LDAIMetrics(success=False, tokens=tokens),
                    raw=raw_response,
                )

            parsed = response.get('parsed') or {}
            return RunnerResult(
                content=raw_content,
                metrics=LDAIMetrics(success=True, tokens=tokens),
                raw=raw_response,
                parsed=parsed,
            )
        except Exception as error:
            log.warning(f'LangChain structured model invocation failed: {error}')
            return RunnerResult(
                content='',
                metrics=LDAIMetrics(success=False, tokens=None),
            )
