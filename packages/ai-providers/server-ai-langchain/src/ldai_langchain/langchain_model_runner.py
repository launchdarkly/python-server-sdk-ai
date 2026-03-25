from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from ldai import LDMessage, log
from ldai.providers.model_runner import ModelRunner
from ldai.providers.types import LDAIMetrics, ModelResponse, StructuredResponse
from ldai_langchain.langchain_helper import LangChainHelper


class LangChainModelRunner(ModelRunner):
    """
    ModelRunner implementation for LangChain.

    Holds a fully-configured BaseChatModel.
    Returned by LangChainConnector.create_model(config).
    """

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def get_llm(self) -> BaseChatModel:
        """
        Return the underlying LangChain BaseChatModel.

        :return: The BaseChatModel instance
        """
        return self._llm

    async def invoke_model(self, messages: List[LDMessage]) -> ModelResponse:
        """
        Invoke the LangChain model with an array of messages.

        :param messages: Array of LDMessage objects representing the conversation
        :return: ModelResponse containing the model's response and metrics
        """
        try:
            langchain_messages = LangChainHelper.convert_messages(messages)
            response: BaseMessage = await self._llm.ainvoke(langchain_messages)
            metrics = LangChainHelper.get_ai_metrics_from_response(response)

            content: str = ''
            if isinstance(response.content, str):
                content = response.content
            else:
                log.warning(
                    f'Multimodal response not supported, expecting a string. '
                    f'Content type: {type(response.content)}, Content: {response.content}'
                )
                metrics = LDAIMetrics(success=False, usage=metrics.usage)

            return ModelResponse(
                message=LDMessage(role='assistant', content=content),
                metrics=metrics,
            )
        except Exception as error:
            log.warning(f'LangChain model invocation failed: {error}')
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
        Invoke the LangChain model with structured output support.

        :param messages: Array of LDMessage objects representing the conversation
        :param response_structure: Dictionary defining the output structure
        :return: StructuredResponse containing the structured data
        """
        try:
            langchain_messages = LangChainHelper.convert_messages(messages)
            structured_llm = self._llm.with_structured_output(response_structure, include_raw=True)
            response = await structured_llm.ainvoke(langchain_messages)

            if not isinstance(response, dict):
                log.warning(f'Structured output did not return a dict. Got: {type(response)}')
                return StructuredResponse(
                    data={},
                    raw_response='',
                    metrics=LDAIMetrics(success=False, usage=None),
                )

            raw_response = response.get('raw')
            usage = LangChainHelper.get_ai_usage_from_response(raw_response) if raw_response is not None else None
            raw_content = raw_response.content if hasattr(raw_response, 'content') else ''

            if response.get('parsing_error'):
                log.warning('LangChain structured model invocation had a parsing error')
                return StructuredResponse(
                    data={},
                    raw_response=raw_content,
                    metrics=LDAIMetrics(success=False, usage=usage),
                )

            return StructuredResponse(
                data=response.get('parsed') or {},
                raw_response=raw_content,
                metrics=LDAIMetrics(success=True, usage=usage),
            )
        except Exception as error:
            log.warning(f'LangChain structured model invocation failed: {error}')
            return StructuredResponse(
                data={},
                raw_response='',
                metrics=LDAIMetrics(success=False, usage=None),
            )

