"""Tests for LangChain Provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ldai import LDMessage

from ldai_langchain import LangChainProvider


class TestConvertMessagesToLangchain:
    """Tests for convert_messages_to_langchain static method."""

    def test_converts_system_messages_to_system_message(self):
        """Should convert system messages to SystemMessage."""
        messages = [LDMessage(role='system', content='You are a helpful assistant.')]
        result = LangChainProvider.convert_messages_to_langchain(messages)

        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == 'You are a helpful assistant.'

    def test_converts_user_messages_to_human_message(self):
        """Should convert user messages to HumanMessage."""
        messages = [LDMessage(role='user', content='Hello, how are you?')]
        result = LangChainProvider.convert_messages_to_langchain(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == 'Hello, how are you?'

    def test_converts_assistant_messages_to_ai_message(self):
        """Should convert assistant messages to AIMessage."""
        messages = [LDMessage(role='assistant', content='I am doing well, thank you!')]
        result = LangChainProvider.convert_messages_to_langchain(messages)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == 'I am doing well, thank you!'

    def test_converts_multiple_messages_in_order(self):
        """Should convert multiple messages in order."""
        messages = [
            LDMessage(role='system', content='You are a helpful assistant.'),
            LDMessage(role='user', content='What is the weather like?'),
            LDMessage(role='assistant', content='I cannot check the weather.'),
        ]
        result = LangChainProvider.convert_messages_to_langchain(messages)

        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)

    def test_throws_error_for_unsupported_message_role(self):
        """Should throw error for unsupported message role."""
        # Create a mock message with unsupported role
        class MockMessage:
            role = 'unknown'
            content = 'Test message'
        
        with pytest.raises(ValueError, match='Unsupported message role: unknown'):
            LangChainProvider.convert_messages_to_langchain([MockMessage()])  # type: ignore

    def test_handles_empty_message_array(self):
        """Should handle empty message array."""
        result = LangChainProvider.convert_messages_to_langchain([])
        assert len(result) == 0


class TestGetAIMetricsFromResponse:
    """Tests for get_ai_metrics_from_response static method."""

    def test_creates_metrics_with_success_true_and_token_usage(self):
        """Should create metrics with success=True and token usage."""
        mock_response = AIMessage(content='Test response')
        mock_response.response_metadata = {
            'tokenUsage': {
                'totalTokens': 100,
                'promptTokens': 50,
                'completionTokens': 50,
            },
        }

        result = LangChainProvider.get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is not None
        assert result.usage.total == 100
        assert result.usage.input == 50
        assert result.usage.output == 50

    def test_creates_metrics_with_snake_case_token_usage(self):
        """Should create metrics with snake_case token usage keys."""
        mock_response = AIMessage(content='Test response')
        mock_response.response_metadata = {
            'token_usage': {
                'total_tokens': 150,
                'prompt_tokens': 75,
                'completion_tokens': 75,
            },
        }

        result = LangChainProvider.get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is not None
        assert result.usage.total == 150
        assert result.usage.input == 75
        assert result.usage.output == 75

    def test_creates_metrics_with_success_true_and_no_usage_when_metadata_missing(self):
        """Should create metrics with success=True and no usage when metadata is missing."""
        mock_response = AIMessage(content='Test response')

        result = LangChainProvider.get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is None


class TestMapProvider:
    """Tests for map_provider static method."""

    def test_maps_gemini_to_google_genai(self):
        """Should map gemini to google-genai."""
        assert LangChainProvider.map_provider('gemini') == 'google-genai'
        assert LangChainProvider.map_provider('Gemini') == 'google-genai'
        assert LangChainProvider.map_provider('GEMINI') == 'google-genai'

    def test_returns_provider_name_unchanged_for_unmapped_providers(self):
        """Should return provider name unchanged for unmapped providers."""
        assert LangChainProvider.map_provider('openai') == 'openai'
        assert LangChainProvider.map_provider('anthropic') == 'anthropic'
        assert LangChainProvider.map_provider('unknown') == 'unknown'


class TestInvokeModel:
    """Tests for invoke_model instance method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MagicMock()

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_returns_success_true_for_string_content(self, mock_llm, mock_logger):
        """Should return success=True for string content."""
        mock_response = AIMessage(content='Test response')
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        provider = LangChainProvider(mock_llm, mock_logger)

        messages = [LDMessage(role='user', content='Hello')]
        result = await provider.invoke_model(messages)

        assert result.metrics.success is True
        assert result.message.content == 'Test response'
        mock_logger.warn.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_success_false_for_non_string_content_and_logs_warning(self, mock_llm, mock_logger):
        """Should return success=False for non-string content and log warning."""
        mock_response = AIMessage(content=[{'type': 'image', 'data': 'base64data'}])
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        provider = LangChainProvider(mock_llm, mock_logger)

        messages = [LDMessage(role='user', content='Hello')]
        result = await provider.invoke_model(messages)

        assert result.metrics.success is False
        assert result.message.content == ''
        mock_logger.warn.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_success_false_when_model_invocation_throws_error(self, mock_llm, mock_logger):
        """Should return success=False when model invocation throws an error."""
        error = Exception('Model invocation failed')
        mock_llm.ainvoke = AsyncMock(side_effect=error)
        provider = LangChainProvider(mock_llm, mock_logger)

        messages = [LDMessage(role='user', content='Hello')]
        result = await provider.invoke_model(messages)

        assert result.metrics.success is False
        assert result.message.content == ''
        assert result.message.role == 'assistant'
        mock_logger.warn.assert_called()


class TestInvokeStructuredModel:
    """Tests for invoke_structured_model instance method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MagicMock()

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_returns_success_true_for_successful_invocation(self, mock_llm, mock_logger):
        """Should return success=True for successful invocation."""
        mock_response = {'result': 'structured data'}
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        provider = LangChainProvider(mock_llm, mock_logger)

        messages = [LDMessage(role='user', content='Hello')]
        response_structure = {'type': 'object', 'properties': {}}
        result = await provider.invoke_structured_model(messages, response_structure)

        assert result.metrics.success is True
        assert result.data == mock_response
        mock_logger.warn.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_success_false_when_structured_model_invocation_throws_error(self, mock_llm, mock_logger):
        """Should return success=False when structured model invocation throws an error."""
        error = Exception('Structured invocation failed')
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(side_effect=error)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        provider = LangChainProvider(mock_llm, mock_logger)

        messages = [LDMessage(role='user', content='Hello')]
        response_structure = {'type': 'object', 'properties': {}}
        result = await provider.invoke_structured_model(messages, response_structure)

        assert result.metrics.success is False
        assert result.data == {}
        assert result.raw_response == ''
        assert result.metrics.usage is not None
        assert result.metrics.usage.total == 0
        mock_logger.warn.assert_called()


class TestGetChatModel:
    """Tests for get_chat_model instance method."""

    def test_returns_underlying_llm(self):
        """Should return the underlying LLM."""
        mock_llm = MagicMock()
        provider = LangChainProvider(mock_llm)

        assert provider.get_chat_model() is mock_llm


