"""Tests for LangChain provider implementation."""

import pytest
from unittest.mock import AsyncMock, Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ldai.models import LDMessage
from ldai.providers.langchain import LangChainProvider
from ldai.tracker import TokenUsage


class TestMessageConversion:
    """Test conversion between LD messages and LangChain messages."""

    def test_convert_multiple_messages(self):
        """Test converting a conversation with all message types."""
        ld_messages = [
            LDMessage(role='system', content='You are helpful'),
            LDMessage(role='user', content='Hello'),
            LDMessage(role='assistant', content='Hi there!'),
        ]
        lc_messages = LangChainProvider.convert_messages_to_langchain(ld_messages)
        
        assert len(lc_messages) == 3
        assert isinstance(lc_messages[0], SystemMessage)
        assert isinstance(lc_messages[1], HumanMessage)
        assert isinstance(lc_messages[2], AIMessage)
        assert lc_messages[0].content == 'You are helpful'
        assert lc_messages[1].content == 'Hello'
        assert lc_messages[2].content == 'Hi there!'

    def test_convert_unsupported_role_raises_error(self):
        """Test that unsupported message roles raise ValueError."""
        ld_messages = [LDMessage(role='function', content='Function result')]
        
        with pytest.raises(ValueError, match='Unsupported message role: function'):
            LangChainProvider.convert_messages_to_langchain(ld_messages)


class TestMetricsExtraction:
    """Test metrics extraction from LangChain response metadata."""

    def test_extract_metrics_with_token_usage(self):
        """Test extracting token usage from response metadata."""
        response = AIMessage(
            content='Hello, world!',
            response_metadata={
                'token_usage': {
                    'total_tokens': 100,
                    'prompt_tokens': 60,
                    'completion_tokens': 40,
                }
            }
        )
        
        metrics = LangChainProvider.get_ai_metrics_from_response(response)
        
        assert metrics.success is True
        assert metrics.usage is not None
        assert metrics.usage.total == 100
        assert metrics.usage.input == 60
        assert metrics.usage.output == 40

    def test_extract_metrics_with_camel_case_token_usage(self):
        """Test extracting token usage with camelCase keys (some providers use this)."""
        response = AIMessage(
            content='Hello, world!',
            response_metadata={
                'token_usage': {
                    'totalTokens': 150,
                    'promptTokens': 90,
                    'completionTokens': 60,
                }
            }
        )
        
        metrics = LangChainProvider.get_ai_metrics_from_response(response)
        
        assert metrics.success is True
        assert metrics.usage is not None
        assert metrics.usage.total == 150
        assert metrics.usage.input == 90
        assert metrics.usage.output == 60

    def test_extract_metrics_without_token_usage(self):
        """Test metrics extraction when no token usage is available."""
        response = AIMessage(content='Hello, world!')
        
        metrics = LangChainProvider.get_ai_metrics_from_response(response)
        
        assert metrics.success is True
        assert metrics.usage is None


class TestInvokeModel:
    """Test model invocation with LangChain provider."""

    @pytest.mark.asyncio
    async def test_invoke_model_success(self):
        """Test successful model invocation."""
        mock_llm = AsyncMock()
        mock_response = AIMessage(
            content='Hello, user!',
            response_metadata={
                'token_usage': {
                    'total_tokens': 20,
                    'prompt_tokens': 10,
                    'completion_tokens': 10,
                }
            }
        )
        mock_llm.ainvoke.return_value = mock_response
        
        provider = LangChainProvider(mock_llm)
        messages = [LDMessage(role='user', content='Hello')]
        
        response = await provider.invoke_model(messages)
        
        assert response.message.role == 'assistant'
        assert response.message.content == 'Hello, user!'
        assert response.metrics.success is True
        assert response.metrics.usage is not None
        assert response.metrics.usage.total == 20

    @pytest.mark.asyncio
    async def test_invoke_model_with_multimodal_content_warning(self):
        """Test that non-string content triggers warning and marks as failure."""
        mock_llm = AsyncMock()
        mock_response = AIMessage(
            content=['text', {'type': 'image'}],  # Non-string content
            response_metadata={'token_usage': {'total_tokens': 20}}
        )
        mock_llm.ainvoke.return_value = mock_response
        
        mock_logger = Mock()
        provider = LangChainProvider(mock_llm, logger=mock_logger)
        messages = [LDMessage(role='user', content='Describe this image')]
        
        response = await provider.invoke_model(messages)
        
        # Should warn about multimodal content
        mock_logger.warn.assert_called_once()
        assert 'Multimodal response not supported' in str(mock_logger.warn.call_args)
        
        # Should mark as failure
        assert response.metrics.success is False
        assert response.message.content == ''

    @pytest.mark.asyncio
    async def test_invoke_model_with_exception(self):
        """Test model invocation handles exceptions gracefully."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception('Model API error')
        
        mock_logger = Mock()
        provider = LangChainProvider(mock_llm, logger=mock_logger)
        messages = [LDMessage(role='user', content='Hello')]
        
        response = await provider.invoke_model(messages)
        
        # Should log the error
        mock_logger.warn.assert_called_once()
        assert 'LangChain model invocation failed' in str(mock_logger.warn.call_args)
        
        # Should return failure response
        assert response.message.role == 'assistant'
        assert response.message.content == ''
        assert response.metrics.success is False
        assert response.metrics.usage is None


class TestInvokeStructuredModel:
    """Test structured output invocation."""

    @pytest.mark.asyncio
    async def test_invoke_structured_model_with_support(self):
        """Test structured output when model supports with_structured_output."""
        mock_llm = Mock()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = {
            'answer': 'Paris',
            'confidence': 0.95
        }
        mock_llm.with_structured_output.return_value = mock_structured_llm
        
        provider = LangChainProvider(mock_llm)
        messages = [LDMessage(role='user', content='What is the capital of France?')]
        schema = {'answer': 'string', 'confidence': 'number'}
        
        response = await provider.invoke_structured_model(messages, schema)
        
        assert response.data == {'answer': 'Paris', 'confidence': 0.95}
        assert response.metrics.success is True
        mock_llm.with_structured_output.assert_called_once_with(schema)

    @pytest.mark.asyncio
    async def test_invoke_structured_model_without_support_json_fallback(self):
        """Test structured output fallback to JSON parsing when not supported."""
        mock_llm = AsyncMock()
        # Model doesn't have with_structured_output
        delattr(mock_llm, 'with_structured_output') if hasattr(mock_llm, 'with_structured_output') else None
        
        mock_response = AIMessage(content='{"answer": "Berlin", "confidence": 0.9}')
        mock_llm.ainvoke.return_value = mock_response
        
        provider = LangChainProvider(mock_llm)
        messages = [LDMessage(role='user', content='What is the capital of Germany?')]
        schema = {'answer': 'string', 'confidence': 'number'}
        
        response = await provider.invoke_structured_model(messages, schema)
        
        assert response.data == {'answer': 'Berlin', 'confidence': 0.9}
        assert response.metrics.success is True

    @pytest.mark.asyncio
    async def test_invoke_structured_model_with_exception(self):
        """Test structured output handles exceptions gracefully."""
        mock_llm = Mock()
        mock_llm.with_structured_output.side_effect = Exception('Structured output error')
        
        mock_logger = Mock()
        provider = LangChainProvider(mock_llm, logger=mock_logger)
        messages = [LDMessage(role='user', content='Question')]
        schema = {'answer': 'string'}
        
        response = await provider.invoke_structured_model(messages, schema)
        
        # Should log the error
        mock_logger.warn.assert_called_once()
        assert 'LangChain structured model invocation failed' in str(mock_logger.warn.call_args)
        
        # Should return failure response
        assert response.data == {}
        assert response.raw_response == ''
        assert response.metrics.success is False

