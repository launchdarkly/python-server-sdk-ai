"""Tests for OpenAI Provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ldai import LDMessage

from ldai_openai import OpenAIProvider


class TestGetAIMetricsFromResponse:
    """Tests for get_ai_metrics_from_response static method."""

    def test_creates_metrics_with_success_true_and_token_usage(self):
        """Should create metrics with success=True and token usage."""
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 100

        result = OpenAIProvider.get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is not None
        assert result.usage.total == 100
        assert result.usage.input == 50
        assert result.usage.output == 50

    def test_creates_metrics_with_success_true_and_no_usage_when_usage_missing(self):
        """Should create metrics with success=True and no usage when usage is missing."""
        mock_response = MagicMock()
        mock_response.usage = None

        result = OpenAIProvider.get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is None

    def test_handles_partial_usage_data(self):
        """Should handle partial usage data."""
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = None
        mock_response.usage.total_tokens = None

        result = OpenAIProvider.get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is not None
        assert result.usage.total == 0
        assert result.usage.input == 30
        assert result.usage.output == 0


class TestInvokeModel:
    """Tests for invoke_model instance method."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_invokes_openai_chat_completions_and_returns_response(self, mock_client):
        """Should invoke OpenAI chat completions and return response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = 'Hello! How can I help you today?'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25

        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(mock_client, 'gpt-3.5-turbo', {})
        messages = [LDMessage(role='user', content='Hello!')]
        result = await provider.invoke_model(messages)

        mock_client.chat.completions.create.assert_called_once_with(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': 'Hello!'}],
        )

        assert result.message.role == 'assistant'
        assert result.message.content == 'Hello! How can I help you today?'
        assert result.metrics.success is True
        assert result.metrics.usage is not None
        assert result.metrics.usage.total == 25
        assert result.metrics.usage.input == 10
        assert result.metrics.usage.output == 15

    @pytest.mark.asyncio
    async def test_returns_unsuccessful_response_when_no_content(self, mock_client):
        """Should return unsuccessful response when no content in response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.usage = None

        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(mock_client, 'gpt-3.5-turbo', {})
        messages = [LDMessage(role='user', content='Hello!')]
        result = await provider.invoke_model(messages)

        assert result.message.role == 'assistant'
        assert result.message.content == ''
        assert result.metrics.success is False

    @pytest.mark.asyncio
    async def test_returns_unsuccessful_response_when_choices_empty(self, mock_client):
        """Should return unsuccessful response when choices array is empty."""
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.usage = None

        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(mock_client, 'gpt-3.5-turbo', {})
        messages = [LDMessage(role='user', content='Hello!')]
        result = await provider.invoke_model(messages)

        assert result.message.role == 'assistant'
        assert result.message.content == ''
        assert result.metrics.success is False

    @pytest.mark.asyncio
    async def test_returns_unsuccessful_response_when_exception_thrown(self, mock_client):
        """Should return unsuccessful response when exception is thrown."""
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception('API Error'))

        provider = OpenAIProvider(mock_client, 'gpt-3.5-turbo', {})
        messages = [LDMessage(role='user', content='Hello!')]
        result = await provider.invoke_model(messages)

        assert result.message.role == 'assistant'
        assert result.message.content == ''
        assert result.metrics.success is False


class TestInvokeStructuredModel:
    """Tests for invoke_structured_model instance method."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_invokes_openai_with_structured_output(self, mock_client):
        """Should invoke OpenAI with structured output and return parsed response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = '{"name": "John", "age": 30, "city": "New York"}'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30

        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(mock_client, 'gpt-3.5-turbo', {})
        messages = [LDMessage(role='user', content='Tell me about a person')]
        response_structure = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'age': {'type': 'number'},
                'city': {'type': 'string'},
            },
            'required': ['name', 'age', 'city'],
        }

        result = await provider.invoke_structured_model(messages, response_structure)

        assert result.data == {'name': 'John', 'age': 30, 'city': 'New York'}
        assert result.raw_response == '{"name": "John", "age": 30, "city": "New York"}'
        assert result.metrics.success is True
        assert result.metrics.usage is not None
        assert result.metrics.usage.total == 30
        assert result.metrics.usage.input == 20
        assert result.metrics.usage.output == 10

    @pytest.mark.asyncio
    async def test_returns_unsuccessful_when_no_content_in_structured_response(self, mock_client):
        """Should return unsuccessful response when no content in structured response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.usage = None

        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(mock_client, 'gpt-3.5-turbo', {})
        messages = [LDMessage(role='user', content='Tell me about a person')]
        response_structure = {'type': 'object'}

        result = await provider.invoke_structured_model(messages, response_structure)

        assert result.data == {}
        assert result.raw_response == ''
        assert result.metrics.success is False

    @pytest.mark.asyncio
    async def test_handles_json_parsing_errors(self, mock_client):
        """Should handle JSON parsing errors gracefully."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = 'invalid json content'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(mock_client, 'gpt-3.5-turbo', {})
        messages = [LDMessage(role='user', content='Tell me about a person')]
        response_structure = {'type': 'object'}

        result = await provider.invoke_structured_model(messages, response_structure)

        assert result.data == {}
        assert result.raw_response == 'invalid json content'
        assert result.metrics.success is False
        assert result.metrics.usage is not None
        assert result.metrics.usage.total == 15

    @pytest.mark.asyncio
    async def test_returns_unsuccessful_response_when_exception_thrown(self, mock_client):
        """Should return unsuccessful response when exception is thrown."""
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception('API Error'))

        provider = OpenAIProvider(mock_client, 'gpt-3.5-turbo', {})
        messages = [LDMessage(role='user', content='Tell me about a person')]
        response_structure = {'type': 'object'}

        result = await provider.invoke_structured_model(messages, response_structure)

        assert result.data == {}
        assert result.raw_response == ''
        assert result.metrics.success is False


class TestGetClient:
    """Tests for get_client instance method."""

    def test_returns_underlying_client(self):
        """Should return the underlying OpenAI client."""
        mock_client = MagicMock()
        provider = OpenAIProvider(mock_client, 'gpt-3.5-turbo', {})

        assert provider.get_client() is mock_client


class TestCreate:
    """Tests for create static factory method."""

    @pytest.mark.asyncio
    async def test_creates_provider_with_correct_model_and_parameters(self):
        """Should create OpenAIProvider with correct model and parameters."""
        mock_ai_config = MagicMock()
        mock_ai_config.to_dict.return_value = {
            'model': {
                'name': 'gpt-4',
                'parameters': {
                    'temperature': 0.7,
                    'max_tokens': 1000,
                },
            },
            'provider': {'name': 'openai'},
        }

        with patch('ldai_openai.openai_provider.AsyncOpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            result = await OpenAIProvider.create(mock_ai_config)

            assert isinstance(result, OpenAIProvider)
            assert result._model_name == 'gpt-4'
            assert result._parameters == {'temperature': 0.7, 'max_tokens': 1000}

    @pytest.mark.asyncio
    async def test_handles_missing_model_config(self):
        """Should handle missing model configuration."""
        mock_ai_config = MagicMock()
        mock_ai_config.to_dict.return_value = {}

        with patch('ldai_openai.openai_provider.AsyncOpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            result = await OpenAIProvider.create(mock_ai_config)

            assert isinstance(result, OpenAIProvider)
            assert result._model_name == ''
            assert result._parameters == {}

