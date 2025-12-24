"""Tests for Vercel Provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ldai import LDMessage

from ldai_vercel import VercelProvider, VercelModelParameters, VercelSDKMapOptions


class TestGetAIMetricsFromResponse:
    """Tests for get_ai_metrics_from_response static method."""

    def test_creates_metrics_with_success_true_and_token_usage(self):
        """Should create metrics with success=True and token usage."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 100

        result = VercelProvider.get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is not None
        assert result.usage.total == 100
        assert result.usage.input == 50
        assert result.usage.output == 50

    def test_creates_metrics_with_success_true_and_no_usage_when_usage_missing(self):
        """Should create metrics with success=True and no usage when usage is missing."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = None

        result = VercelProvider.get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is None

    def test_handles_partial_usage_data(self):
        """Should handle partial usage data."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = None
        mock_response.usage.total_tokens = None

        result = VercelProvider.get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is not None
        assert result.usage.total == 0
        assert result.usage.input == 30
        assert result.usage.output == 0

    def test_returns_success_false_for_error_finish_reason(self):
        """Should return success=False for error finish reason."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = 'error'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 100

        result = VercelProvider.get_ai_metrics_from_response(mock_response)

        assert result.success is False
        assert result.usage is not None
        assert result.usage.total == 100


class TestInvokeModel:
    """Tests for invoke_model instance method."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_invokes_litellm_and_returns_response(self, mock_logger):
        """Should invoke LiteLLM and return response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = 'Hello! How can I help you today?'
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25

        with patch('ldai_vercel.vercel_provider.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            provider = VercelProvider('openai/gpt-3.5-turbo', VercelModelParameters(), mock_logger)
            messages = [LDMessage(role='user', content='Hello!')]
            result = await provider.invoke_model(messages)

            mock_acompletion.assert_called_once_with(
                model='openai/gpt-3.5-turbo',
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
    async def test_handles_response_without_usage_data(self, mock_logger):
        """Should handle response without usage data."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = 'Hello! How can I help you today?'
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = None

        with patch('ldai_vercel.vercel_provider.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            provider = VercelProvider('openai/gpt-3.5-turbo', VercelModelParameters(), mock_logger)
            messages = [LDMessage(role='user', content='Hello!')]
            result = await provider.invoke_model(messages)

            assert result.message.role == 'assistant'
            assert result.message.content == 'Hello! How can I help you today?'
            assert result.metrics.success is True
            assert result.metrics.usage is None

    @pytest.mark.asyncio
    async def test_handles_errors_and_returns_failure_metrics(self, mock_logger):
        """Should handle errors and return failure metrics."""
        with patch('ldai_vercel.vercel_provider.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = Exception('API call failed')

            provider = VercelProvider('openai/gpt-3.5-turbo', VercelModelParameters(), mock_logger)
            messages = [LDMessage(role='user', content='Hello!')]
            result = await provider.invoke_model(messages)

            mock_logger.warn.assert_called()
            assert result.message.role == 'assistant'
            assert result.message.content == ''
            assert result.metrics.success is False


class TestInvokeStructuredModel:
    """Tests for invoke_structured_model instance method."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_invokes_litellm_with_structured_output(self, mock_logger):
        """Should invoke LiteLLM with structured output and return parsed response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = '{"name": "John Doe", "age": 30, "isActive": true}'
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25

        with patch('ldai_vercel.vercel_provider.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            provider = VercelProvider('openai/gpt-3.5-turbo', VercelModelParameters(), mock_logger)
            messages = [LDMessage(role='user', content='Generate user data')]
            response_structure = {'name': 'string', 'age': 0, 'isActive': True}

            result = await provider.invoke_structured_model(messages, response_structure)

            assert result.data == {'name': 'John Doe', 'age': 30, 'isActive': True}
            assert result.raw_response == '{"name": "John Doe", "age": 30, "isActive": true}'
            assert result.metrics.success is True
            assert result.metrics.usage is not None
            assert result.metrics.usage.total == 25

    @pytest.mark.asyncio
    async def test_handles_structured_response_without_usage_data(self, mock_logger):
        """Should handle structured response without usage data."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = '{"result": "success"}'
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = None

        with patch('ldai_vercel.vercel_provider.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            provider = VercelProvider('openai/gpt-3.5-turbo', VercelModelParameters(), mock_logger)
            messages = [LDMessage(role='user', content='Generate result')]
            response_structure = {'result': 'string'}

            result = await provider.invoke_structured_model(messages, response_structure)

            assert result.data == {'result': 'success'}
            assert result.metrics.success is True
            assert result.metrics.usage is None

    @pytest.mark.asyncio
    async def test_handles_errors_and_returns_failure_metrics(self, mock_logger):
        """Should handle errors and return failure metrics."""
        with patch('ldai_vercel.vercel_provider.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = Exception('API call failed')

            provider = VercelProvider('openai/gpt-3.5-turbo', VercelModelParameters(), mock_logger)
            messages = [LDMessage(role='user', content='Generate result')]
            response_structure = {'result': 'string'}

            result = await provider.invoke_structured_model(messages, response_structure)

            mock_logger.warn.assert_called()
            assert result.data == {}
            assert result.raw_response == ''
            assert result.metrics.success is False

    @pytest.mark.asyncio
    async def test_handles_invalid_json_response(self, mock_logger):
        """Should handle invalid JSON response gracefully."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = 'invalid json content'
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        with patch('ldai_vercel.vercel_provider.acompletion', new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            provider = VercelProvider('openai/gpt-3.5-turbo', VercelModelParameters(), mock_logger)
            messages = [LDMessage(role='user', content='Generate result')]
            response_structure = {'result': 'string'}

            result = await provider.invoke_structured_model(messages, response_structure)

            assert result.data == {}
            assert result.raw_response == 'invalid json content'
            assert result.metrics.success is False
            mock_logger.warn.assert_called()


class TestGetModelName:
    """Tests for get_model_name instance method."""

    def test_returns_model_name(self):
        """Should return the model name."""
        provider = VercelProvider('openai/gpt-4', VercelModelParameters())
        assert provider.get_model_name() == 'openai/gpt-4'


class TestMapProvider:
    """Tests for map_provider static method."""

    def test_maps_gemini_to_gemini(self):
        """Should map gemini to gemini."""
        assert VercelProvider.map_provider('gemini') == 'gemini'
        assert VercelProvider.map_provider('Gemini') == 'gemini'
        assert VercelProvider.map_provider('GEMINI') == 'gemini'

    def test_maps_google_to_gemini(self):
        """Should map google to gemini."""
        assert VercelProvider.map_provider('google') == 'gemini'

    def test_returns_provider_name_unchanged_for_standard_providers(self):
        """Should return provider name unchanged for standard providers."""
        assert VercelProvider.map_provider('openai') == 'openai'
        assert VercelProvider.map_provider('anthropic') == 'anthropic'
        assert VercelProvider.map_provider('cohere') == 'cohere'
        assert VercelProvider.map_provider('mistral') == 'mistral'

    def test_returns_provider_name_unchanged_for_unmapped_providers(self):
        """Should return provider name unchanged for unmapped providers."""
        assert VercelProvider.map_provider('unknown') == 'unknown'


class TestMapParameters:
    """Tests for map_parameters static method."""

    def test_maps_parameters_correctly(self):
        """Should map parameters correctly."""
        parameters = {
            'max_tokens': 100,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'presence_penalty': 0.1,
            'frequency_penalty': 0.2,
            'stop': ['stop1', 'stop2'],
            'seed': 42,
        }

        result = VercelProvider.map_parameters(parameters)

        assert result.max_tokens == 100
        assert result.temperature == 0.7
        assert result.top_p == 0.9
        assert result.top_k == 50
        assert result.presence_penalty == 0.1
        assert result.frequency_penalty == 0.2
        assert result.stop == ['stop1', 'stop2']
        assert result.seed == 42

    def test_handles_max_completion_tokens(self):
        """Should use max_completion_tokens if max_tokens is not present."""
        parameters = {
            'max_completion_tokens': 200,
        }

        result = VercelProvider.map_parameters(parameters)

        assert result.max_tokens == 200

    def test_prefers_max_tokens_over_max_completion_tokens(self):
        """Should prefer max_tokens over max_completion_tokens."""
        parameters = {
            'max_tokens': 100,
            'max_completion_tokens': 200,
        }

        result = VercelProvider.map_parameters(parameters)

        assert result.max_tokens == 100

    def test_returns_empty_parameters_for_none_input(self):
        """Should return empty parameters for None input."""
        result = VercelProvider.map_parameters(None)

        assert result.max_tokens is None
        assert result.temperature is None


class TestToLitellmConfig:
    """Tests for to_litellm_config static method."""

    def test_creates_config_with_correct_model_name(self):
        """Should create config with correct model name."""
        mock_ai_config = MagicMock()
        mock_ai_config.to_dict.return_value = {
            'model': {'name': 'gpt-4'},
            'provider': {'name': 'openai'},
        }

        result = VercelProvider.to_litellm_config(mock_ai_config)

        assert result.model == 'openai/gpt-4'

    def test_handles_missing_provider(self):
        """Should handle missing provider."""
        mock_ai_config = MagicMock()
        mock_ai_config.to_dict.return_value = {
            'model': {'name': 'gpt-4'},
        }

        result = VercelProvider.to_litellm_config(mock_ai_config)

        assert result.model == 'gpt-4'

    def test_merges_messages_and_non_interpolated_messages(self):
        """Should merge messages and non_interpolated_messages."""
        mock_ai_config = MagicMock()
        mock_ai_config.to_dict.return_value = {
            'model': {'name': 'gpt-4'},
            'provider': {'name': 'openai'},
            'messages': [{'role': 'user', 'content': 'Hello'}],
        }

        options = VercelSDKMapOptions(
            non_interpolated_messages=[LDMessage(role='assistant', content='Hi there')]
        )

        result = VercelProvider.to_litellm_config(mock_ai_config, options)

        assert len(result.messages) == 2
        assert result.messages[0].role == 'user'
        assert result.messages[0].content == 'Hello'
        assert result.messages[1].role == 'assistant'
        assert result.messages[1].content == 'Hi there'

    def test_maps_parameters(self):
        """Should map parameters correctly."""
        mock_ai_config = MagicMock()
        mock_ai_config.to_dict.return_value = {
            'model': {
                'name': 'gpt-4',
                'parameters': {
                    'max_tokens': 100,
                    'temperature': 0.7,
                },
            },
            'provider': {'name': 'openai'},
        }

        result = VercelProvider.to_litellm_config(mock_ai_config)

        assert result.max_tokens == 100
        assert result.temperature == 0.7


class TestCreateModelName:
    """Tests for create_model_name static method."""

    def test_creates_model_name_with_provider(self):
        """Should create model name with provider."""
        mock_ai_config = MagicMock()
        mock_ai_config.to_dict.return_value = {
            'model': {'name': 'gpt-4'},
            'provider': {'name': 'openai'},
        }

        result = VercelProvider.create_model_name(mock_ai_config)

        assert result == 'openai/gpt-4'

    def test_creates_model_name_without_provider(self):
        """Should create model name without provider."""
        mock_ai_config = MagicMock()
        mock_ai_config.to_dict.return_value = {
            'model': {'name': 'gpt-4'},
        }

        result = VercelProvider.create_model_name(mock_ai_config)

        assert result == 'gpt-4'

    def test_maps_provider_name(self):
        """Should map provider name."""
        mock_ai_config = MagicMock()
        mock_ai_config.to_dict.return_value = {
            'model': {'name': 'claude-3-opus'},
            'provider': {'name': 'anthropic'},
        }

        result = VercelProvider.create_model_name(mock_ai_config)

        assert result == 'anthropic/claude-3-opus'


class TestCreate:
    """Tests for create static factory method."""

    @pytest.mark.asyncio
    async def test_creates_provider_with_correct_model_and_parameters(self):
        """Should create VercelProvider with correct model and parameters."""
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

        result = await VercelProvider.create(mock_ai_config)

        assert isinstance(result, VercelProvider)
        assert result.get_model_name() == 'openai/gpt-4'
        assert result._parameters.temperature == 0.7
        assert result._parameters.max_tokens == 1000


class TestCreateAIMetrics:
    """Tests for deprecated create_ai_metrics static method."""

    def test_delegates_to_get_ai_metrics_from_response(self):
        """Should delegate to get_ai_metrics_from_response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 100

        result = VercelProvider.create_ai_metrics(mock_response)

        assert result.success is True
        assert result.usage is not None
        assert result.usage.total == 100


class TestVercelModelParameters:
    """Tests for VercelModelParameters dataclass."""

    def test_to_dict_excludes_none_values(self):
        """Should exclude None values from dict."""
        params = VercelModelParameters(
            max_tokens=100,
            temperature=0.7,
        )

        result = params.to_dict()

        assert result == {
            'max_tokens': 100,
            'temperature': 0.7,
        }

    def test_to_dict_returns_empty_for_all_none(self):
        """Should return empty dict for all None values."""
        params = VercelModelParameters()

        result = params.to_dict()

        assert result == {}

