"""Tests for OpenAI Provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ldai import LDMessage

from ldai_openai import OpenAIModelRunner, OpenAIRunnerFactory, get_ai_metrics_from_response, get_ai_usage_from_response


class TestGetAIUsageFromResponse:
    """Tests for OpenAIHelper.get_ai_usage_from_response."""

    def test_returns_usage_when_present(self):
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 100
        u = get_ai_usage_from_response(mock_response)
        assert u is not None
        assert u.total == 100
        assert u.input == 50
        assert u.output == 50

    def test_returns_none_when_usage_missing(self):
        mock_response = MagicMock()
        mock_response.usage = None
        assert get_ai_usage_from_response(mock_response) is None

    def test_returns_none_when_all_counts_zero(self):
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 0
        mock_response.usage.prompt_tokens = 0
        mock_response.usage.completion_tokens = 0
        assert get_ai_usage_from_response(mock_response) is None


class TestGetAIMetricsFromResponse:
    """Tests for get_ai_metrics_from_response."""

    def test_creates_metrics_with_success_true_and_token_usage(self):
        """Should create metrics with success=True and token usage."""
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 100

        result = get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is not None
        assert result.usage.total == 100
        assert result.usage.input == 50
        assert result.usage.output == 50

    def test_creates_metrics_with_success_true_and_no_usage_when_usage_missing(self):
        """Should create metrics with success=True and no usage when usage is missing."""
        mock_response = MagicMock()
        mock_response.usage = None

        result = get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is None

    def test_handles_partial_usage_data(self):
        """Should handle partial usage data."""
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = None
        mock_response.usage.total_tokens = None

        result = get_ai_metrics_from_response(mock_response)

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

        provider = OpenAIModelRunner(mock_client, 'gpt-3.5-turbo', {})
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

        provider = OpenAIModelRunner(mock_client, 'gpt-3.5-turbo', {})
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

        provider = OpenAIModelRunner(mock_client, 'gpt-3.5-turbo', {})
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

        provider = OpenAIModelRunner(mock_client, 'gpt-3.5-turbo', {})
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

        provider = OpenAIModelRunner(mock_client, 'gpt-3.5-turbo', {})
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

        provider = OpenAIModelRunner(mock_client, 'gpt-3.5-turbo', {})
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

        provider = OpenAIModelRunner(mock_client, 'gpt-3.5-turbo', {})
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

        provider = OpenAIModelRunner(mock_client, 'gpt-3.5-turbo', {})
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
        provider = OpenAIRunnerFactory(mock_client)

        assert provider.get_client() is mock_client


class TestCreateModel:
    """Tests for create_model instance method."""

    def test_creates_connector_with_correct_model_and_parameters(self):
        """Should create OpenAIRunnerFactory with correct model and parameters."""
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

        with patch('ldai_openai.openai_runner_factory.AsyncOpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            result = OpenAIRunnerFactory().create_model(mock_ai_config)

            assert isinstance(result, OpenAIModelRunner)
            assert result._model_name == 'gpt-4'
            assert result._parameters == {'temperature': 0.7, 'max_tokens': 1000}

    def test_handles_missing_model_config(self):
        """Should handle missing model configuration."""
        mock_ai_config = MagicMock()
        mock_ai_config.to_dict.return_value = {}

        with patch('ldai_openai.openai_runner_factory.AsyncOpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            result = OpenAIRunnerFactory().create_model(mock_ai_config)

            assert isinstance(result, OpenAIModelRunner)
            assert result._model_name == ''
            assert result._parameters == {}


class TestCreateAgent:
    """Tests for OpenAIRunnerFactory.create_agent."""

    def test_creates_agent_runner_with_instructions_and_tool_definitions(self):
        """Should create OpenAIAgentRunner with instructions and tool definitions."""
        mock_ai_config = MagicMock()
        mock_ai_config.instructions = "You are a helpful assistant."
        mock_ai_config.to_dict.return_value = {
            'model': {
                'name': 'gpt-4',
                'parameters': {
                    'temperature': 0.7,
                    'tools': [
                        {'name': 'get-weather', 'description': 'Get weather', 'parameters': {}},
                    ],
                },
            },
        }

        mock_client = MagicMock()
        factory = OpenAIRunnerFactory(mock_client)
        result = factory.create_agent(mock_ai_config, {'get-weather': lambda loc: 'sunny'})

        from ldai_openai import OpenAIAgentRunner
        assert isinstance(result, OpenAIAgentRunner)
        assert result._model_name == 'gpt-4'
        assert result._instructions == "You are a helpful assistant."
        assert result._parameters == {'temperature': 0.7}
        assert len(result._tool_definitions) == 1
        assert result._tool_definitions[0]['name'] == 'get-weather'

    def test_creates_agent_runner_with_no_tools(self):
        """Should create OpenAIAgentRunner with no tool definitions."""
        mock_ai_config = MagicMock()
        mock_ai_config.instructions = "You are a helpful assistant."
        mock_ai_config.to_dict.return_value = {
            'model': {'name': 'gpt-4', 'parameters': {}},
        }

        mock_client = MagicMock()
        factory = OpenAIRunnerFactory(mock_client)
        result = factory.create_agent(mock_ai_config, {})

        from ldai_openai import OpenAIAgentRunner
        assert isinstance(result, OpenAIAgentRunner)
        assert result._tool_definitions == []


class TestOpenAIAgentRunner:
    """Tests for OpenAIAgentRunner.run."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_runs_agent_and_returns_result_with_no_tool_calls(self, mock_client):
        """Should return AgentResult when model responds with no tool calls."""
        from ldai_openai import OpenAIAgentRunner

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The answer is 42."
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        runner = OpenAIAgentRunner(mock_client, 'gpt-4', {}, 'You are helpful.', [], {})
        result = await runner.run("What is the answer?")

        assert result.output == "The answer is 42."
        assert result.metrics.success is True

    @pytest.mark.asyncio
    async def test_executes_tool_calls_and_returns_final_response(self, mock_client):
        """Should execute tool calls and continue loop until final response."""
        from ldai_openai import OpenAIAgentRunner

        # First response: has a tool call
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "get-weather"
        tool_call.function.arguments = '{"location": "Paris"}'

        first_response = MagicMock()
        first_response.choices = [MagicMock()]
        first_response.choices[0].message.content = None
        first_response.choices[0].message.tool_calls = [tool_call]
        first_response.usage = MagicMock()
        first_response.usage.prompt_tokens = 10
        first_response.usage.completion_tokens = 5
        first_response.usage.total_tokens = 15

        # Second response: final answer
        second_response = MagicMock()
        second_response.choices = [MagicMock()]
        second_response.choices[0].message.content = "It is sunny in Paris."
        second_response.choices[0].message.tool_calls = None
        second_response.usage = MagicMock()
        second_response.usage.prompt_tokens = 20
        second_response.usage.completion_tokens = 8
        second_response.usage.total_tokens = 28

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[first_response, second_response]
        )

        weather_fn = MagicMock(return_value="Sunny, 25°C")
        runner = OpenAIAgentRunner(
            mock_client, 'gpt-4', {}, 'You are helpful.',
            [{'name': 'get-weather', 'description': 'Get weather', 'parameters': {}}],
            {'get-weather': weather_fn},
        )
        result = await runner.run("What is the weather in Paris?")

        assert result.output == "It is sunny in Paris."
        assert result.metrics.success is True
        weather_fn.assert_called_once_with(location="Paris")
        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_returns_failure_when_exception_thrown(self, mock_client):
        """Should return unsuccessful AgentResult when exception is thrown."""
        from ldai_openai import OpenAIAgentRunner

        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        runner = OpenAIAgentRunner(mock_client, 'gpt-4', {}, '', [], {})
        result = await runner.run("Hello")

        assert result.output == ""
        assert result.metrics.success is False
