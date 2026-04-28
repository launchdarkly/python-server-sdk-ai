"""Tests for OpenAI Provider."""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from ldai import LDMessage

from ldai_openai import OpenAIModelRunner, OpenAIRunnerFactory, get_ai_metrics_from_response, get_ai_usage_from_response


def _make_usage(attrs: dict):
    """Build a simple namespace with only the given attributes (no MagicMock auto-attrs)."""
    class _Usage:
        pass
    u = _Usage()
    for k, v in attrs.items():
        setattr(u, k, v)
    return u


def _make_completions_response(total=100, prompt=50, completion=50):
    """Build a mock chat completions response with .usage (chat completions field names)."""
    mock = MagicMock()
    mock.context_wrapper = None
    mock.usage = _make_usage({
        'total_tokens': total,
        'prompt_tokens': prompt,
        'completion_tokens': completion,
    })
    return mock


def _make_runner_result(total=100, input_tokens=50, output_tokens=50):
    """Build a mock openai-agents RunResult with .context_wrapper.usage (agents field names)."""
    mock_ctx = MagicMock()
    mock_ctx.usage = _make_usage({
        'total_tokens': total,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
    })
    mock = MagicMock()
    mock.context_wrapper = mock_ctx
    mock.usage = None
    return mock


class TestGetAIUsageFromResponse:
    """Tests for OpenAIHelper.get_ai_usage_from_response."""

    def test_returns_usage_from_chat_completions_response(self):
        u = get_ai_usage_from_response(_make_completions_response(total=100, prompt=50, completion=50))
        assert u is not None
        assert u.total == 100
        assert u.input == 50
        assert u.output == 50

    def test_returns_usage_from_runner_result(self):
        u = get_ai_usage_from_response(_make_runner_result(total=43, input_tokens=30, output_tokens=13))
        assert u is not None
        assert u.total == 43
        assert u.input == 30
        assert u.output == 13

    def test_returns_none_when_usage_missing(self):
        mock_response = MagicMock()
        mock_response.usage = None
        mock_response.context_wrapper = None
        assert get_ai_usage_from_response(mock_response) is None

    def test_returns_none_when_all_counts_zero(self):
        u = get_ai_usage_from_response(_make_completions_response(total=0, prompt=0, completion=0))
        assert u is None

    def test_zero_input_tokens_not_conflated_with_missing(self):
        """input_tokens=0 should be used as-is, not fall through to prompt_tokens."""
        u = get_ai_usage_from_response(
            _make_runner_result(total=10, input_tokens=0, output_tokens=10)
        )
        assert u is not None
        assert u.input == 0
        assert u.output == 10


class TestGetAIMetricsFromResponse:
    """Tests for get_ai_metrics_from_response."""

    def test_creates_metrics_with_success_true_and_token_usage(self):
        """Should create metrics with success=True and token usage."""
        result = get_ai_metrics_from_response(_make_completions_response(total=100, prompt=50, completion=50))
        assert result.success is True
        assert result.usage is not None
        assert result.usage.total == 100
        assert result.usage.input == 50
        assert result.usage.output == 50

    def test_creates_metrics_with_success_true_and_no_usage_when_usage_missing(self):
        """Should create metrics with success=True and no usage when usage is missing."""
        mock_response = MagicMock()
        mock_response.usage = None
        mock_response.context_wrapper = None

        result = get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is None

    def test_handles_partial_usage_data(self):
        """Should handle partial usage data."""
        mock_response = MagicMock()
        mock_response.context_wrapper = None
        mock_response.usage = _make_usage({'prompt_tokens': 30, 'completion_tokens': None, 'total_tokens': None})

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
        mock_response.context_wrapper = None
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = 'Hello! How can I help you today?'
        mock_response.usage = _make_usage({'total_tokens': 25, 'prompt_tokens': 10, 'completion_tokens': 15})

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
        mock_response.context_wrapper = None
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = '{"name": "John", "age": 30, "city": "New York"}'
        mock_response.usage = _make_usage({'total_tokens': 30, 'prompt_tokens': 20, 'completion_tokens': 10})

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
        mock_response.context_wrapper = None
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = 'invalid json content'
        mock_response.usage = _make_usage({'total_tokens': 15, 'prompt_tokens': 10, 'completion_tokens': 5})

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


def _make_agents_mock(runner_run_mock: Any) -> MagicMock:
    """Build a mock ``agents`` module with Agent, Runner, FunctionTool, ModelSettings."""
    mock_runner_cls = MagicMock()
    mock_runner_cls.run = runner_run_mock

    mock_tool_context_module = MagicMock()
    mock_tool_context_module.ToolContext = MagicMock()

    agents_mock = MagicMock()
    agents_mock.Agent = MagicMock()
    agents_mock.Runner = mock_runner_cls
    agents_mock.FunctionTool = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    agents_mock.ModelSettings = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

    return agents_mock, mock_tool_context_module


class TestOpenAIAgentRunner:
    """Tests for OpenAIAgentRunner.run."""

    def _make_run_result(self, output: str, total: int = 15, input_tokens: int = 10, output_tokens: int = 5):
        """Build a mock RunResult with final_output and context_wrapper.usage."""
        mock_usage = MagicMock()
        mock_usage.total_tokens = total
        mock_usage.input_tokens = input_tokens
        mock_usage.output_tokens = output_tokens

        mock_ctx = MagicMock()
        mock_ctx.usage = mock_usage

        mock_result = MagicMock()
        mock_result.final_output = output
        mock_result.context_wrapper = mock_ctx
        return mock_result

    @pytest.mark.asyncio
    async def test_runs_agent_and_returns_result_with_no_tool_calls(self):
        """Should return RunnerResult when Runner.run returns a final output."""
        import sys

        from ldai_openai import OpenAIAgentRunner

        mock_run_result = self._make_run_result("The answer is 42.", total=15, input_tokens=10, output_tokens=5)
        mock_run_result.new_items = []
        agents_mock, tc_mock = _make_agents_mock(AsyncMock(return_value=mock_run_result))

        runner = OpenAIAgentRunner('gpt-4', {}, 'You are helpful.', [], {})
        with patch.dict(sys.modules, {'agents': agents_mock, 'agents.tool_context': tc_mock}):
            result = await runner.run("What is the answer?")

        assert result.content == "The answer is 42."
        assert result.metrics.success is True
        assert result.metrics.usage is not None
        assert result.metrics.usage.total == 15

    @pytest.mark.asyncio
    async def test_executes_tool_calls_and_returns_final_response(self):
        """Should delegate tool-calling loop to Runner.run and return final output."""
        import sys

        from ldai_openai import OpenAIAgentRunner

        mock_run_result = self._make_run_result("It is sunny in Paris.", total=43, input_tokens=30, output_tokens=13)
        mock_run_result.new_items = []
        agents_mock, tc_mock = _make_agents_mock(AsyncMock(return_value=mock_run_result))

        weather_fn = MagicMock(return_value="Sunny, 25°C")
        runner = OpenAIAgentRunner(
            'gpt-4', {}, 'You are helpful.',
            [{'name': 'get-weather', 'description': 'Get weather', 'parameters': {}}],
            {'get-weather': weather_fn},
        )
        with patch.dict(sys.modules, {'agents': agents_mock, 'agents.tool_context': tc_mock}):
            result = await runner.run("What is the weather in Paris?")

        assert result.content == "It is sunny in Paris."
        assert result.metrics.success is True
        assert result.metrics.usage.total == 43

    @pytest.mark.asyncio
    async def test_returns_failure_when_exception_thrown(self):
        """Should return unsuccessful RunnerResult when Runner.run raises."""
        import sys

        from ldai_openai import OpenAIAgentRunner

        agents_mock, tc_mock = _make_agents_mock(AsyncMock(side_effect=Exception("API Error")))

        runner = OpenAIAgentRunner('gpt-4', {}, '', [], {})
        with patch.dict(sys.modules, {'agents': agents_mock, 'agents.tool_context': tc_mock}):
            result = await runner.run("Hello")

        assert result.content == ""
        assert result.metrics.success is False

    @pytest.mark.asyncio
    async def test_returns_failure_when_openai_agents_not_installed(self):
        """Should return unsuccessful RunnerResult when openai-agents is not installed."""
        import sys

        from ldai_openai import OpenAIAgentRunner

        runner = OpenAIAgentRunner('gpt-4', {}, '', [], {})
        with patch.dict(sys.modules, {'agents': None}):
            result = await runner.run("Hello")

        assert result.content == ""
        assert result.metrics.success is False
