"""Tests for LangChain Provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ldai import LDMessage

from ldai_langchain import (
    LangChainModelRunner,
    LangChainRunnerFactory,
    convert_messages_to_langchain,
    get_ai_metrics_from_response,
    get_ai_usage_from_response,
    get_tool_calls_from_response,
    map_provider,
    sum_token_usage_from_messages,
)


class TestConvertMessages:
    """Tests for convert_messages_to_langchain."""

    def test_converts_system_messages_to_system_message(self):
        """Should convert system messages to SystemMessage."""
        messages = [LDMessage(role='system', content='You are a helpful assistant.')]
        result = convert_messages_to_langchain(messages)

        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == 'You are a helpful assistant.'

    def test_converts_user_messages_to_human_message(self):
        """Should convert user messages to HumanMessage."""
        messages = [LDMessage(role='user', content='Hello, how are you?')]
        result = convert_messages_to_langchain(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == 'Hello, how are you?'

    def test_converts_assistant_messages_to_ai_message(self):
        """Should convert assistant messages to AIMessage."""
        messages = [LDMessage(role='assistant', content='I am doing well, thank you!')]
        result = convert_messages_to_langchain(messages)

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
        result = convert_messages_to_langchain(messages)

        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)

    def test_throws_error_for_unsupported_message_role(self):
        """Should throw error for unsupported message role."""
        class MockMessage:
            role = 'unknown'
            content = 'Test message'

        with pytest.raises(ValueError, match='Unsupported message role: unknown'):
            convert_messages_to_langchain([MockMessage()])  # type: ignore

    def test_handles_empty_message_array(self):
        """Should handle empty message array."""
        result = convert_messages_to_langchain([])
        assert len(result) == 0


class TestGetAIMetricsFromResponse:
    """Tests for get_ai_metrics_from_response."""

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

        result = get_ai_metrics_from_response(mock_response)

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

        result = get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is not None
        assert result.usage.total == 150
        assert result.usage.input == 75
        assert result.usage.output == 75

    def test_creates_metrics_with_success_true_and_no_usage_when_metadata_missing(self):
        """Should create metrics with success=True and no usage when metadata is missing."""
        mock_response = AIMessage(content='Test response')

        result = get_ai_metrics_from_response(mock_response)

        assert result.success is True
        assert result.usage is None

    def test_usage_metadata_preferred_over_response_metadata(self):
        """usage_metadata should be used when it has non-zero counts."""
        mock_response = AIMessage(content='Test')
        mock_response.usage_metadata = {
            'total_tokens': 10,
            'input_tokens': 4,
            'output_tokens': 6,
        }
        mock_response.response_metadata = {
            'tokenUsage': {
                'totalTokens': 999,
                'promptTokens': 500,
                'completionTokens': 499,
            },
        }
        usage = get_ai_usage_from_response(mock_response)
        assert usage is not None
        assert usage.total == 10
        assert usage.input == 4
        assert usage.output == 6


class TestGetAIUsageFromResponse:
    """Tests for LangChainHelper.get_ai_usage_from_response."""

    def test_returns_none_when_no_usage(self):
        msg = AIMessage(content='hi')
        assert get_ai_usage_from_response(msg) is None

    def test_returns_none_when_all_zeros_in_metadata(self):
        msg = AIMessage(content='hi')
        msg.usage_metadata = {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0}
        assert get_ai_usage_from_response(msg) is None


class TestGetToolCallsFromResponse:
    """Tests for LangChainHelper.get_tool_calls_from_response."""

    def test_returns_empty_when_no_tool_calls(self):
        msg = AIMessage(content='hi')
        assert get_tool_calls_from_response(msg) == []

    def test_returns_empty_when_tool_calls_not_a_sequence(self):
        msg = AIMessage(content='hi')
        msg.tool_calls = None  # type: ignore
        assert get_tool_calls_from_response(msg) == []

    def test_extracts_names_from_dict_tool_calls(self):
        msg = AIMessage(content='')
        msg.tool_calls = [  # type: ignore
            {'name': 'search', 'args': {}, 'id': '1'},
            {'name': 'calc', 'args': {}, 'id': '2'},
        ]
        assert get_tool_calls_from_response(msg) == ['search', 'calc']

    def test_returns_empty_when_tool_calls_is_not_a_list(self):
        msg = AIMessage(content='hi')
        msg.tool_calls = ()  # type: ignore
        assert get_tool_calls_from_response(msg) == []

    def test_skips_entries_without_name(self):
        msg = AIMessage(content='')
        msg.tool_calls = [{'name': 'a', 'id': '1'}, {}, {'name': 'b', 'id': '2'}]  # type: ignore
        assert get_tool_calls_from_response(msg) == ['a', 'b']


class TestMapProvider:
    """Tests for map_provider."""

    def test_maps_gemini_to_google_genai(self):
        """Should map gemini to google-genai."""
        assert map_provider('gemini') == 'google-genai'
        assert map_provider('Gemini') == 'google-genai'
        assert map_provider('GEMINI') == 'google-genai'

    def test_maps_bedrock_and_model_families_to_bedrock_converse(self):
        """Should map bedrock and bedrock:model_family to bedrock_converse."""
        assert map_provider('bedrock') == 'bedrock_converse'
        assert map_provider('Bedrock:Anthropic') == 'bedrock_converse'
        assert map_provider('bedrock:anthropic') == 'bedrock_converse'
        assert map_provider('bedrock:amazon') == 'bedrock_converse'
        assert map_provider('bedrock:cohere') == 'bedrock_converse'

    def test_returns_provider_name_unchanged_for_unmapped_providers(self):
        """Should return provider name unchanged for unmapped providers."""
        assert map_provider('openai') == 'openai'
        assert map_provider('anthropic') == 'anthropic'
        assert map_provider('unknown') == 'unknown'


class TestInvokeModel:
    """Tests for invoke_model instance method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_returns_success_true_for_string_content(self, mock_llm):
        """Should return success=True for string content."""
        mock_response = AIMessage(content='Test response')
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        provider = LangChainModelRunner(mock_llm)

        messages = [LDMessage(role='user', content='Hello')]
        result = await provider.invoke_model(messages)

        assert result.metrics.success is True
        assert result.message.content == 'Test response'

    @pytest.mark.asyncio
    async def test_returns_success_false_for_non_string_content_and_logs_warning(self, mock_llm):
        """Should return success=False for non-string content and log warning."""
        mock_response = AIMessage(content=[{'type': 'image', 'data': 'base64data'}])
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        provider = LangChainModelRunner(mock_llm)

        messages = [LDMessage(role='user', content='Hello')]
        result = await provider.invoke_model(messages)

        assert result.metrics.success is False
        assert result.message.content == ''

    @pytest.mark.asyncio
    async def test_returns_success_false_when_model_invocation_throws_error(self, mock_llm):
        """Should return success=False when model invocation throws an error."""
        error = Exception('Model invocation failed')
        mock_llm.ainvoke = AsyncMock(side_effect=error)
        provider = LangChainModelRunner(mock_llm)

        messages = [LDMessage(role='user', content='Hello')]
        result = await provider.invoke_model(messages)

        assert result.metrics.success is False
        assert result.message.content == ''
        assert result.message.role == 'assistant'


class TestInvokeStructuredModel:
    """Tests for invoke_structured_model instance method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_returns_success_true_for_successful_invocation(self, mock_llm):
        """Should return success=True for successful invocation."""
        parsed_data = {'result': 'structured data'}
        mock_response = {'parsed': parsed_data, 'raw': None}
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        provider = LangChainModelRunner(mock_llm)

        messages = [LDMessage(role='user', content='Hello')]
        response_structure = {'type': 'object', 'properties': {}}
        result = await provider.invoke_structured_model(messages, response_structure)

        assert result.metrics.success is True
        assert result.data == parsed_data

    @pytest.mark.asyncio
    async def test_returns_success_false_when_structured_model_invocation_throws_error(self, mock_llm):
        """Should return success=False when structured model invocation throws an error."""
        error = Exception('Structured invocation failed')
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(side_effect=error)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        provider = LangChainModelRunner(mock_llm)

        messages = [LDMessage(role='user', content='Hello')]
        response_structure = {'type': 'object', 'properties': {}}
        result = await provider.invoke_structured_model(messages, response_structure)

        assert result.metrics.success is False
        assert result.data == {}
        assert result.raw_response == ''
        assert result.metrics.usage is None


class TestGetToolCallsFromResponse:
    """Tests for get_tool_calls_from_response."""

    def test_returns_tool_call_names_in_order(self):
        """Should return tool call names from response.tool_calls."""
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {'name': 'search', 'args': {}},
            {'name': 'calculator', 'args': {}},
        ]
        assert get_tool_calls_from_response(mock_response) == ['search', 'calculator']

    def test_returns_empty_list_when_tool_calls_is_empty(self):
        """Should return empty list when tool_calls is an empty list."""
        mock_response = MagicMock()
        mock_response.tool_calls = []
        assert get_tool_calls_from_response(mock_response) == []

    def test_returns_empty_list_when_no_tool_calls_attribute(self):
        """Should return empty list when response has no tool_calls attribute."""
        mock_response = MagicMock(spec=[])
        assert get_tool_calls_from_response(mock_response) == []

    def test_returns_empty_list_when_tool_calls_is_not_a_list(self):
        """Should return empty list when tool_calls is not a list."""
        mock_response = MagicMock()
        mock_response.tool_calls = 'not-a-list'
        assert get_tool_calls_from_response(mock_response) == []

    def test_skips_tool_calls_without_name(self):
        """Should skip tool calls that have no name."""
        mock_response = MagicMock()
        mock_response.tool_calls = [{'args': {}}, {'name': 'search', 'args': {}}]
        assert get_tool_calls_from_response(mock_response) == ['search']


class TestSumTokenUsageFromMessages:
    """Tests for sum_token_usage_from_messages."""

    def test_sums_usage_across_messages(self):
        """Should sum token usage from all messages."""
        msg1 = AIMessage(content='a')
        msg1.usage_metadata = {'total_tokens': 10, 'input_tokens': 6, 'output_tokens': 4}
        msg2 = AIMessage(content='b')
        msg2.usage_metadata = {'total_tokens': 20, 'input_tokens': 12, 'output_tokens': 8}

        result = sum_token_usage_from_messages([msg1, msg2])

        assert result is not None
        assert result.total == 30
        assert result.input == 18
        assert result.output == 12

    def test_returns_none_when_no_usage_on_any_message(self):
        """Should return None when no message has usage metadata."""
        msg = AIMessage(content='hello')
        assert sum_token_usage_from_messages([msg]) is None

    def test_returns_none_for_empty_list(self):
        """Should return None for an empty message list."""
        assert sum_token_usage_from_messages([]) is None

    def test_skips_messages_without_usage(self):
        """Should skip messages that have no usage and sum the rest."""
        msg1 = AIMessage(content='a')
        msg2 = AIMessage(content='b')
        msg2.usage_metadata = {'total_tokens': 5, 'input_tokens': 3, 'output_tokens': 2}

        result = sum_token_usage_from_messages([msg1, msg2])

        assert result is not None
        assert result.total == 5
        assert result.input == 3
        assert result.output == 2


class TestGetLlm:
    """Tests for LangChainModelRunner.get_llm."""

    def test_returns_underlying_llm(self):
        """Should return the underlying LLM."""
        mock_llm = MagicMock()
        runner = LangChainModelRunner(mock_llm)

        assert runner.get_llm() is mock_llm


class TestCreateAgent:
    """Tests for LangChainRunnerFactory.create_agent."""

    def test_creates_agent_runner_with_instructions_and_tool_definitions(self):
        """Should create LangChainAgentRunner wrapping a compiled graph."""
        from unittest.mock import patch
        from ldai_langchain import LangChainAgentRunner

        mock_ai_config = MagicMock()
        mock_ai_config.instructions = "You are a helpful assistant."
        mock_ai_config.to_dict.return_value = {
            'model': {
                'name': 'gpt-4',
                'parameters': {
                    'tools': [
                        {'name': 'get-weather', 'description': 'Get weather', 'parameters': {}},
                    ],
                },
            },
            'provider': {'name': 'openai'},
        }

        mock_agent = MagicMock()
        with patch('ldai_langchain.langchain_runner_factory.create_langchain_model') as mock_create, \
             patch('ldai_langchain.langchain_runner_factory.build_tools') as mock_tools, \
             patch('langchain.agents.create_agent', return_value=mock_agent):
            mock_create.return_value = MagicMock()
            mock_tools.return_value = [MagicMock()]

            factory = LangChainRunnerFactory()
            result = factory.create_agent(mock_ai_config, {'get-weather': lambda loc: 'sunny'})

            assert isinstance(result, LangChainAgentRunner)
            assert result._agent is mock_agent

    def test_creates_agent_runner_with_no_tools(self):
        """Should create LangChainAgentRunner with no tool definitions."""
        from unittest.mock import patch
        from ldai_langchain import LangChainAgentRunner

        mock_ai_config = MagicMock()
        mock_ai_config.instructions = "You are a helpful assistant."
        mock_ai_config.to_dict.return_value = {
            'model': {'name': 'gpt-4', 'parameters': {}},
            'provider': {'name': 'openai'},
        }

        mock_agent = MagicMock()
        with patch('ldai_langchain.langchain_runner_factory.create_langchain_model') as mock_create, \
             patch('ldai_langchain.langchain_runner_factory.build_tools', return_value=[]), \
             patch('langchain.agents.create_agent', return_value=mock_agent):
            mock_create.return_value = MagicMock()

            factory = LangChainRunnerFactory()
            result = factory.create_agent(mock_ai_config, {})

            assert isinstance(result, LangChainAgentRunner)
            assert result._agent is mock_agent


class TestLangChainAgentRunner:
    """Tests for LangChainAgentRunner.run."""

    @pytest.mark.asyncio
    async def test_runs_agent_and_returns_result(self):
        """Should return AgentResult with the last message content from the graph."""
        from ldai_langchain import LangChainAgentRunner

        final_msg = AIMessage(content="The answer is 42.")
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [final_msg]})

        runner = LangChainAgentRunner(mock_agent)
        result = await runner.run("What is the answer?")

        assert result.output == "The answer is 42."
        assert result.metrics.success is True
        mock_agent.ainvoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "What is the answer?"}]}
        )

    @pytest.mark.asyncio
    async def test_aggregates_token_usage_across_messages(self):
        """Should sum token usage from all messages in the graph result."""
        from ldai_langchain import LangChainAgentRunner

        msg1 = AIMessage(content="intermediate")
        msg1.usage_metadata = {'total_tokens': 10, 'input_tokens': 6, 'output_tokens': 4}
        msg2 = AIMessage(content="final answer")
        msg2.usage_metadata = {'total_tokens': 20, 'input_tokens': 12, 'output_tokens': 8}

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [msg1, msg2]})

        runner = LangChainAgentRunner(mock_agent)
        result = await runner.run("Hello")

        assert result.output == "final answer"
        assert result.metrics.success is True
        assert result.metrics.usage is not None
        assert result.metrics.usage.total == 30
        assert result.metrics.usage.input == 18
        assert result.metrics.usage.output == 12

    @pytest.mark.asyncio
    async def test_returns_failure_when_exception_thrown(self):
        """Should return unsuccessful AgentResult when exception is thrown."""
        from ldai_langchain import LangChainAgentRunner

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=Exception("Graph Error"))

        runner = LangChainAgentRunner(mock_agent)
        result = await runner.run("Hello")

        assert result.output == ""
        assert result.metrics.success is False


class TestBuildTools:
    """Tests for build_structured_tools (sync vs async registry callables)."""

    def test_registers_sync_callable_as_structured_tool_func(self):
        from ldai.models import AIAgentConfig, ModelConfig, ProviderConfig
        from ldai_langchain.langchain_helper import build_structured_tools

        def sync_tool(x: str = '') -> str:
            return 'ok'

        cfg = AIAgentConfig(
            key='n',
            enabled=True,
            model=ModelConfig(
                name='gpt-4',
                parameters={'tools': [{'name': 'my_tool', 'type': 'function', 'parameters': {}}]},
            ),
            provider=ProviderConfig(name='openai'),
            instructions='',
        )
        tools = build_structured_tools(cfg, {'my_tool': sync_tool})
        assert len(tools) == 1
        assert tools[0].func is sync_tool
        assert getattr(tools[0], 'coroutine', None) is None

    def test_registers_async_callable_as_structured_tool_coroutine(self):
        from ldai.models import AIAgentConfig, ModelConfig, ProviderConfig
        from ldai_langchain.langchain_helper import build_structured_tools

        async def async_tool(x: str = '') -> str:
            return 'ok'

        cfg = AIAgentConfig(
            key='n',
            enabled=True,
            model=ModelConfig(
                name='gpt-4',
                parameters={'tools': [{'name': 'my_tool', 'type': 'function', 'parameters': {}}]},
            ),
            provider=ProviderConfig(name='openai'),
            instructions='',
        )
        tools = build_structured_tools(cfg, {'my_tool': async_tool})
        assert len(tools) == 1
        assert tools[0].coroutine is async_tool
