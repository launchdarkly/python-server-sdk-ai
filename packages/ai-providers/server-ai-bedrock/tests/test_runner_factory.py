"""Tests for BedrockRunnerFactory."""

import pytest
from unittest.mock import MagicMock, patch

from ldai_bedrock import BedrockAgentRunner, BedrockModelRunner, BedrockRunnerFactory


class TestGetClient:
    """Tests for get_client / client lazy-construction."""

    def test_returns_supplied_client(self):
        mock_client = MagicMock()
        factory = BedrockRunnerFactory(client=mock_client)
        assert factory.get_client() is mock_client

    def test_constructs_client_lazily_when_omitted(self):
        with patch('boto3.client') as mock_boto:
            built_client = MagicMock()
            mock_boto.return_value = built_client

            factory = BedrockRunnerFactory(region_name='us-east-1')
            client = factory.get_client()

            mock_boto.assert_called_once_with(
                'bedrock-runtime',
                region_name='us-east-1',
            )
            assert client is built_client

    def test_reuses_client_on_second_call(self):
        with patch('boto3.client') as mock_boto:
            mock_boto.return_value = MagicMock()
            factory = BedrockRunnerFactory()
            first = factory.get_client()
            second = factory.get_client()
            assert first is second
            assert mock_boto.call_count == 1


class TestCreateModel:
    """Tests for BedrockRunnerFactory.create_model."""

    def test_creates_runner_with_correct_model_and_parameters(self):
        mock_ai_config = MagicMock()
        mock_ai_config.messages = None
        mock_ai_config.to_dict.return_value = {
            'model': {
                'name': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
                'parameters': {
                    'temperature': 0.7,
                    'maxTokens': 1024,
                },
            },
            'provider': {'name': 'bedrock'},
        }

        mock_client = MagicMock()
        result = BedrockRunnerFactory(client=mock_client).create_model(mock_ai_config)

        assert isinstance(result, BedrockModelRunner)
        assert result._model_id == 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        assert result._parameters == {'temperature': 0.7, 'maxTokens': 1024}

    def test_handles_missing_model_config(self):
        mock_ai_config = MagicMock()
        mock_ai_config.messages = None
        mock_ai_config.to_dict.return_value = {}

        mock_client = MagicMock()
        result = BedrockRunnerFactory(client=mock_client).create_model(mock_ai_config)

        assert isinstance(result, BedrockModelRunner)
        assert result._model_id == ''
        assert result._parameters == {}


class TestCreateAgent:
    """Tests for BedrockRunnerFactory.create_agent."""

    def test_creates_agent_runner_with_instructions_and_tool_definitions(self):
        mock_ai_config = MagicMock()
        mock_ai_config.instructions = 'You are a helpful assistant.'
        mock_ai_config.to_dict.return_value = {
            'model': {
                'name': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
                'parameters': {
                    'temperature': 0.7,
                    'tools': [
                        {'name': 'get-weather', 'description': 'Get weather', 'parameters': {}},
                    ],
                },
            },
        }

        mock_client = MagicMock()
        factory = BedrockRunnerFactory(client=mock_client)
        result = factory.create_agent(mock_ai_config, {'get-weather': lambda loc: 'sunny'})

        assert isinstance(result, BedrockAgentRunner)
        assert result._model_id == 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        assert result._instructions == 'You are a helpful assistant.'
        # The ``tools`` parameter is consumed and surfaced via ``_tool_definitions``.
        assert result._parameters == {'temperature': 0.7}
        assert len(result._tool_definitions) == 1
        assert result._tool_definitions[0]['name'] == 'get-weather'

    def test_creates_agent_runner_with_no_tools(self):
        mock_ai_config = MagicMock()
        mock_ai_config.instructions = 'You are a helpful assistant.'
        mock_ai_config.to_dict.return_value = {
            'model': {'name': 'anthropic.claude-3-5-sonnet-20240620-v1:0', 'parameters': {}},
        }

        mock_client = MagicMock()
        factory = BedrockRunnerFactory(client=mock_client)
        result = factory.create_agent(mock_ai_config, {})

        assert isinstance(result, BedrockAgentRunner)
        assert result._tool_definitions == []


class TestCreateAgentGraph:
    """Tests for BedrockRunnerFactory.create_agent_graph."""

    def test_raises_not_implemented_error(self):
        factory = BedrockRunnerFactory(client=MagicMock())
        with pytest.raises(NotImplementedError):
            factory.create_agent_graph(MagicMock(), {})
