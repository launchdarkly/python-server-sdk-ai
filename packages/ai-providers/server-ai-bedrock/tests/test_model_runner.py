"""Tests for BedrockModelRunner."""

import pytest
from unittest.mock import MagicMock

from ldai_bedrock import BedrockModelRunner


def _make_response(text: str = 'Hello!', *, status: int = 200,
                   tokens: tuple = (25, 10, 15), latency_ms: int = 50,
                   tool_use: dict = None) -> dict:
    """Build a synthetic Bedrock Converse response dict."""
    total, inp, out = tokens
    content_blocks = []
    if text is not None:
        content_blocks.append({'text': text})
    if tool_use is not None:
        content_blocks.append({'toolUse': tool_use})
    return {
        'ResponseMetadata': {'HTTPStatusCode': status},
        'output': {
            'message': {
                'role': 'assistant',
                'content': content_blocks,
            },
        },
        'usage': {
            'totalTokens': total,
            'inputTokens': inp,
            'outputTokens': out,
        },
        'metrics': {'latencyMs': latency_ms},
        'stopReason': 'end_turn',
    }


class TestRunCompletion:
    """Tests for the run() method (chat-completion path)."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_invokes_converse_and_returns_response(self, mock_client):
        mock_client.converse = MagicMock(return_value=_make_response('Hello! How can I help you today?'))

        runner = BedrockModelRunner(
            mock_client,
            'anthropic.claude-3-5-sonnet-20240620-v1:0',
            {'temperature': 0.5, 'maxTokens': 200},
        )
        result = await runner.run('Hello!')

        # The model invocation should request inferenceConfig with the
        # Converse-recognised keys, the user prompt as a Bedrock message, and
        # the expected modelId.
        mock_client.converse.assert_called_once()
        kwargs = mock_client.converse.call_args.kwargs
        assert kwargs['modelId'] == 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        assert kwargs['messages'] == [
            {'role': 'user', 'content': [{'text': 'Hello!'}]},
        ]
        assert kwargs['inferenceConfig'] == {'temperature': 0.5, 'maxTokens': 200}
        assert 'toolConfig' not in kwargs

        assert result.content == 'Hello! How can I help you today?'
        assert result.metrics.success is True
        assert result.metrics.tokens is not None
        assert result.metrics.tokens.total == 25
        assert result.metrics.duration_ms == 50

    @pytest.mark.asyncio
    async def test_returns_unsuccessful_response_when_no_content(self, mock_client):
        mock_client.converse = MagicMock(return_value={
            'ResponseMetadata': {'HTTPStatusCode': 200},
            'output': {'message': {'role': 'assistant', 'content': []}},
            'usage': {'totalTokens': 0, 'inputTokens': 0, 'outputTokens': 0},
        })

        runner = BedrockModelRunner(mock_client, 'model-id', {})
        result = await runner.run('Hello!')

        assert result.content == ''
        assert result.metrics.success is False

    @pytest.mark.asyncio
    async def test_returns_unsuccessful_response_when_exception_thrown(self, mock_client):
        mock_client.converse = MagicMock(side_effect=Exception('AWS error'))

        runner = BedrockModelRunner(mock_client, 'model-id', {})
        result = await runner.run('Hello!')

        assert result.content == ''
        assert result.metrics.success is False

    @pytest.mark.asyncio
    async def test_passes_through_tool_config_when_tools_present(self, mock_client):
        mock_client.converse = MagicMock(return_value=_make_response())

        runner = BedrockModelRunner(
            mock_client,
            'model-id',
            {
                'temperature': 0.2,
                'tools': [
                    {'name': 'get_time', 'description': 'Return the current time.', 'parameters': {'type': 'object'}},
                ],
            },
        )
        await runner.run('What time is it?')

        kwargs = mock_client.converse.call_args.kwargs
        assert kwargs['toolConfig'] == {
            'tools': [
                {
                    'toolSpec': {
                        'name': 'get_time',
                        'description': 'Return the current time.',
                        'inputSchema': {'json': {'type': 'object'}},
                    },
                },
            ],
        }
        # ``tools`` should not leak into inferenceConfig or additional fields.
        assert kwargs['inferenceConfig'] == {'temperature': 0.2}
        assert 'additionalModelRequestFields' not in kwargs

    @pytest.mark.asyncio
    async def test_routes_unknown_parameters_to_additional_fields(self, mock_client):
        mock_client.converse = MagicMock(return_value=_make_response())

        runner = BedrockModelRunner(
            mock_client,
            'model-id',
            {'temperature': 0.1, 'top_k': 50, 'reasoning_budget': 1024},
        )
        await runner.run('Question')

        kwargs = mock_client.converse.call_args.kwargs
        assert kwargs['inferenceConfig'] == {'temperature': 0.1}
        assert kwargs['additionalModelRequestFields'] == {
            'top_k': 50,
            'reasoning_budget': 1024,
        }

    @pytest.mark.asyncio
    async def test_accumulates_history_across_successful_calls(self, mock_client):
        mock_client.converse = MagicMock(side_effect=[
            _make_response('First response'),
            _make_response('Second response'),
        ])

        runner = BedrockModelRunner(mock_client, 'model-id', {})
        await runner.run('First question')
        await runner.run('Second question')

        second_call_messages = mock_client.converse.call_args_list[1].kwargs['messages']
        assert second_call_messages == [
            {'role': 'user', 'content': [{'text': 'First question'}]},
            {'role': 'assistant', 'content': [{'text': 'First response'}]},
            {'role': 'user', 'content': [{'text': 'Second question'}]},
        ]

    @pytest.mark.asyncio
    async def test_multi_turn_false_does_not_accumulate_history(self, mock_client):
        mock_client.converse = MagicMock(side_effect=[
            _make_response('First response'),
            _make_response('Second response'),
        ])

        runner = BedrockModelRunner(mock_client, 'model-id', {}, multi_turn=False)
        baseline_len = len(runner._history)

        await runner.run('First question')
        assert len(runner._history) == baseline_len

        await runner.run('Second question')
        assert len(runner._history) == baseline_len

        second_call_messages = mock_client.converse.call_args_list[1].kwargs['messages']
        assert second_call_messages == [
            {'role': 'user', 'content': [{'text': 'Second question'}]},
        ]

    @pytest.mark.asyncio
    async def test_does_not_accumulate_history_on_failed_call(self, mock_client):
        mock_client.converse = MagicMock(side_effect=[Exception('boom'), _make_response('Recovery')])

        runner = BedrockModelRunner(mock_client, 'model-id', {})
        await runner.run('Hello!')
        await runner.run('Try again')

        second_call_messages = mock_client.converse.call_args_list[1].kwargs['messages']
        assert second_call_messages == [
            {'role': 'user', 'content': [{'text': 'Try again'}]},
        ]


class TestRunStructured:
    """Tests for the structured-output path of run()."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_returns_parsed_json_when_schema_supplied(self, mock_client):
        mock_client.converse = MagicMock(
            return_value=_make_response('{"name": "John", "age": 30}'),
        )

        runner = BedrockModelRunner(mock_client, 'model-id', {})
        schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'age': {'type': 'number'},
            },
            'required': ['name', 'age'],
        }

        result = await runner.run('Tell me about a person', output_type=schema)

        assert result.parsed == {'name': 'John', 'age': 30}
        assert result.content == '{"name": "John", "age": 30}'
        assert result.metrics.success is True

        # The schema should be injected as a system-prompt block to steer Bedrock.
        kwargs = mock_client.converse.call_args.kwargs
        assert 'system' in kwargs
        assert any('Schema:' in block['text'] for block in kwargs['system'])

    @pytest.mark.asyncio
    async def test_returns_unsuccessful_when_response_is_not_valid_json(self, mock_client):
        mock_client.converse = MagicMock(return_value=_make_response('not json'))

        runner = BedrockModelRunner(mock_client, 'model-id', {})
        result = await runner.run('Question', output_type={'type': 'object'})

        assert result.parsed is None
        assert result.content == 'not json'
        assert result.metrics.success is False

    @pytest.mark.asyncio
    async def test_returns_unsuccessful_when_exception_thrown(self, mock_client):
        mock_client.converse = MagicMock(side_effect=Exception('AWS error'))

        runner = BedrockModelRunner(mock_client, 'model-id', {})
        result = await runner.run('Question', output_type={'type': 'object'})

        assert result.parsed is None
        assert result.content == ''
        assert result.metrics.success is False
