"""Tests for BedrockAgentRunner."""

import sys
import types

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ldai_bedrock import BedrockAgentRunner


def _make_strands_mock(invoke_async_mock):
    """
    Construct a stand-in ``strands`` module with the surface BedrockAgentRunner
    uses (``Agent``, ``tool`` decorator).
    """
    strands_mock = types.ModuleType('strands')

    agent_instances = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            agent_instances.append(self)

        async def invoke_async(self, prompt):
            return await invoke_async_mock(prompt, self.kwargs)

    strands_mock.Agent = FakeAgent  # type: ignore[attr-defined]
    strands_mock.tool = lambda fn: fn  # type: ignore[attr-defined]
    strands_mock._agent_instances = agent_instances  # type: ignore[attr-defined]
    return strands_mock


def _make_strands_result(content: str, *, total: int = 15, inp: int = 10, out: int = 5,
                        tool_metrics: dict = None):
    """Mock the public surface of a strands AgentResult we read in the runner."""
    metrics_obj = MagicMock()
    metrics_obj.accumulated_usage = {
        'totalTokens': total,
        'inputTokens': inp,
        'outputTokens': out,
    }
    metrics_obj.tool_metrics = tool_metrics or {}

    result = MagicMock()
    result.metrics = metrics_obj
    result.message = {
        'role': 'assistant',
        'content': [{'text': content}],
    }
    return result


class TestBedrockAgentRunner:
    """Tests for BedrockAgentRunner.run."""

    @pytest.mark.asyncio
    async def test_runs_agent_and_returns_result_with_no_tool_calls(self):
        invoke_mock = AsyncMock(return_value=_make_strands_result('The answer is 42.'))
        strands_mock = _make_strands_mock(invoke_mock)

        runner = BedrockAgentRunner(
            'anthropic.claude-3-5-sonnet-20240620-v1:0',
            {},
            'You are helpful.',
            [],
            {},
        )
        with patch.dict(sys.modules, {'strands': strands_mock}):
            result = await runner.run('What is the answer?')

        assert result.content == 'The answer is 42.'
        assert result.metrics.success is True
        assert result.metrics.tokens is not None
        assert result.metrics.tokens.total == 15
        assert result.metrics.tool_calls is None

    @pytest.mark.asyncio
    async def test_records_tool_calls_from_metrics(self):
        tool_metrics = {
            'get_weather': MagicMock(call_count=2),
        }
        invoke_mock = AsyncMock(return_value=_make_strands_result(
            'It is sunny.',
            tool_metrics=tool_metrics,
        ))
        strands_mock = _make_strands_mock(invoke_mock)

        runner = BedrockAgentRunner(
            'model-id',
            {},
            'You are helpful.',
            [{'name': 'get_weather', 'description': 'Get weather', 'parameters': {}}],
            {'get_weather': lambda loc: 'sunny'},
        )
        with patch.dict(sys.modules, {'strands': strands_mock}):
            result = await runner.run('Weather in Paris?')

        assert result.content == 'It is sunny.'
        assert result.metrics.success is True
        assert result.metrics.tool_calls == ['get_weather', 'get_weather']

    @pytest.mark.asyncio
    async def test_passes_instructions_and_tools_to_agent(self):
        invoke_mock = AsyncMock(return_value=_make_strands_result('done'))
        strands_mock = _make_strands_mock(invoke_mock)

        runner = BedrockAgentRunner(
            'model-id',
            {},
            'Be terse.',
            [{'name': 'lookup', 'description': 'Look up', 'parameters': {}}],
            {'lookup': lambda q: 'answer'},
        )
        with patch.dict(sys.modules, {'strands': strands_mock}):
            await runner.run('go')

        assert len(strands_mock._agent_instances) == 1
        kwargs = strands_mock._agent_instances[0].kwargs
        assert kwargs['model'] == 'model-id'
        assert kwargs['system_prompt'] == 'Be terse.'
        assert len(kwargs['tools']) == 1

    @pytest.mark.asyncio
    async def test_returns_failure_when_exception_thrown(self):
        invoke_mock = AsyncMock(side_effect=Exception('boom'))
        strands_mock = _make_strands_mock(invoke_mock)

        runner = BedrockAgentRunner('model-id', {}, '', [], {})
        with patch.dict(sys.modules, {'strands': strands_mock}):
            result = await runner.run('Hello')

        assert result.content == ''
        assert result.metrics.success is False

    @pytest.mark.asyncio
    async def test_returns_failure_when_strands_not_installed(self):
        runner = BedrockAgentRunner('model-id', {}, '', [], {})
        with patch.dict(sys.modules, {'strands': None}):
            result = await runner.run('Hello')

        assert result.content == ''
        assert result.metrics.success is False
