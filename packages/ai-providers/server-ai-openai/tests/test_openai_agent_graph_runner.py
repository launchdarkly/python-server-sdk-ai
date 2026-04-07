"""Tests for OpenAIAgentGraphRunner and OpenAIRunnerFactory.create_agent_graph()."""

import logging
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ldai.agent_graph import AgentGraphDefinition
from ldai.models import AIAgentGraphConfig, AIAgentConfig, Edge, ModelConfig, ProviderConfig
from ldai.providers import AgentGraphResult, ToolRegistry
from ldai_openai.openai_agent_graph_runner import OpenAIAgentGraphRunner, _make_span_hooks
from ldai_openai.openai_runner_factory import OpenAIRunnerFactory


def _make_graph(enabled: bool = True) -> AgentGraphDefinition:
    """Build a minimal single-node AgentGraphDefinition for testing."""
    root_config = AIAgentConfig(
        key='root-agent',
        enabled=enabled,
        model=ModelConfig(name='gpt-4'),
        provider=ProviderConfig(name='openai'),
        instructions='You are a helpful assistant.',
        tracker=MagicMock(),
    )
    graph_config = AIAgentGraphConfig(
        key='test-graph',
        root_config_key='root-agent',
        edges=[],
        enabled=enabled,
    )
    nodes = AgentGraphDefinition.build_nodes(graph_config, {'root-agent': root_config})
    return AgentGraphDefinition(
        agent_graph=graph_config,
        nodes=nodes,
        context=MagicMock(),
        enabled=enabled,
        tracker=MagicMock(),
    )


# --- Factory ---

def test_openai_runner_factory_create_agent_graph_returns_runner():
    graph = _make_graph()
    tools: ToolRegistry = {'search': lambda q: q}
    factory = OpenAIRunnerFactory(client=MagicMock())
    runner = factory.create_agent_graph(graph, tools)
    assert isinstance(runner, OpenAIAgentGraphRunner)


def test_openai_runner_factory_create_agent_graph_wires_graph_and_tools():
    graph = _make_graph()
    tools: ToolRegistry = {'my_tool': lambda: None}
    factory = OpenAIRunnerFactory(client=MagicMock())
    runner = factory.create_agent_graph(graph, tools)
    assert runner._graph is graph
    assert runner._tools is tools


# --- OpenAIAgentGraphRunner ---

def test_openai_agent_graph_runner_stores_graph_and_tools():
    graph = _make_graph()
    tools: ToolRegistry = {}
    runner = OpenAIAgentGraphRunner(graph, tools)
    assert runner._graph is graph
    assert runner._tools is tools


@pytest.mark.asyncio
async def test_openai_agent_graph_runner_run_raises_when_agents_not_installed():
    graph = _make_graph()
    runner = OpenAIAgentGraphRunner(graph, {})

    with patch.dict('sys.modules', {'agents': None}):
        # The import inside run() will fail — runner should return failure result
        # rather than propagate the ImportError, since it's caught by the except block
        result = await runner.run("test input")
        assert isinstance(result, AgentGraphResult)
        assert result.metrics.success is False


@pytest.mark.asyncio
async def test_openai_agent_graph_runner_run_tracks_invocation_failure_on_exception():
    graph = _make_graph()
    tracker = graph.get_tracker()
    runner = OpenAIAgentGraphRunner(graph, {})

    with patch.dict('sys.modules', {'agents': None}):
        result = await runner.run("fail")

    assert result.metrics.success is False
    tracker.track_invocation_failure.assert_called_once()
    tracker.track_latency.assert_called_once()


@pytest.mark.asyncio
async def test_openai_agent_graph_runner_run_success():
    graph = _make_graph()
    tracker = graph.get_tracker()

    mock_result = MagicMock()
    mock_result.final_output = "agent answer"
    mock_result.context_wrapper.usage.total_tokens = 0
    mock_result.context_wrapper.usage.input_tokens = 0
    mock_result.context_wrapper.usage.output_tokens = 0

    mock_runner_module = MagicMock()
    mock_runner_module.run = AsyncMock(return_value=mock_result)

    mock_agents = MagicMock()
    mock_agents.Runner = mock_runner_module
    mock_agents.Agent = MagicMock(return_value=MagicMock())
    mock_agents.FunctionTool = MagicMock()
    mock_agents.Handoff = MagicMock()
    mock_agents.RunContextWrapper = MagicMock()
    mock_agents.Tool = MagicMock()
    mock_agents.handoff = MagicMock()

    mock_agents_ext = MagicMock()
    mock_agents_ext.RECOMMENDED_PROMPT_PREFIX = '[PREFIX]'

    mock_tool_context = MagicMock()

    with patch.dict('sys.modules', {
        'agents': mock_agents,
        'agents.extensions': MagicMock(),
        'agents.extensions.handoff_prompt': mock_agents_ext,
        'agents.tool_context': mock_tool_context,
    }):
        runner = OpenAIAgentGraphRunner(graph, {})
        result = await runner.run("find restaurants")

    assert isinstance(result, AgentGraphResult)
    assert result.output == "agent answer"
    assert result.metrics.success is True
    tracker.track_invocation_success.assert_called_once()
    tracker.track_path.assert_called_once()
    tracker.track_latency.assert_called_once()

    root_tracker = graph.get_node('root-agent').get_config().tracker
    root_tracker.track_duration.assert_called_once()
    root_tracker.track_tokens.assert_called_once()
    root_tracker.track_success.assert_called_once()


class _StubRunHooks:
    """Minimal base class that stands in for agents.RunHooks in tests."""


def _make_test_hooks(graph, name_map):
    return _make_span_hooks(_StubRunHooks, graph, name_map)


@pytest.mark.asyncio
async def test_ldai_agent_span_hooks_on_agent_start_creates_span():
    """Verify on_agent_start creates a span and annotates it with ai config metadata."""
    graph = _make_graph()
    hooks = _make_test_hooks(graph, {'root_agent': 'root-agent'})

    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span
    mock_trace = MagicMock()
    mock_trace.get_tracer.return_value = mock_tracer

    mock_agent = MagicMock()
    mock_agent.name = 'root_agent'

    import ldai.observe as obs
    with patch.object(obs, '_OTEL_AVAILABLE', True), \
         patch.object(obs, '_otel_trace', mock_trace), \
         patch.object(obs, '_otel_context', MagicMock(get_current=MagicMock(return_value={}))), \
         patch('ldai.observe.annotate_span_with_ai_config_metadata') as mock_annotate:
        await hooks.on_agent_start(MagicMock(), mock_agent)

    mock_tracer.start_as_current_span.assert_called_once_with("ld.ai.agent", context={}, end_on_exit=False)
    mock_annotate.assert_called_once()


@pytest.mark.asyncio
async def test_ldai_agent_span_hooks_on_agent_start_closes_previous_span():
    """Verify on_agent_start ends the previous agent's open span on handoff."""
    graph = _make_graph()
    hooks = _make_test_hooks(graph, {'root_agent': 'root-agent'})

    first_span = MagicMock()
    first_span.is_recording.return_value = True
    first_span.__enter__ = MagicMock(return_value=first_span)
    first_span.__exit__ = MagicMock(return_value=False)
    second_span = MagicMock()
    second_span.is_recording.return_value = True
    second_span.__enter__ = MagicMock(return_value=second_span)
    second_span.__exit__ = MagicMock(return_value=False)
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.side_effect = [first_span, second_span]

    agent_a = MagicMock()
    agent_a.name = 'root_agent'
    agent_b = MagicMock()
    agent_b.name = 'root_agent'  # same node, second call simulates a handoff scenario

    import ldai.observe as obs
    with patch.object(obs, '_OTEL_AVAILABLE', True), \
         patch.object(obs, '_otel_trace', MagicMock(get_tracer=MagicMock(return_value=mock_tracer))), \
         patch.object(obs, '_otel_context', MagicMock(get_current=MagicMock(return_value={}))):
        await hooks.on_agent_start(MagicMock(), agent_a)
        # Second on_agent_start should end first_span before opening second_span
        await hooks.on_agent_start(MagicMock(), agent_b)

    first_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_ldai_agent_span_hooks_on_agent_end_ends_span():
    """Verify on_agent_end calls span.end() for a span opened by on_agent_start."""
    graph = _make_graph()
    hooks = _make_test_hooks(graph, {'root_agent': 'root-agent'})

    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    mock_agent = MagicMock()
    mock_agent.name = 'root_agent'

    import ldai.observe as obs
    with patch.object(obs, '_OTEL_AVAILABLE', True), \
         patch.object(obs, '_otel_trace', MagicMock(get_tracer=MagicMock(return_value=mock_tracer))), \
         patch.object(obs, '_otel_context', MagicMock(get_current=MagicMock(return_value={}))), \
         patch('ldai.observe.annotate_span_with_ai_config_metadata'):
        await hooks.on_agent_start(MagicMock(), mock_agent)
        await hooks.on_agent_end(MagicMock(), mock_agent, output="result")

    mock_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_ldai_agent_span_hooks_on_agent_end_no_op_if_no_start():
    """Verify on_agent_end is a no-op when no matching on_agent_start."""
    graph = _make_graph()
    hooks = _make_test_hooks(graph, {})

    mock_agent = MagicMock()
    mock_agent.name = 'unknown_agent'
    await hooks.on_agent_end(MagicMock(), mock_agent, output="result")


@pytest.mark.asyncio
async def test_openai_agent_graph_runner_run_uses_hooks():
    """Verify that run() passes a _LDAIAgentSpanHooks instance to Runner.run()."""
    graph = _make_graph()

    mock_result = MagicMock()
    mock_result.final_output = "agent answer"
    mock_result.context_wrapper.usage.total_tokens = 0
    mock_result.context_wrapper.usage.input_tokens = 0
    mock_result.context_wrapper.usage.output_tokens = 0

    mock_runner_module = MagicMock()
    mock_runner_module.run = AsyncMock(return_value=mock_result)

    mock_agents = MagicMock()
    mock_agents.Runner = mock_runner_module
    mock_agents.Agent = MagicMock(return_value=MagicMock())
    mock_agents.FunctionTool = MagicMock()
    mock_agents.Handoff = MagicMock()
    mock_agents.RunContextWrapper = MagicMock()
    mock_agents.Tool = MagicMock()
    mock_agents.handoff = MagicMock()

    mock_agents_ext = MagicMock()
    mock_agents_ext.RECOMMENDED_PROMPT_PREFIX = '[PREFIX]'

    mock_tool_context = MagicMock()

    with patch.dict('sys.modules', {
        'agents': mock_agents,
        'agents.extensions': MagicMock(),
        'agents.extensions.handoff_prompt': mock_agents_ext,
        'agents.tool_context': mock_tool_context,
    }):
        runner = OpenAIAgentGraphRunner(graph, {})
        result = await runner.run("test observability")

    assert result.metrics.success is True
    # Verify Runner.run was called with a hooks kwarg that has on_agent_start/end
    call_kwargs = mock_runner_module.run.call_args.kwargs
    assert 'hooks' in call_kwargs
    hooks = call_kwargs['hooks']
    assert callable(getattr(hooks, 'on_agent_start', None))
    assert callable(getattr(hooks, 'on_agent_end', None))


@pytest.mark.asyncio
async def test_openai_agent_graph_runner_run_logs_invoke(caplog):
    """Verify that run() emits expected info-level log messages for observability."""
    graph = _make_graph()

    mock_result = MagicMock()
    mock_result.final_output = "agent answer"
    mock_result.context_wrapper.usage.total_tokens = 0
    mock_result.context_wrapper.usage.input_tokens = 0
    mock_result.context_wrapper.usage.output_tokens = 0

    mock_runner_module = MagicMock()
    mock_runner_module.run = AsyncMock(return_value=mock_result)

    mock_agents = MagicMock()
    mock_agents.Runner = mock_runner_module
    mock_agents.Agent = MagicMock(return_value=MagicMock())
    mock_agents.FunctionTool = MagicMock()
    mock_agents.Handoff = MagicMock()
    mock_agents.RunContextWrapper = MagicMock()
    mock_agents.Tool = MagicMock()
    mock_agents.handoff = MagicMock()

    mock_agents_ext = MagicMock()
    mock_agents_ext.RECOMMENDED_PROMPT_PREFIX = '[PREFIX]'

    mock_tool_context = MagicMock()

    with patch.dict('sys.modules', {
        'agents': mock_agents,
        'agents.extensions': MagicMock(),
        'agents.extensions.handoff_prompt': mock_agents_ext,
        'agents.tool_context': mock_tool_context,
    }), caplog.at_level(logging.INFO, logger="ldai_openai.openai_agent_graph_runner"):
        runner = OpenAIAgentGraphRunner(graph, {})
        result = await runner.run("test logging")

    assert result.metrics.success is True
    log_messages = [r.message for r in caplog.records]
    assert any("invoke called" in msg for msg in log_messages), f"Expected 'invoke called' in logs: {log_messages}"
