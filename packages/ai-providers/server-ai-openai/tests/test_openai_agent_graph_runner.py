"""Tests for OpenAIAgentGraphRunner and OpenAIRunnerFactory.create_agent_graph()."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ldai.agent_graph import AgentGraphDefinition
from ldai.models import AIAgentGraphConfig, AIAgentConfig, Edge, ModelConfig, ProviderConfig
from ldai.providers import AgentGraphResult, ToolRegistry
from ldai_openai.openai_agent_graph_runner import OpenAIAgentGraphRunner
from ldai_openai.openai_runner_factory import OpenAIRunnerFactory


def _make_graph(enabled: bool = True) -> AgentGraphDefinition:
    """Build a minimal single-node AgentGraphDefinition for testing."""
    node_tracker = MagicMock()
    graph_tracker = MagicMock()
    root_config = AIAgentConfig(
        key='root-agent',
        enabled=enabled,
        model=ModelConfig(name='gpt-4'),
        provider=ProviderConfig(name='openai'),
        instructions='You are a helpful assistant.',
        create_tracker=lambda: node_tracker,
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
        create_tracker=lambda: graph_tracker,
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
    tracker = graph.create_tracker()
    runner = OpenAIAgentGraphRunner(graph, {})

    with patch.dict('sys.modules', {'agents': None}):
        result = await runner.run("fail")

    assert result.metrics.success is False
    tracker.track_invocation_failure.assert_called_once()
    tracker.track_latency.assert_called_once()


@pytest.mark.asyncio
async def test_openai_agent_graph_runner_run_success():
    graph = _make_graph()
    tracker = graph.create_tracker()

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

    root_tracker = graph.get_node('root-agent').get_config().create_tracker()
    root_tracker.track_duration.assert_called_once()
    root_tracker.track_tokens.assert_called_once()
    root_tracker.track_success.assert_called_once()
