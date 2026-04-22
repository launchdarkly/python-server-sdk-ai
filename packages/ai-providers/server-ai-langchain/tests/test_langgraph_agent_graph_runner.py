"""Tests for LangGraphAgentGraphRunner and LangChainRunnerFactory.create_agent_graph()."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ldai.agent_graph import AgentGraphDefinition
from ldai.models import AIAgentGraphConfig, AIAgentConfig, ModelConfig, ProviderConfig
from ldai.providers import AgentGraphResult, ToolRegistry
from ldai_langchain.langgraph_agent_graph_runner import LangGraphAgentGraphRunner
from ldai_langchain.langchain_runner_factory import LangChainRunnerFactory


def _make_graph(enabled: bool = True) -> AgentGraphDefinition:
    graph_tracker = MagicMock()
    node_tracker = MagicMock()
    root_config = AIAgentConfig(
        key='root-agent',
        enabled=enabled,
        create_tracker=MagicMock(return_value=node_tracker),
        model=ModelConfig(name='gpt-4'),
        provider=ProviderConfig(name='openai'),
        instructions='You are a helpful assistant.',
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

def test_langchain_runner_factory_create_agent_graph_returns_runner():
    graph = _make_graph()
    tools: ToolRegistry = {'fetch_weather': lambda loc: f'weather in {loc}'}
    factory = LangChainRunnerFactory()
    runner = factory.create_agent_graph(graph, tools)
    assert isinstance(runner, LangGraphAgentGraphRunner)


def test_langchain_runner_factory_create_agent_graph_wires_graph_and_tools():
    graph = _make_graph()
    tools: ToolRegistry = {}
    factory = LangChainRunnerFactory()
    runner = factory.create_agent_graph(graph, tools)
    assert runner._graph is graph
    assert runner._tools is tools


# --- LangGraphAgentGraphRunner ---

def test_langgraph_runner_stores_graph_and_tools():
    graph = _make_graph()
    tools: ToolRegistry = {}
    runner = LangGraphAgentGraphRunner(graph, tools)
    assert runner._graph is graph
    assert runner._tools is tools


@pytest.mark.asyncio
async def test_langgraph_runner_run_raises_when_langgraph_not_installed():
    graph = _make_graph()
    runner = LangGraphAgentGraphRunner(graph, {})

    with patch.dict('sys.modules', {'langgraph': None, 'langgraph.graph': None}):
        result = await runner.run("test")
        assert isinstance(result, AgentGraphResult)
        assert result.metrics.success is False


@pytest.mark.asyncio
async def test_langgraph_runner_run_tracks_failure_on_exception():
    graph = _make_graph()
    tracker = graph.create_tracker()
    runner = LangGraphAgentGraphRunner(graph, {})

    with patch.dict('sys.modules', {'langgraph': None, 'langgraph.graph': None}):
        result = await runner.run("fail")

    assert result.metrics.success is False
    tracker.track_invocation_failure.assert_called_once()
    tracker.track_duration.assert_called_once()


@pytest.mark.asyncio
async def test_langgraph_runner_run_success():
    graph = _make_graph()
    tracker = graph.create_tracker()

    mock_message = MagicMock()
    mock_message.content = "langgraph answer"
    mock_message.usage_metadata = None
    mock_message.response_metadata = None

    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value={'messages': [mock_message]})

    mock_state_graph_instance = MagicMock()
    mock_state_graph_instance.add_node = MagicMock()
    mock_state_graph_instance.add_edge = MagicMock()
    mock_state_graph_instance.compile = MagicMock(return_value=mock_compiled)

    mock_langgraph_graph = MagicMock()
    mock_langgraph_graph.END = 'END'
    mock_langgraph_graph.START = 'START'
    mock_langgraph_graph.StateGraph = MagicMock(return_value=mock_state_graph_instance)

    mock_human_message = MagicMock()
    mock_lc_core_messages = MagicMock()
    mock_lc_core_messages.HumanMessage = MagicMock(return_value=mock_human_message)
    mock_lc_core_messages.AnyMessage = MagicMock()

    mock_model_response = MagicMock()
    mock_model_response.content = 'langgraph answer'
    mock_model_response.usage_metadata = None
    mock_model_response.response_metadata = None
    mock_model_response.tool_calls = None

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_model_response)

    mock_init_model = MagicMock()
    mock_init_model.return_value = mock_llm
    mock_langchain_chat = MagicMock()
    mock_langchain_chat.init_chat_model = mock_init_model

    with patch.dict('sys.modules', {
        'langgraph': MagicMock(),
        'langgraph.graph': mock_langgraph_graph,
        'langchain_core': MagicMock(),
        'langchain_core.messages': mock_lc_core_messages,
        'langchain': MagicMock(),
        'langchain.chat_models': mock_langchain_chat,
        'typing_extensions': __import__('typing_extensions'),
    }):
        runner = LangGraphAgentGraphRunner(graph, {})
        result = await runner.run("find restaurants")

    assert isinstance(result, AgentGraphResult)
    assert result.output == "langgraph answer"
    assert result.metrics.success is True
    tracker.track_path.assert_called_once_with([])
    tracker.track_invocation_success.assert_called_once()
    tracker.track_duration.assert_called_once()
