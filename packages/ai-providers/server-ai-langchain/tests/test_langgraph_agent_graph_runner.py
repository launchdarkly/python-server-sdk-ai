"""Tests for LangGraphAgentGraphRunner and LangChainRunnerFactory.create_agent_graph()."""

from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ldai.agent_graph import AgentGraphDefinition
from ldai.evaluator import Evaluator
from ldai.models import AIAgentConfig, AIAgentGraphConfig, ModelConfig, ProviderConfig
from ldai.providers import ToolRegistry
from ldai.providers.types import AgentGraphRunnerResult

from ldai_langchain.langchain_runner_factory import LangChainRunnerFactory
from ldai_langchain.langgraph_agent_graph_runner import LangGraphAgentGraphRunner


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
        evaluator=Evaluator.noop(),
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
        assert isinstance(result, AgentGraphRunnerResult)
        assert result.metrics.success is False


@pytest.mark.asyncio
async def test_langgraph_runner_run_returns_failure_on_exception():
    """Runner now returns AgentGraphRunnerResult; managed layer drives tracker events."""
    graph = _make_graph()
    runner = LangGraphAgentGraphRunner(graph, {})

    with patch.dict('sys.modules', {'langgraph': None, 'langgraph.graph': None}):
        result = await runner.run("fail")

    assert isinstance(result, AgentGraphRunnerResult)
    assert result.metrics.success is False
    assert result.metrics.duration_ms is not None


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

    assert isinstance(result, AgentGraphRunnerResult)
    assert result.metrics.duration_ms is not None
    # Tracker events now fire from the managed layer (ManagedAgentGraph) using
    # result.metrics; the runner no longer touches the graph tracker directly.
    tracker.track_path.assert_not_called()
    tracker.track_invocation_success.assert_not_called()
    tracker.track_duration.assert_not_called()


@pytest.mark.asyncio
async def test_langgraph_runner_run_resets_node_metrics_between_runs():
    """Successive runs do not leak stale node metrics from a previous run.

    Mirrors ``test_openai_agent_graph_runner_run_resets_node_metrics_between_runs``
    in the OpenAI provider tests.  Each ``run()`` invocation must produce its
    own fresh ``node_metrics`` rather than a union of all prior runs' metrics.

    Strategy: bypass ``_build_graph()`` by pre-populating ``_compiled`` and
    ``_node_keys`` on the runner.  The mock compiled graph's ``ainvoke`` is a
    side-effect coroutine that fires callbacks on the handler passed in via
    ``config['callbacks']`` — the same handler the real LangGraph executor
    would invoke.  Each call fires events for only ``root-agent`` so we can
    assert the second result's ``node_metrics`` reflects only the second run.
    """
    graph = _make_graph()

    mock_message = MagicMock()
    mock_message.content = "answer"
    mock_message.usage_metadata = None
    mock_message.response_metadata = None

    async def fire_callbacks(_payload, *, config):
        handler = config['callbacks'][0]
        # If state leaked across runs, the handler passed in here on the
        # second call would already contain entries from the first run before
        # any callback fires.  We assert below that this is not the case.
        run_id = uuid4()
        handler.on_chain_start({}, {}, run_id=run_id, name='root-agent')
        handler.on_chain_end({}, run_id=run_id)
        return {'messages': [mock_message]}

    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(side_effect=fire_callbacks)

    mock_human_message = MagicMock()
    mock_lc_core_messages = MagicMock()
    mock_lc_core_messages.HumanMessage = MagicMock(return_value=mock_human_message)

    runner = LangGraphAgentGraphRunner(graph, {})
    # Bypass _build_graph(): provide a pre-compiled graph and the node keys
    # that the callback handler would otherwise be initialised with.
    runner._compiled = mock_compiled
    runner._node_keys = {'root-agent'}
    runner._fn_name_to_config_key = {}

    with patch.dict('sys.modules', {
        'langchain_core': MagicMock(),
        'langchain_core.messages': mock_lc_core_messages,
    }):
        first = await runner.run("attempt 1")
        assert first.metrics.success is True
        assert 'root-agent' in first.metrics.node_metrics
        first_metrics = first.metrics.node_metrics['root-agent']

        second = await runner.run("attempt 2")

    assert second.metrics.success is True
    assert 'root-agent' in second.metrics.node_metrics
    # The second run's per-node metrics must be a fresh object, not the
    # accumulated state from the first run.  If the runner leaked the
    # callback handler (or its state dict) across invocations, the second
    # run would return the same LDAIMetrics instance with cumulative values.
    assert second.metrics.node_metrics['root-agent'] is not first_metrics
    # Path and node_metrics keys reflect only the second invocation.
    assert second.metrics.path == ['root-agent']
    assert set(second.metrics.node_metrics.keys()) == {'root-agent'}
