"""Tests for LangGraphAgentGraphRunner and LangChainRunnerFactory.create_agent_graph()."""

from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ldai.agent_graph import AgentGraphDefinition
from ldai.evaluator import Evaluator
from ldai.models import (
    AIAgentConfig,
    AIAgentGraphConfig,
    JudgeConfiguration,
    ModelConfig,
    ProviderConfig,
)
from ldai.providers import ToolRegistry
from ldai.providers.types import AgentGraphRunnerResult, EvalRequest

from ldai_langchain.langchain_runner_factory import LangChainRunnerFactory
from ldai_langchain.langgraph_agent_graph_runner import (
    LangGraphAgentGraphRunner,
    _maybe_record_eval_request,
)


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

    Strategy: stub ``_build_graph`` to return a mock compiled graph whose
    ``ainvoke`` fires callbacks on the handler the runner passes via
    ``config['callbacks']`` — the same handler the real LangGraph executor
    would invoke. Each call fires events for only ``root-agent`` so we can
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
    # Stub _build_graph(): return a pre-compiled mock plus the node keys
    # that the callback handler would otherwise be initialised with.
    runner._build_graph = MagicMock(  # type: ignore[method-assign]
        return_value=(mock_compiled, {}, {'root-agent'}),
    )

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


# --- _maybe_record_eval_request unit tests ---

def _msg(content):
    m = MagicMock()
    m.content = content
    return m


def _response(content, tool_calls=None):
    r = MagicMock()
    r.content = content
    r.tool_calls = tool_calls
    return r


def test_maybe_record_eval_request_emits_for_plain_response():
    out = []
    _maybe_record_eval_request(
        out,
        node_key='root',
        msgs=[_msg('user prompt')],
        response=_response('final answer'),
        handoff_tool_names=frozenset(),
    )
    assert len(out) == 1
    req = out[0]
    assert isinstance(req, EvalRequest)
    assert req.node_key == 'root'
    assert req.input == 'user prompt'
    assert req.output == 'final answer'


def test_maybe_record_eval_request_skips_when_response_has_functional_tool_call():
    out = []
    _maybe_record_eval_request(
        out,
        node_key='root',
        msgs=[_msg('user prompt')],
        response=_response('', tool_calls=[{'name': 'search', 'args': {}}]),
        handoff_tool_names=frozenset(['transfer_to_x']),
    )
    assert out == []


def test_maybe_record_eval_request_emits_when_only_handoff_tool_calls():
    out = []
    _maybe_record_eval_request(
        out,
        node_key='root',
        msgs=[_msg('user prompt')],
        response=_response(
            'handing off now',
            tool_calls=[{'name': 'transfer_to_specialist', 'args': {}}],
        ),
        handoff_tool_names=frozenset(['transfer_to_specialist']),
    )
    assert len(out) == 1
    assert out[0].output == 'handing off now'


def test_maybe_record_eval_request_skips_when_output_is_blank():
    out = []
    _maybe_record_eval_request(
        out,
        node_key='root',
        msgs=[_msg('user prompt')],
        response=_response('   '),
        handoff_tool_names=frozenset(),
    )
    assert out == []


def test_maybe_record_eval_request_joins_msgs_with_crlf():
    out = []
    _maybe_record_eval_request(
        out,
        node_key='root',
        msgs=[_msg('system'), _msg('user')],
        response=_response('answer'),
        handoff_tool_names=frozenset(),
    )
    assert out[0].input == 'system\r\nuser'


# --- Runner-level eval_requests behavior ---


def _make_graph_with_judge(node_keys_with_judges=None) -> AgentGraphDefinition:
    """Build a 2-node graph (root -> specialist) with judges optionally configured."""
    if node_keys_with_judges is None:
        node_keys_with_judges = {'root-agent'}
    graph_tracker = MagicMock()

    def _agent_config(key: str) -> AIAgentConfig:
        jc = (
            JudgeConfiguration(judges=[JudgeConfiguration.Judge(key='j1', sampling_rate=1.0)])
            if key in node_keys_with_judges
            else None
        )
        return AIAgentConfig(
            key=key,
            enabled=True,
            create_tracker=MagicMock(return_value=MagicMock()),
            model=ModelConfig(name='gpt-4'),
            provider=ProviderConfig(name='openai'),
            instructions=f'You are {key}.',
            evaluator=Evaluator.noop(),
            judge_configuration=jc,
        )

    from ldai.models import Edge
    graph_config = AIAgentGraphConfig(
        key='judge-graph',
        root_config_key='root-agent',
        edges=[Edge(key='e1', source_config='root-agent', target_config='specialist-agent')],
        enabled=True,
    )
    configs = {
        'root-agent': _agent_config('root-agent'),
        'specialist-agent': _agent_config('specialist-agent'),
    }
    nodes = AgentGraphDefinition.build_nodes(graph_config, configs)
    return AgentGraphDefinition(
        agent_graph=graph_config,
        nodes=nodes,
        context=MagicMock(),
        enabled=True,
        create_tracker=lambda: graph_tracker,
    )


@pytest.mark.asyncio
async def test_runner_eval_requests_absent_when_no_judges_configured():
    """A graph whose nodes have no judge_configuration must produce no EvalRequests."""
    graph = _make_graph()  # nodes use Evaluator.noop() and no judge_configuration

    mock_message = MagicMock()
    mock_message.content = "answer"
    mock_message.usage_metadata = None
    mock_message.response_metadata = None

    async def fire(_payload, *, config):
        return {'messages': [mock_message]}

    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(side_effect=fire)

    mock_human_message = MagicMock()
    mock_lc_core_messages = MagicMock()
    mock_lc_core_messages.HumanMessage = MagicMock(return_value=mock_human_message)

    runner = LangGraphAgentGraphRunner(graph, {})
    runner._build_graph = MagicMock(  # type: ignore[method-assign]
        return_value=(mock_compiled, {}, {'root-agent'}),
    )

    with patch.dict('sys.modules', {
        'langchain_core': MagicMock(),
        'langchain_core.messages': mock_lc_core_messages,
    }):
        result = await runner.run("hello")

    assert result.eval_requests is None or result.eval_requests == []


@pytest.mark.asyncio
async def test_runner_eval_requests_populated_for_node_with_judges():
    """When a node has judges configured, its activation emits an EvalRequest with input/output captured."""
    graph = _make_graph_with_judge(node_keys_with_judges={'root-agent'})

    captured_eval_requests = []

    def fake_build_graph(eval_requests_list):
        # Capture the runner-provided list and emulate the closure appending to it.
        captured_eval_requests.append(eval_requests_list)
        mock_compiled = MagicMock()

        async def fire(_payload, *, config):
            # Pretend the LangGraph node closure invoked _maybe_record_eval_request.
            eval_requests_list.append(
                EvalRequest(node_key='root-agent', input='hello', output='final answer')
            )
            mock_message = MagicMock()
            mock_message.content = "final answer"
            mock_message.usage_metadata = None
            mock_message.response_metadata = None
            return {'messages': [mock_message]}

        mock_compiled.ainvoke = AsyncMock(side_effect=fire)
        return mock_compiled, {}, {'root-agent'}

    mock_human_message = MagicMock()
    mock_lc_core_messages = MagicMock()
    mock_lc_core_messages.HumanMessage = MagicMock(return_value=mock_human_message)

    runner = LangGraphAgentGraphRunner(graph, {})
    runner._build_graph = fake_build_graph  # type: ignore[method-assign]

    with patch.dict('sys.modules', {
        'langchain_core': MagicMock(),
        'langchain_core.messages': mock_lc_core_messages,
    }):
        result = await runner.run("hello")

    assert result.eval_requests is not None
    assert len(result.eval_requests) == 1
    assert result.eval_requests[0].node_key == 'root-agent'
    assert result.eval_requests[0].input == 'hello'
    assert result.eval_requests[0].output == 'final answer'


@pytest.mark.asyncio
async def test_runner_eval_requests_isolated_between_runs():
    """Concurrent / successive runs must each receive a fresh eval_requests list."""
    graph = _make_graph_with_judge(node_keys_with_judges={'root-agent'})

    seen_lists = []

    def fake_build_graph(eval_requests_list):
        seen_lists.append(id(eval_requests_list))
        mock_compiled = MagicMock()

        async def fire(_payload, *, config):
            eval_requests_list.append(
                EvalRequest(node_key='root-agent', input='x', output='y')
            )
            mock_message = MagicMock()
            mock_message.content = "y"
            mock_message.usage_metadata = None
            mock_message.response_metadata = None
            return {'messages': [mock_message]}

        mock_compiled.ainvoke = AsyncMock(side_effect=fire)
        return mock_compiled, {}, {'root-agent'}

    mock_human_message = MagicMock()
    mock_lc_core_messages = MagicMock()
    mock_lc_core_messages.HumanMessage = MagicMock(return_value=mock_human_message)

    runner = LangGraphAgentGraphRunner(graph, {})
    runner._build_graph = fake_build_graph  # type: ignore[method-assign]

    with patch.dict('sys.modules', {
        'langchain_core': MagicMock(),
        'langchain_core.messages': mock_lc_core_messages,
    }):
        first = await runner.run("a")
        second = await runner.run("b")

    # Each call gets a fresh list object.
    assert len(seen_lists) == 2
    assert seen_lists[0] != seen_lists[1]
    assert len(first.eval_requests) == 1
    assert len(second.eval_requests) == 1
