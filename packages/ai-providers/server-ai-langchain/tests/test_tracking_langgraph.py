"""
Integration tests for LangGraphAgentGraphRunner tracking pipeline.

Uses real AIGraphTracker and LDAIConfigTracker backed by a mock LD client,
and a fake LangChain model to verify that the correct LD events are emitted
with the correct payloads — without making real API calls.
"""

import pytest
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

from ldai.agent_graph import AgentGraphDefinition
from ldai.models import AIAgentGraphConfig, AIAgentConfig, Edge, ModelConfig, ProviderConfig
from ldai.tracker import AIGraphTracker, LDAIConfigTracker
from ldai_langchain.langgraph_agent_graph_runner import LangGraphAgentGraphRunner

pytestmark = pytest.mark.skipif(
    pytest.importorskip('langgraph', reason='langgraph not installed') is None,
    reason='langgraph not installed',
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(
    mock_ld_client: MagicMock,
    node_key: str = 'root-agent',
    graph_key: str = 'test-graph',
    tool_names: list = None,
) -> AgentGraphDefinition:
    """
    Build an AgentGraphDefinition backed by real tracker objects that record
    events to a mock LD client.
    """
    context = MagicMock()

    node_tracker = LDAIConfigTracker(
        ld_client=mock_ld_client,
        variation_key='test-variation',
        config_key=node_key,
        version=1,
        model_name='gpt-4',
        provider_name='openai',
        context=context,
        run_id='test-run-id',
        graph_key=graph_key,
    )

    graph_tracker = AIGraphTracker(
        ld_client=mock_ld_client,
        variation_key='test-variation',
        graph_key=graph_key,
        version=1,
        context=context,
    )

    tool_defs = (
        [{'name': name, 'type': 'function', 'description': '', 'parameters': {}}
         for name in tool_names]
        if tool_names else None
    )

    root_config = AIAgentConfig(
        key=node_key,
        enabled=True,
        model=ModelConfig(name='gpt-4', parameters={'tools': tool_defs} if tool_defs else {}),
        provider=ProviderConfig(name='openai'),
        instructions='You are a helpful assistant.',
        create_tracker=lambda: node_tracker,
    )

    graph_config = AIAgentGraphConfig(
        key=graph_key,
        root_config_key=node_key,
        edges=[],
        enabled=True,
    )

    nodes = AgentGraphDefinition.build_nodes(graph_config, {node_key: root_config})
    return AgentGraphDefinition(
        agent_graph=graph_config,
        nodes=nodes,
        context=context,
        enabled=True,
        create_tracker=lambda: graph_tracker,
    )


def _make_fake_response(
    content: str,
    input_tokens: int = 10,
    output_tokens: int = 5,
    tool_call_names: list = None,
):
    """Create a real AIMessage with usage metadata and optional tool calls."""
    from langchain_core.messages import AIMessage

    tool_calls = [
        {'name': name, 'args': {}, 'id': f'call_{i}', 'type': 'tool_call'}
        for i, name in enumerate(tool_call_names or [])
    ]

    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        usage_metadata={
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
        },
    )


def _events(mock_ld_client: MagicMock) -> dict:
    """Return dict of event_name -> list of (data, value) from all track() calls."""
    result = defaultdict(list)
    for call in mock_ld_client.track.call_args_list:
        name, _ctx, data, value = call.args
        result[name].append((data, value))
    return dict(result)


def _mock_model(response):
    """Return a mock LangChain model that always returns response on ainvoke()."""
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=response)
    model.bind_tools.return_value = model
    return model


def _make_two_node_graph(mock_ld_client: MagicMock) -> 'AgentGraphDefinition':
    """Build a two-node AgentGraphDefinition (root-agent → child-agent)."""
    context = MagicMock()

    root_tracker = LDAIConfigTracker(
        ld_client=mock_ld_client,
        variation_key='test-variation',
        config_key='root-agent',
        version=1,
        model_name='gpt-4',
        provider_name='openai',
        context=context,
        run_id='test-run-id',
        graph_key='two-node-graph',
    )
    child_tracker = LDAIConfigTracker(
        ld_client=mock_ld_client,
        variation_key='test-variation',
        config_key='child-agent',
        version=1,
        model_name='gpt-4',
        provider_name='openai',
        context=context,
        run_id='test-run-id',
        graph_key='two-node-graph',
    )
    graph_tracker = AIGraphTracker(
        ld_client=mock_ld_client,
        variation_key='test-variation',
        graph_key='two-node-graph',
        version=1,
        context=context,
    )

    root_config = AIAgentConfig(
        key='root-agent',
        enabled=True,
        model=ModelConfig(name='gpt-4', parameters={}),
        provider=ProviderConfig(name='openai'),
        instructions='You are root.',
        create_tracker=lambda: root_tracker,
    )
    child_config = AIAgentConfig(
        key='child-agent',
        enabled=True,
        model=ModelConfig(name='gpt-4', parameters={}),
        provider=ProviderConfig(name='openai'),
        instructions='You are child.',
        create_tracker=lambda: child_tracker,
    )

    edge = Edge(key='root-to-child', source_config='root-agent', target_config='child-agent')
    graph_config = AIAgentGraphConfig(
        key='two-node-graph',
        root_config_key='root-agent',
        edges=[edge],
        enabled=True,
    )

    nodes = AgentGraphDefinition.build_nodes(graph_config, {
        'root-agent': root_config,
        'child-agent': child_config,
    })
    return AgentGraphDefinition(
        agent_graph=graph_config,
        nodes=nodes,
        context=context,
        enabled=True,
        create_tracker=lambda: graph_tracker,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tracks_node_and_graph_tokens_on_success():
    """Node-level and graph-level token events fire with the correct counts."""
    from uuid import uuid4
    from langchain_core.messages import AIMessage as _AIMsg
    from langchain_core.outputs import LLMResult, ChatGeneration
    from ldai_langchain.langgraph_callback_handler import LDMetricsCallbackHandler

    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client)
    fake_response = _make_fake_response('Sunny.', input_tokens=10, output_tokens=5)

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               return_value=_mock_model(fake_response)):
        runner = LangGraphAgentGraphRunner(graph, {})
        result = await runner.run("What's the weather?")

    assert result.metrics.success is True
    assert result.output == 'Sunny.'

    # Manually simulate what the callback handler would collect and flush
    # (mock models don't fire LangChain callbacks, so we test flush directly)
    mock_ld_client2 = MagicMock()
    graph2 = _make_graph(mock_ld_client2)
    tracker2 = graph2.create_tracker()

    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')

    llm_result = LLMResult(
        generations=[[ChatGeneration(
            message=_AIMsg(content='Sunny.', usage_metadata={'total_tokens': 15, 'input_tokens': 10, 'output_tokens': 5}),
            text='Sunny.',
        )]],
        llm_output={},
    )
    handler.on_llm_end(llm_result, run_id=uuid4(), parent_run_id=node_run_id)
    handler.on_chain_end({}, run_id=node_run_id)
    handler.flush(graph2)

    ev2 = _events(mock_ld_client2)
    assert ev2['$ld:ai:tokens:total'][0][1] == 15
    assert ev2['$ld:ai:tokens:input'][0][1] == 10
    assert ev2['$ld:ai:tokens:output'][0][1] == 5
    assert ev2['$ld:ai:generation:success'][0][1] == 1
    assert '$ld:ai:duration:total' in ev2

    # Graph-level events from the real run
    ev = _events(mock_ld_client)
    assert ev['$ld:ai:graph:total_tokens'][0][1] == 15
    assert ev['$ld:ai:graph:invocation_success'][0][1] == 1
    assert '$ld:ai:graph:latency' in ev
    assert '$ld:ai:graph:path' in ev


@pytest.mark.asyncio
async def test_tracks_execution_path():
    """The path event contains the executed node key."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, node_key='my-agent')
    fake_response = _make_fake_response('Done.')

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               return_value=_mock_model(fake_response)):
        runner = LangGraphAgentGraphRunner(graph, {})
        await runner.run('hello')

    ev = _events(mock_ld_client)
    path_data = ev['$ld:ai:graph:path'][0][0]
    assert 'my-agent' in path_data['path']


@pytest.mark.asyncio
async def test_tracks_tool_calls():
    """A tool_call event fires for each tool name found in the model response."""
    from uuid import uuid4
    from ldai_langchain.langgraph_callback_handler import LDMetricsCallbackHandler

    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, tool_names=['get_weather'])

    # Model returns a tool call on the first invoke, then a final answer.
    tool_response = _make_fake_response('Calling tool.', tool_call_names=['get_weather'])
    final_response = _make_fake_response('It is sunny in NYC.')

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(side_effect=[tool_response, final_response])
    mock_model.bind_tools.return_value = mock_model

    def get_weather(location: str = 'NYC') -> str:
        """Return the current weather for a location."""
        return 'sunny'

    tool_registry = {'get_weather': get_weather}

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               return_value=mock_model):
        runner = LangGraphAgentGraphRunner(graph, tool_registry)
        await runner.run('What is the weather?')

    # Simulate tool call tracking via the callback handler directly
    mock_ld_client2 = MagicMock()
    graph2 = _make_graph(mock_ld_client2, tool_names=['get_weather'])
    tracker2 = graph2.create_tracker()

    handler = LDMetricsCallbackHandler({'root-agent'}, {'get_weather': 'get_weather'})
    # Agent node must appear in path for flush() to emit its events
    agent_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=agent_run_id, name='root-agent')
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    handler.on_tool_end('sunny', run_id=uuid4(), parent_run_id=tools_run_id, name='get_weather')
    handler.flush(graph2)

    ev2 = _events(mock_ld_client2)
    tool_events = ev2.get('$ld:ai:tool_call', [])
    assert len(tool_events) == 1
    assert tool_events[0][0]['toolKey'] == 'get_weather'


@pytest.mark.asyncio
async def test_tracks_multiple_tool_calls():
    """One tool_call event fires per tool name in the response."""
    from uuid import uuid4
    from ldai_langchain.langgraph_callback_handler import LDMetricsCallbackHandler

    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, tool_names=['search', 'summarize'])

    # Both tools called in one response; second invoke returns a final answer.
    tool_response = _make_fake_response('Done.', tool_call_names=['search', 'summarize'])
    final_response = _make_fake_response('Here is the summary.')

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(side_effect=[tool_response, final_response])
    mock_model.bind_tools.return_value = mock_model

    def search(q: str = '') -> str:
        """Search for information."""
        return q

    def summarize(text: str = '') -> str:
        """Summarize the given text."""
        return text

    tool_registry = {'search': search, 'summarize': summarize}

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               return_value=mock_model):
        runner = LangGraphAgentGraphRunner(graph, tool_registry)
        await runner.run('Search and summarize.')

    # Simulate multiple tool calls via the callback handler directly
    mock_ld_client2 = MagicMock()
    graph2 = _make_graph(mock_ld_client2, tool_names=['search', 'summarize'])
    tracker2 = graph2.create_tracker()

    fn_map = {'search': 'search', 'summarize': 'summarize'}
    handler = LDMetricsCallbackHandler({'root-agent'}, fn_map)
    # Agent node must appear in path for flush() to emit its events
    agent_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=agent_run_id, name='root-agent')
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    handler.on_tool_end('result', run_id=uuid4(), parent_run_id=tools_run_id, name='search')
    handler.on_tool_end('summary', run_id=uuid4(), parent_run_id=tools_run_id, name='summarize')
    handler.flush(graph2)

    ev2 = _events(mock_ld_client2)
    tool_keys = [data['toolKey'] for data, _ in ev2.get('$ld:ai:tool_call', [])]
    assert sorted(tool_keys) == ['search', 'summarize']


@pytest.mark.asyncio
async def test_tracks_graph_key_on_node_events():
    """Node-level events include the graphKey so they can be correlated to the graph."""
    from uuid import uuid4
    from langchain_core.messages import AIMessage as _AIMsg
    from langchain_core.outputs import LLMResult, ChatGeneration
    from ldai_langchain.langgraph_callback_handler import LDMetricsCallbackHandler

    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, graph_key='my-graph')
    tracker = graph.create_tracker()

    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')

    llm_result = LLMResult(
        generations=[[ChatGeneration(
            message=_AIMsg(content='OK.', usage_metadata={'total_tokens': 8, 'input_tokens': 5, 'output_tokens': 3}),
            text='OK.',
        )]],
        llm_output={},
    )
    handler.on_llm_end(llm_result, run_id=uuid4(), parent_run_id=node_run_id)
    handler.flush(graph)

    ev = _events(mock_ld_client)
    token_data = ev['$ld:ai:tokens:total'][0][0]
    assert token_data.get('graphKey') == 'my-graph'


@pytest.mark.asyncio
async def test_tracks_failure_and_latency_on_model_error():
    """When the model raises, failure and latency events fire; success does not."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client)

    error_model = MagicMock()
    error_model.ainvoke = AsyncMock(side_effect=RuntimeError('model error'))
    error_model.bind_tools.return_value = error_model

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               return_value=error_model):
        runner = LangGraphAgentGraphRunner(graph, {})
        result = await runner.run('fail')

    assert result.metrics.success is False

    ev = _events(mock_ld_client)
    assert '$ld:ai:graph:invocation_failure' in ev
    assert '$ld:ai:graph:latency' in ev
    assert '$ld:ai:graph:invocation_success' not in ev


@pytest.mark.asyncio
async def test_multi_node_tracks_per_node_tokens_and_path():
    """Each node emits its own token events; path and graph total cover both nodes."""
    from uuid import uuid4
    from langchain_core.messages import AIMessage as _AIMsg
    from langchain_core.outputs import LLMResult, ChatGeneration
    from ldai_langchain.langgraph_callback_handler import LDMetricsCallbackHandler

    mock_ld_client = MagicMock()
    graph = _make_two_node_graph(mock_ld_client)

    root_response = _make_fake_response('Root done.', input_tokens=10, output_tokens=5)
    child_response = _make_fake_response('Child done.', input_tokens=3, output_tokens=2)

    def model_factory(node_config, **kwargs):
        if node_config.key == 'root-agent':
            return _mock_model(root_response)
        return _mock_model(child_response)

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               side_effect=model_factory):
        runner = LangGraphAgentGraphRunner(graph, {})
        result = await runner.run('hello')

    assert result.metrics.success is True

    # Simulate per-node token events via callback handler (mock models don't fire callbacks)
    mock_ld_client2 = MagicMock()
    graph2 = _make_two_node_graph(mock_ld_client2)
    tracker2 = graph2.create_tracker()

    handler = LDMetricsCallbackHandler({'root-agent', 'child-agent'}, {})

    root_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=root_run_id, name='root-agent')
    root_llm_result = LLMResult(
        generations=[[ChatGeneration(
            message=_AIMsg(content='Root done.', usage_metadata={'total_tokens': 15, 'input_tokens': 10, 'output_tokens': 5}),
            text='Root done.',
        )]],
        llm_output={},
    )
    handler.on_llm_end(root_llm_result, run_id=uuid4(), parent_run_id=root_run_id)

    child_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=child_run_id, name='child-agent')
    child_llm_result = LLMResult(
        generations=[[ChatGeneration(
            message=_AIMsg(content='Child done.', usage_metadata={'total_tokens': 5, 'input_tokens': 3, 'output_tokens': 2}),
            text='Child done.',
        )]],
        llm_output={},
    )
    handler.on_llm_end(child_llm_result, run_id=uuid4(), parent_run_id=child_run_id)

    handler.flush(graph2)

    ev2 = _events(mock_ld_client2)

    # Per-node token events identified by configKey
    root_tokens = [(d, v) for d, v in ev2.get('$ld:ai:tokens:total', []) if d.get('configKey') == 'root-agent']
    child_tokens = [(d, v) for d, v in ev2.get('$ld:ai:tokens:total', []) if d.get('configKey') == 'child-agent']
    assert root_tokens[0][1] == 15
    assert child_tokens[0][1] == 5

    # Graph-level total from the real runner run
    ev = _events(mock_ld_client)
    assert ev['$ld:ai:graph:total_tokens'][0][1] == 20

    # Execution path includes both node keys (from real run)
    path_data = ev['$ld:ai:graph:path'][0][0]
    assert 'root-agent' in path_data['path']
    assert 'child-agent' in path_data['path']


def _make_multi_child_graph(mock_ld_client: MagicMock) -> 'AgentGraphDefinition':
    """Build a 3-node graph: orchestrator → agent-a, orchestrator → agent-b."""
    context = MagicMock()

    def _node_tracker(key: str) -> LDAIConfigTracker:
        return LDAIConfigTracker(
            ld_client=mock_ld_client,
            run_id='test-run-id',
            variation_key='test-variation',
            config_key=key,
            version=1,
            model_name='gpt-4',
            provider_name='openai',
            context=context,
            graph_key='multi-child-graph',
        )

    graph_tracker = AIGraphTracker(
        ld_client=mock_ld_client,
        variation_key='test-variation',
        graph_key='multi-child-graph',
        version=1,
        context=context,
    )

    configs = {
        'orchestrator': AIAgentConfig(
            key='orchestrator',
            enabled=True,
            model=ModelConfig(name='gpt-4', parameters={}),
            provider=ProviderConfig(name='openai'),
            instructions='Route to the appropriate specialist agent.',
            create_tracker=lambda: _node_tracker('orchestrator'),
        ),
        'agent-a': AIAgentConfig(
            key='agent-a',
            enabled=True,
            model=ModelConfig(name='gpt-4', parameters={}),
            provider=ProviderConfig(name='openai'),
            instructions='You handle topic A.',
            create_tracker=lambda: _node_tracker('agent-a'),
        ),
        'agent-b': AIAgentConfig(
            key='agent-b',
            enabled=True,
            model=ModelConfig(name='gpt-4', parameters={}),
            provider=ProviderConfig(name='openai'),
            instructions='You handle topic B.',
            create_tracker=lambda: _node_tracker('agent-b'),
        ),
    }

    edges = [
        Edge(key='orch-to-a', source_config='orchestrator', target_config='agent-a'),
        Edge(key='orch-to-b', source_config='orchestrator', target_config='agent-b'),
    ]
    graph_config = AIAgentGraphConfig(
        key='multi-child-graph',
        root_config_key='orchestrator',
        edges=edges,
        enabled=True,
    )
    nodes = AgentGraphDefinition.build_nodes(graph_config, configs)
    return AgentGraphDefinition(
        agent_graph=graph_config,
        nodes=nodes,
        context=context,
        enabled=True,
        create_tracker=lambda: graph_tracker,
    )


@pytest.mark.asyncio
async def test_multi_child_routes_via_handoff_not_fan_out():
    """Orchestrator with two children routes to exactly one child via handoff tool,
    not a fan-out that invokes both children."""
    from langchain_core.messages import AIMessage

    mock_ld_client = MagicMock()
    graph = _make_multi_child_graph(mock_ld_client)

    # Orchestrator calls transfer_to_agent_a (handoff tool name derived from child key)
    orchestrator_response = AIMessage(
        content='',
        tool_calls=[{
            'name': 'transfer_to_agent_a',
            'args': {},
            'id': 'call_handoff_1',
            'type': 'tool_call',
        }],
    )
    agent_a_response = _make_fake_response('Agent A handled it.')
    agent_b_model = _mock_model(_make_fake_response('Agent B handled it.'))

    def model_factory(node_config, **kwargs):
        if node_config.key == 'orchestrator':
            return _mock_model(orchestrator_response)
        if node_config.key == 'agent-a':
            return _mock_model(agent_a_response)
        return agent_b_model

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               side_effect=model_factory):
        runner = LangGraphAgentGraphRunner(graph, {})
        result = await runner.run('hello')

    assert result.metrics.success is True
    assert 'Agent A' in result.output
    # Agent B's model must never have been invoked — no fan-out
    agent_b_model.ainvoke.assert_not_called()


def _make_multi_child_graph_with_tools(mock_ld_client: MagicMock, tool_names: list) -> 'AgentGraphDefinition':
    """Build a 3-node graph where the orchestrator also has functional tools."""
    context = MagicMock()

    def _node_tracker(key: str) -> LDAIConfigTracker:
        return LDAIConfigTracker(
            ld_client=mock_ld_client,
            run_id='test-run-id',
            variation_key='test-variation',
            config_key=key,
            version=1,
            model_name='gpt-4',
            provider_name='openai',
            context=context,
            graph_key='multi-child-tools-graph',
        )

    graph_tracker = AIGraphTracker(
        ld_client=mock_ld_client,
        variation_key='test-variation',
        graph_key='multi-child-tools-graph',
        version=1,
        context=context,
    )

    tool_defs = [{'name': n, 'type': 'function', 'description': '', 'parameters': {}} for n in tool_names]
    configs = {
        'orchestrator': AIAgentConfig(
            key='orchestrator',
            enabled=True,
            model=ModelConfig(name='gpt-4', parameters={'tools': tool_defs}),
            provider=ProviderConfig(name='openai'),
            instructions='Route to a specialist after gathering info.',
            create_tracker=lambda: _node_tracker('orchestrator'),
        ),
        'agent-a': AIAgentConfig(
            key='agent-a',
            enabled=True,
            model=ModelConfig(name='gpt-4', parameters={}),
            provider=ProviderConfig(name='openai'),
            instructions='You handle topic A.',
            create_tracker=lambda: _node_tracker('agent-a'),
        ),
        'agent-b': AIAgentConfig(
            key='agent-b',
            enabled=True,
            model=ModelConfig(name='gpt-4', parameters={}),
            provider=ProviderConfig(name='openai'),
            instructions='You handle topic B.',
            create_tracker=lambda: _node_tracker('agent-b'),
        ),
    }

    edges = [
        Edge(key='orch-to-a', source_config='orchestrator', target_config='agent-a'),
        Edge(key='orch-to-b', source_config='orchestrator', target_config='agent-b'),
    ]
    graph_config = AIAgentGraphConfig(
        key='multi-child-tools-graph',
        root_config_key='orchestrator',
        edges=edges,
        enabled=True,
    )
    nodes = AgentGraphDefinition.build_nodes(graph_config, configs)
    return AgentGraphDefinition(
        agent_graph=graph_config,
        nodes=nodes,
        context=context,
        enabled=True,
        create_tracker=lambda: graph_tracker,
    )


@pytest.mark.asyncio
async def test_functional_tool_loops_back_when_handoff_tools_present():
    """When a node has both functional tools and handoff tools, calling a functional
    tool must loop back to the node so the LLM sees the result — not silently terminate."""
    from langchain_core.messages import AIMessage

    mock_ld_client = MagicMock()
    graph = _make_multi_child_graph_with_tools(mock_ld_client, tool_names=['search'])

    # Step 1: orchestrator calls functional tool 'search'
    tool_call_response = AIMessage(
        content='',
        tool_calls=[{'name': 'search', 'args': {'query': 'topic A'}, 'id': 'call_search_1', 'type': 'tool_call'}],
    )
    # Step 2: after seeing tool result, orchestrator hands off to agent-a
    handoff_response = AIMessage(
        content='',
        tool_calls=[{'name': 'transfer_to_agent_a', 'args': {}, 'id': 'call_handoff_1', 'type': 'tool_call'}],
    )
    agent_a_response = _make_fake_response('Agent A handled it.')

    orchestrator_model = MagicMock()
    orchestrator_model.ainvoke = AsyncMock(side_effect=[tool_call_response, handoff_response])
    orchestrator_model.bind_tools.return_value = orchestrator_model

    agent_a_model = _mock_model(agent_a_response)
    agent_b_model = _mock_model(_make_fake_response('Agent B handled it.'))

    def search(query: str = '') -> str:
        """Search for information."""
        return f'results for {query}'

    tool_registry = {'search': search}

    def model_factory(node_config, **kwargs):
        if node_config.key == 'orchestrator':
            return orchestrator_model
        if node_config.key == 'agent-a':
            return agent_a_model
        return agent_b_model

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               side_effect=model_factory):
        runner = LangGraphAgentGraphRunner(graph, tool_registry)
        result = await runner.run('Find info and route to the right agent.')

    assert result.metrics.success is True
    assert 'Agent A' in result.output
    # Orchestrator must have been called twice: once before tool result, once after
    assert orchestrator_model.ainvoke.call_count == 2
    # Agent B must never have been invoked
    agent_b_model.ainvoke.assert_not_called()
