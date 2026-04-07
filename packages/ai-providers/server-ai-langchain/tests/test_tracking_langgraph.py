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
        tracker=node_tracker,
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
        tracker=graph_tracker,
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
    )
    child_tracker = LDAIConfigTracker(
        ld_client=mock_ld_client,
        variation_key='test-variation',
        config_key='child-agent',
        version=1,
        model_name='gpt-4',
        provider_name='openai',
        context=context,
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
        tracker=root_tracker,
    )
    child_config = AIAgentConfig(
        key='child-agent',
        enabled=True,
        model=ModelConfig(name='gpt-4', parameters={}),
        provider=ProviderConfig(name='openai'),
        instructions='You are child.',
        tracker=child_tracker,
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
        tracker=graph_tracker,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tracks_node_and_graph_tokens_on_success():
    """Node-level and graph-level token events fire with the correct counts."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client)
    fake_response = _make_fake_response('Sunny.', input_tokens=10, output_tokens=5)

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               return_value=_mock_model(fake_response)):
        runner = LangGraphAgentGraphRunner(graph, {})
        result = await runner.run("What's the weather?")

    assert result.metrics.success is True
    assert result.output == 'Sunny.'

    ev = _events(mock_ld_client)

    # Node-level token events
    assert ev['$ld:ai:tokens:total'][0][1] == 15
    assert ev['$ld:ai:tokens:input'][0][1] == 10
    assert ev['$ld:ai:tokens:output'][0][1] == 5
    assert ev['$ld:ai:generation:success'][0][1] == 1
    assert '$ld:ai:duration:total' in ev

    # Graph-level events
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

    ev = _events(mock_ld_client)
    tool_events = ev.get('$ld:ai:tool_call', [])
    assert len(tool_events) == 1
    assert tool_events[0][0]['toolKey'] == 'get_weather'


@pytest.mark.asyncio
async def test_tracks_multiple_tool_calls():
    """One tool_call event fires per tool name in the response."""
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

    ev = _events(mock_ld_client)
    tool_keys = [data['toolKey'] for data, _ in ev.get('$ld:ai:tool_call', [])]
    assert sorted(tool_keys) == ['search', 'summarize']


@pytest.mark.asyncio
async def test_tracks_graph_key_on_node_events():
    """Node-level events include the graphKey so they can be correlated to the graph."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, graph_key='my-graph')
    fake_response = _make_fake_response('OK.', input_tokens=5, output_tokens=3)

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               return_value=_mock_model(fake_response)):
        runner = LangGraphAgentGraphRunner(graph, {})
        await runner.run('hello')

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
    mock_ld_client = MagicMock()
    graph = _make_two_node_graph(mock_ld_client)

    root_response = _make_fake_response('Root done.', input_tokens=10, output_tokens=5)
    child_response = _make_fake_response('Child done.', input_tokens=3, output_tokens=2)

    def model_factory(node_config):
        if node_config.key == 'root-agent':
            return _mock_model(root_response)
        return _mock_model(child_response)

    with patch('ldai_langchain.langgraph_agent_graph_runner.create_langchain_model',
               side_effect=model_factory):
        runner = LangGraphAgentGraphRunner(graph, {})
        result = await runner.run('hello')

    assert result.metrics.success is True

    ev = _events(mock_ld_client)

    # Per-node token events identified by configKey
    root_tokens = [(d, v) for d, v in ev.get('$ld:ai:tokens:total', []) if d.get('configKey') == 'root-agent']
    child_tokens = [(d, v) for d, v in ev.get('$ld:ai:tokens:total', []) if d.get('configKey') == 'child-agent']
    assert root_tokens[0][1] == 15
    assert child_tokens[0][1] == 5

    # Graph-level total accumulates both nodes (10+3 in, 5+2 out)
    assert ev['$ld:ai:graph:total_tokens'][0][1] == 20

    # Execution path includes both node keys
    path_data = ev['$ld:ai:graph:path'][0][0]
    assert 'root-agent' in path_data['path']
    assert 'child-agent' in path_data['path']
