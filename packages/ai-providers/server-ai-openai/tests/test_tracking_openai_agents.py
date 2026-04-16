"""
Integration tests for OpenAIAgentGraphRunner tracking pipeline.

Uses real AIGraphTracker and LDAIConfigTracker backed by a mock LD client,
and a crafted RunResult to verify that the correct LD events are emitted
with the correct payloads — without making real API calls.
"""

import pytest
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

from ldai.agent_graph import AgentGraphDefinition
from ldai.models import AIAgentGraphConfig, AIAgentConfig, Edge, ModelConfig, ProviderConfig
from ldai.tracker import AIGraphTracker, LDAIConfigTracker
from ldai_openai.openai_agent_graph_runner import OpenAIAgentGraphRunner


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


def _make_run_result(
    output: str = 'agent answer',
    total_tokens: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    tool_call_items: list = None,
) -> MagicMock:
    """
    Build a mock RunResult that resembles the openai-agents SDK RunResult shape
    expected by OpenAIAgentGraphRunner.
    """
    entry = MagicMock()
    entry.total_tokens = total_tokens
    entry.input_tokens = input_tokens
    entry.output_tokens = output_tokens

    result = MagicMock()
    result.final_output = output
    result.new_items = tool_call_items or []
    result.usage = None  # prevent fallthrough to .usage attribute in get_ai_usage_from_response
    result.context_wrapper.usage.total_tokens = total_tokens
    result.context_wrapper.usage.input_tokens = input_tokens
    result.context_wrapper.usage.output_tokens = output_tokens
    result.context_wrapper.usage.request_usage_entries = [entry]
    return result


def _tool_registry(*config_names: str) -> dict:
    """Registry entries whose callable __name__ matches runtime tool names from the SDK."""

    def _stub(name: str):
        def fn():
            pass

        fn.__name__ = name
        return fn

    return {n: _stub(n) for n in config_names}


def _make_tool_call_item(agent_name: str, tool_name: str) -> MagicMock:
    """
    Create a mock ToolCallItem with a ResponseFunctionToolCall raw item so that
    get_tool_calls_from_run_items() correctly extracts the tool name.
    """
    from agents.items import ToolCallItem
    from openai.types.responses import ResponseFunctionToolCall

    raw = MagicMock(spec=ResponseFunctionToolCall)
    raw.name = tool_name

    agent = MagicMock()
    agent.name = agent_name

    item = MagicMock(spec=ToolCallItem)
    item.agent = agent
    item.raw_item = raw
    return item


def _make_agents_modules(run_result: MagicMock) -> dict:
    """Build the sys.modules patch dict for the agents package."""
    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(return_value=run_result)

    mock_agents = MagicMock()
    mock_agents.Runner = mock_runner
    mock_agents.Agent = MagicMock(return_value=MagicMock())
    mock_agents.Handoff = MagicMock()
    mock_agents.Tool = MagicMock()
    mock_agents.function_tool = lambda fn: MagicMock()
    mock_agents.handoff = MagicMock(return_value=MagicMock())

    mock_ext = MagicMock()
    mock_ext.RECOMMENDED_PROMPT_PREFIX = '[PREFIX]'

    return {
        'agents': mock_agents,
        'agents.extensions': MagicMock(),
        'agents.extensions.handoff_prompt': mock_ext,
        'agents.tool_context': MagicMock(),
    }


def _make_two_node_graph(mock_ld_client: MagicMock) -> AgentGraphDefinition:
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


def _events(mock_ld_client: MagicMock) -> dict:
    """Return dict of event_name -> list of (data, value) from all track() calls."""
    result = defaultdict(list)
    for call in mock_ld_client.track.call_args_list:
        name, _ctx, data, value = call.args
        result[name].append((data, value))
    return dict(result)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tracks_graph_invocation_success_and_latency():
    """Graph-level success and latency events fire on a successful run."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client)
    run_result = _make_run_result(output='done')

    with patch.dict('sys.modules', _make_agents_modules(run_result)):
        runner = OpenAIAgentGraphRunner(graph, {})
        result = await runner.run('hello')

    assert result.metrics.success is True
    assert result.output == 'done'

    ev = _events(mock_ld_client)
    assert ev['$ld:ai:graph:invocation_success'][0][1] == 1
    assert '$ld:ai:graph:latency' in ev
    assert '$ld:ai:graph:path' in ev


@pytest.mark.asyncio
async def test_tracks_per_node_tokens_and_success():
    """Node-level token and success events fire with correct values."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, node_key='root-agent', graph_key='test-graph')
    run_result = _make_run_result(
        output='answer',
        total_tokens=30,
        input_tokens=20,
        output_tokens=10,
    )

    with patch.dict('sys.modules', _make_agents_modules(run_result)):
        runner = OpenAIAgentGraphRunner(graph, {})
        await runner.run('hello')

    ev = _events(mock_ld_client)

    # Node-level events
    assert ev['$ld:ai:tokens:total'][0][1] == 30
    assert ev['$ld:ai:tokens:input'][0][1] == 20
    assert ev['$ld:ai:tokens:output'][0][1] == 10
    assert ev['$ld:ai:generation:success'][0][1] == 1

    # Graph-level total tokens
    assert ev['$ld:ai:graph:total_tokens'][0][1] == 30


@pytest.mark.asyncio
async def test_tracks_graph_key_on_node_events():
    """Node-level events include graphKey so they can be correlated to the graph."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, graph_key='my-graph')
    run_result = _make_run_result(total_tokens=15, input_tokens=10, output_tokens=5)

    with patch.dict('sys.modules', _make_agents_modules(run_result)):
        runner = OpenAIAgentGraphRunner(graph, {})
        await runner.run('hello')

    ev = _events(mock_ld_client)
    token_data = ev['$ld:ai:tokens:total'][0][0]
    assert token_data.get('graphKey') == 'my-graph'


@pytest.mark.asyncio
async def test_tracks_tool_calls_from_run_items():
    """A tool_call event fires for tools registered on the graph and in the tool registry."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, node_key='root-agent', tool_names=['get_weather'])

    tool_item = _make_tool_call_item('root-agent', 'get_weather')
    run_result = _make_run_result(output='done', tool_call_items=[tool_item])

    with patch.dict('sys.modules', _make_agents_modules(run_result)):
        runner = OpenAIAgentGraphRunner(graph, _tool_registry('get_weather'))
        await runner.run('What is the weather?')

    ev = _events(mock_ld_client)
    tool_events = ev.get('$ld:ai:tool_call', [])
    assert len(tool_events) == 1
    assert tool_events[0][0]['toolKey'] == 'get_weather'


@pytest.mark.asyncio
async def test_tracks_multiple_tool_calls():
    """One tool_call event fires per registered tool in RunResult.new_items."""
    mock_ld_client = MagicMock()
    graph = _make_graph(
        mock_ld_client, node_key='root-agent', tool_names=['search', 'summarize']
    )

    items = [
        _make_tool_call_item('root-agent', 'search'),
        _make_tool_call_item('root-agent', 'summarize'),
    ]
    run_result = _make_run_result(output='done', tool_call_items=items)

    with patch.dict('sys.modules', _make_agents_modules(run_result)):
        runner = OpenAIAgentGraphRunner(graph, _tool_registry('search', 'summarize'))
        await runner.run('Search and summarize.')

    ev = _events(mock_ld_client)
    tool_keys = [data['toolKey'] for data, _ in ev.get('$ld:ai:tool_call', [])]
    assert sorted(tool_keys) == ['search', 'summarize']


@pytest.mark.asyncio
async def test_does_not_track_tool_calls_without_graph_and_registry_config():
    """RunResult tool items that are not backed by graph + registry tools are ignored."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, node_key='root-agent')

    tool_item = _make_tool_call_item('root-agent', 'orphan_tool')
    run_result = _make_run_result(output='done', tool_call_items=[tool_item])

    with patch.dict('sys.modules', _make_agents_modules(run_result)):
        runner = OpenAIAgentGraphRunner(graph, {})
        await runner.run('prompt')

    ev = _events(mock_ld_client)
    assert ev.get('$ld:ai:tool_call', []) == []


@pytest.mark.asyncio
async def test_tracks_failure_and_latency_on_runner_error():
    """When Runner.run raises, failure and latency events fire; success does not."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client)

    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(side_effect=RuntimeError('runner error'))
    mock_agents = MagicMock()
    mock_agents.Runner = mock_runner
    mock_agents.Agent = MagicMock(return_value=MagicMock())
    mock_agents.Handoff = MagicMock()
    mock_agents.Tool = MagicMock()
    mock_agents.function_tool = lambda fn: MagicMock()
    mock_agents.handoff = MagicMock(return_value=MagicMock())
    mock_ext = MagicMock()
    mock_ext.RECOMMENDED_PROMPT_PREFIX = '[PREFIX]'

    with patch.dict('sys.modules', {
        'agents': mock_agents,
        'agents.extensions': MagicMock(),
        'agents.extensions.handoff_prompt': mock_ext,
        'agents.tool_context': MagicMock(),
    }):
        runner = OpenAIAgentGraphRunner(graph, {})
        result = await runner.run('fail')

    assert result.metrics.success is False

    ev = _events(mock_ld_client)
    assert '$ld:ai:graph:invocation_failure' in ev
    assert '$ld:ai:graph:latency' in ev
    assert '$ld:ai:graph:invocation_success' not in ev


@pytest.mark.asyncio
async def test_multi_node_tracks_per_node_tokens_and_handoff():
    """Each node emits its own token events; handoff event fires between them."""
    mock_ld_client = MagicMock()
    graph = _make_two_node_graph(mock_ld_client)

    root_entry = MagicMock()
    root_entry.total_tokens = 15
    root_entry.input_tokens = 10
    root_entry.output_tokens = 5

    child_entry = MagicMock()
    child_entry.total_tokens = 9
    child_entry.input_tokens = 6
    child_entry.output_tokens = 3

    run_result = MagicMock()
    run_result.final_output = 'child answer'
    run_result.new_items = []
    run_result.usage = None
    run_result.context_wrapper.usage.total_tokens = 24
    run_result.context_wrapper.usage.input_tokens = 16
    run_result.context_wrapper.usage.output_tokens = 8
    run_result.context_wrapper.usage.request_usage_entries = [root_entry, child_entry]

    on_handoff_callbacks = []

    def capture_handoff(**kwargs):
        cb = kwargs.get('on_handoff')
        if cb:
            on_handoff_callbacks.append(cb)
        return MagicMock()

    async def mock_run(agent, input_str, **kwargs):
        # Simulate the root→child handoff before returning
        if on_handoff_callbacks:
            run_ctx = MagicMock()
            run_ctx.usage.request_usage_entries = [root_entry]
            on_handoff_callbacks[0](run_ctx)
        return run_result

    mock_runner_cls = MagicMock()
    mock_runner_cls.run = mock_run

    mock_agents = MagicMock()
    mock_agents.Runner = mock_runner_cls
    mock_agents.Agent = MagicMock(return_value=MagicMock())
    mock_agents.Handoff = MagicMock()
    mock_agents.Tool = MagicMock()
    mock_agents.function_tool = lambda fn: MagicMock()
    mock_agents.handoff = capture_handoff

    mock_ext = MagicMock()
    mock_ext.RECOMMENDED_PROMPT_PREFIX = '[PREFIX]'

    with patch.dict('sys.modules', {
        'agents': mock_agents,
        'agents.extensions': MagicMock(),
        'agents.extensions.handoff_prompt': mock_ext,
        'agents.tool_context': MagicMock(),
    }):
        runner = OpenAIAgentGraphRunner(graph, {})
        result = await runner.run('hello')

    assert result.metrics.success is True

    ev = _events(mock_ld_client)

    # Per-node token events identified by configKey
    root_tokens = [(d, v) for d, v in ev.get('$ld:ai:tokens:total', []) if d.get('configKey') == 'root-agent']
    child_tokens = [(d, v) for d, v in ev.get('$ld:ai:tokens:total', []) if d.get('configKey') == 'child-agent']
    assert root_tokens[0][1] == 15
    assert child_tokens[0][1] == 9

    # Execution path includes both node keys
    path_data = ev['$ld:ai:graph:path'][0][0]
    assert 'root-agent' in path_data['path']
    assert 'child-agent' in path_data['path']

    # Handoff event fires with correct source and target
    handoff_events = ev.get('$ld:ai:graph:handoff_success', [])
    assert len(handoff_events) == 1
    assert handoff_events[0][0]['sourceKey'] == 'root-agent'
    assert handoff_events[0][0]['targetKey'] == 'child-agent'
