"""
Unit tests for LDMetricsCallbackHandler.

Tests the callback handler directly by simulating the events that LangChain
fires during a graph run — without needing a real or mock LangGraph execution.
"""

from collections import defaultdict
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ldai.agent_graph import AgentGraphDefinition
from ldai.models import AIAgentConfig, AIAgentGraphConfig, ModelConfig, ProviderConfig
from ldai.tracker import AIGraphTracker, LDAIConfigTracker, TokenUsage
from ldai.evaluator import Evaluator
from ldai_langchain.langgraph_callback_handler import LDMetricsCallbackHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(mock_ld_client: MagicMock, node_key: str = 'root-agent', graph_key: str = 'test-graph'):
    """Build a minimal single-node AgentGraphDefinition for flush() tests."""
    context = MagicMock()
    node_tracker = LDAIConfigTracker(
        ld_client=mock_ld_client,
        variation_key='v1',
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
        variation_key='v1',
        graph_key=graph_key,
        version=1,
        context=context,
    )
    node_config = AIAgentConfig(
        key=node_key,
        enabled=True,
        evaluator=Evaluator.noop(),
        model=ModelConfig(name='gpt-4', parameters={}),
        provider=ProviderConfig(name='openai'),
        instructions='Be helpful.',
        create_tracker=lambda: node_tracker,
    )
    graph_config = AIAgentGraphConfig(
        key=graph_key,
        root_config_key=node_key,
        edges=[],
        enabled=True,
    )
    nodes = AgentGraphDefinition.build_nodes(graph_config, {node_key: node_config})
    return AgentGraphDefinition(
        agent_graph=graph_config,
        nodes=nodes,
        context=context,
        enabled=True,
        create_tracker=lambda: graph_tracker,
    )


def _llm_result(total: int, prompt: int, completion: int) -> LLMResult:
    return LLMResult(
        generations=[[ChatGeneration(
            message=AIMessage(
                content='ok',
                usage_metadata={'total_tokens': total, 'input_tokens': prompt, 'output_tokens': completion},
            ),
            text='ok',
        )]],
        llm_output={},
    )


def _events(mock_ld_client: MagicMock) -> dict:
    result = defaultdict(list)
    for call in mock_ld_client.track.call_args_list:
        name, _ctx, data, value = call.args
        result[name].append((data, value))
    return dict(result)


# ---------------------------------------------------------------------------
# on_chain_start tests
# ---------------------------------------------------------------------------

def test_on_chain_start_records_agent_node():
    """Agent node name is recorded in path and run_to_node map."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, name='root-agent')
    assert handler.path == ['root-agent']


def test_on_chain_start_deduplicates_path():
    """Multiple starts for the same node appear only once in path."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    run_id1 = uuid4()
    run_id2 = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id1, name='root-agent')
    handler.on_chain_start({}, {}, run_id=run_id2, name='root-agent')
    assert handler.path == ['root-agent']


def test_on_chain_start_tools_node_attributed_to_agent():
    """A '__tools' chain start maps its run_id to the stripped agent node key."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    # Tool node should NOT appear in path
    assert handler.path == []
    # But the run_id should be attributed to the agent node for tool event lookup
    assert handler._run_to_node.get(tools_run_id) == 'root-agent'


def test_on_chain_start_unknown_name_ignored():
    """Names not in node_keys and not __tools suffixed are ignored."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    handler.on_chain_start({}, {}, run_id=uuid4(), name='some-other-chain')
    assert handler.path == []


def test_on_chain_start_none_name_ignored():
    """None name does not raise."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    handler.on_chain_start({}, {}, run_id=uuid4(), name=None)
    assert handler.path == []


def test_on_chain_start_tools_for_unknown_agent_ignored():
    """A '__tools' chain whose stripped name is not in node_keys is ignored."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, name='other-agent__tools')
    assert run_id not in handler._run_to_node


def test_on_chain_start_records_path_order():
    """Multiple distinct agent nodes appear in path in order of first appearance."""
    handler = LDMetricsCallbackHandler({'node-a', 'node-b'}, {})
    handler.on_chain_start({}, {}, run_id=uuid4(), name='node-a')
    handler.on_chain_start({}, {}, run_id=uuid4(), name='node-b')
    assert handler.path == ['node-a', 'node-b']


# ---------------------------------------------------------------------------
# on_chain_end / duration tests
# ---------------------------------------------------------------------------

def test_on_chain_end_accumulates_duration():
    """Duration is computed and stored after chain_end."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, name='root-agent')
    handler.on_chain_end({}, run_id=run_id)
    # Duration may be 0 on fast machines but the key must be present
    assert 'root-agent' in handler.node_durations_ms


def test_on_chain_end_accumulates_across_multiple_runs():
    """Duration accumulates (not overwritten) when a node runs multiple times."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})

    run1 = uuid4()
    handler.on_chain_start({}, {}, run_id=run1, name='root-agent')
    handler.on_chain_end({}, run_id=run1)
    duration_after_first = handler.node_durations_ms.get('root-agent', 0)

    run2 = uuid4()
    handler.on_chain_start({}, {}, run_id=run2, name='root-agent')
    handler.on_chain_end({}, run_id=run2)
    duration_after_second = handler.node_durations_ms.get('root-agent', 0)

    assert duration_after_second >= duration_after_first


def test_on_chain_end_unknown_run_id_ignored():
    """chain_end for an unknown run_id does not raise."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    handler.on_chain_end({}, run_id=uuid4())  # should not raise


# ---------------------------------------------------------------------------
# on_llm_end / token tests
# ---------------------------------------------------------------------------

def test_on_llm_end_accumulates_tokens():
    """Token usage from llm_output is recorded for the parent node."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')

    result = _llm_result(total=15, prompt=10, completion=5)
    handler.on_llm_end(result, run_id=uuid4(), parent_run_id=node_run_id)

    tokens = handler.node_tokens.get('root-agent')
    assert tokens is not None
    assert tokens.total == 15
    assert tokens.input == 10
    assert tokens.output == 5


def test_on_llm_end_accumulates_across_multiple_calls():
    """Multiple LLM calls for the same node accumulate token counts."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')

    result1 = _llm_result(total=10, prompt=7, completion=3)
    result2 = _llm_result(total=6, prompt=4, completion=2)
    handler.on_llm_end(result1, run_id=uuid4(), parent_run_id=node_run_id)
    handler.on_llm_end(result2, run_id=uuid4(), parent_run_id=node_run_id)

    tokens = handler.node_tokens['root-agent']
    assert tokens.total == 16
    assert tokens.input == 11
    assert tokens.output == 5


def test_on_llm_end_none_parent_run_id_ignored():
    """LLM end with parent_run_id=None does not raise."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    result = _llm_result(total=5, prompt=3, completion=2)
    handler.on_llm_end(result, run_id=uuid4(), parent_run_id=None)
    assert handler.node_tokens == {}


def test_on_llm_end_unknown_parent_run_id_ignored():
    """LLM end for a run_id not in _run_to_node is silently ignored."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    result = _llm_result(total=5, prompt=3, completion=2)
    handler.on_llm_end(result, run_id=uuid4(), parent_run_id=uuid4())
    assert handler.node_tokens == {}


def test_on_llm_end_camel_case_token_keys():
    """camelCase token keys in response_metadata (e.g. some AWS Bedrock models) are parsed."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')

    msg = AIMessage(content='ok', response_metadata={
        'tokenUsage': {'totalTokens': 20, 'promptTokens': 12, 'completionTokens': 8}
    })
    result = LLMResult(
        generations=[[ChatGeneration(message=msg, text='ok')]],
        llm_output={},
    )
    handler.on_llm_end(result, run_id=uuid4(), parent_run_id=node_run_id)

    tokens = handler.node_tokens.get('root-agent')
    assert tokens is not None
    assert tokens.total == 20
    assert tokens.input == 12
    assert tokens.output == 8


# ---------------------------------------------------------------------------
# on_tool_end tests
# ---------------------------------------------------------------------------

def test_on_tool_end_records_tool_call():
    """Tool end event records config key for the owning agent node."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {'fetch_weather': 'get_weather_open_meteo'})
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    handler.on_tool_end('sunny', run_id=uuid4(), parent_run_id=tools_run_id, name='fetch_weather')
    assert handler.node_tool_calls.get('root-agent') == ['get_weather_open_meteo']


def test_on_tool_end_skips_unregistered_tools():
    """Tool end is ignored for tools not in the fn_name_to_config_key map (e.g. handoff tools)."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    handler.on_tool_end('result', run_id=uuid4(), parent_run_id=tools_run_id, name='transfer_to_child')
    assert handler.node_tool_calls.get('root-agent') is None


def test_on_tool_end_multiple_tools_accumulated():
    """Multiple tool calls are accumulated in order."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {'search': 'search', 'summarize': 'summarize'})
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    handler.on_tool_end('r1', run_id=uuid4(), parent_run_id=tools_run_id, name='search')
    handler.on_tool_end('r2', run_id=uuid4(), parent_run_id=tools_run_id, name='summarize')
    assert handler.node_tool_calls.get('root-agent') == ['search', 'summarize']


def test_on_tool_end_none_parent_run_id_ignored():
    """Tool end with parent_run_id=None does not raise."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    handler.on_tool_end('result', run_id=uuid4(), parent_run_id=None, name='my_tool')
    assert handler.node_tool_calls == {}


def test_on_tool_end_none_name_ignored():
    """Tool end with name=None does not raise."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, name='root-agent')
    handler.on_tool_end('result', run_id=uuid4(), parent_run_id=run_id, name=None)
    assert handler.node_tool_calls == {}


# ---------------------------------------------------------------------------
# flush() tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flush_emits_token_events_to_ld_tracker():
    """flush() calls track_tokens on the node's config tracker."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, node_key='root-agent', graph_key='g1')
    tracker = graph.create_tracker()

    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')
    handler.on_llm_end(_llm_result(15, 10, 5), run_id=uuid4(), parent_run_id=node_run_id)
    await handler.flush(graph)

    ev = _events(mock_ld_client)
    assert ev['$ld:ai:tokens:total'][0][1] == 15
    assert ev['$ld:ai:tokens:input'][0][1] == 10
    assert ev['$ld:ai:tokens:output'][0][1] == 5
    assert ev['$ld:ai:generation:success'][0][1] == 1


@pytest.mark.asyncio
async def test_flush_emits_duration():
    """flush() calls track_duration when duration was recorded."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client)
    tracker = graph.create_tracker()

    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, name='root-agent')
    handler.on_chain_end({}, run_id=run_id)
    await handler.flush(graph)

    ev = _events(mock_ld_client)
    assert '$ld:ai:duration:total' in ev


@pytest.mark.asyncio
async def test_flush_emits_tool_calls():
    """flush() calls track_tool_call for each recorded tool invocation."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client)
    tracker = graph.create_tracker()

    handler = LDMetricsCallbackHandler({'root-agent'}, {'fn_search': 'search'})
    # The agent node must be started first so it appears in the path for flush()
    agent_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=agent_run_id, name='root-agent')
    # Tool calls are attributed via the __tools chain run_id
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    handler.on_tool_end('r', run_id=uuid4(), parent_run_id=tools_run_id, name='fn_search')
    await handler.flush(graph)

    ev = _events(mock_ld_client)
    tool_events = ev.get('$ld:ai:tool_call', [])
    assert len(tool_events) == 1
    assert tool_events[0][0]['toolKey'] == 'search'


@pytest.mark.asyncio
async def test_flush_includes_graph_key_in_node_events():
    """flush() passes graph_key to the node tracker so graphKey appears in events."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client, graph_key='my-graph')
    tracker = graph.create_tracker()

    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')
    handler.on_llm_end(_llm_result(5, 3, 2), run_id=uuid4(), parent_run_id=node_run_id)
    await handler.flush(graph)

    ev = _events(mock_ld_client)
    token_data = ev['$ld:ai:tokens:total'][0][0]
    assert token_data.get('graphKey') == 'my-graph'


@pytest.mark.asyncio
async def test_flush_with_no_graph_key_on_node_tracker():
    """When node tracker has no graph_key, events omit graphKey."""
    mock_ld_client = MagicMock()
    context = MagicMock()
    node_tracker = LDAIConfigTracker(
        ld_client=mock_ld_client,
        variation_key='v1',
        config_key='root-agent',
        version=1,
        model_name='gpt-4',
        provider_name='openai',
        context=context,
        run_id='test-run-id',
    )
    node_config = AIAgentConfig(
        key='root-agent',
        enabled=True,
        evaluator=Evaluator.noop(),
        model=ModelConfig(name='gpt-4', parameters={}),
        provider=ProviderConfig(name='openai'),
        instructions='Be helpful.',
        create_tracker=lambda: node_tracker,
    )
    graph_config = AIAgentGraphConfig(
        key='test-graph',
        root_config_key='root-agent',
        edges=[],
        enabled=True,
    )
    nodes = AgentGraphDefinition.build_nodes(graph_config, {'root-agent': node_config})
    graph = AgentGraphDefinition(
        agent_graph=graph_config,
        nodes=nodes,
        context=context,
        enabled=True,
        create_tracker=lambda: AIGraphTracker(mock_ld_client, 'v1', 'test-graph', 1, context),
    )

    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')
    handler.on_llm_end(_llm_result(5, 3, 2), run_id=uuid4(), parent_run_id=node_run_id)
    await handler.flush(graph)

    ev = _events(mock_ld_client)
    token_data = ev['$ld:ai:tokens:total'][0][0]
    assert 'graphKey' not in token_data


@pytest.mark.asyncio
async def test_flush_skips_nodes_not_in_path():
    """flush() only emits events for nodes that were actually executed."""
    mock_ld_client = MagicMock()
    graph = _make_graph(mock_ld_client)
    tracker = graph.create_tracker()

    # Handler with 'root-agent' in node_keys but never started
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    await handler.flush(graph)

    ev = _events(mock_ld_client)
    assert '$ld:ai:tokens:total' not in ev
    assert '$ld:ai:generation:success' not in ev


@pytest.mark.asyncio
async def test_flush_skips_node_without_tracker():
    """flush() silently skips nodes whose config has no tracker."""
    mock_ld_client = MagicMock()
    context = MagicMock()

    node_config_no_tracker = AIAgentConfig(
        key='no-track',
        enabled=True,
        create_tracker=lambda: None,
        evaluator=Evaluator.noop(),
        model=ModelConfig(name='gpt-4', parameters={}),
        provider=ProviderConfig(name='openai'),
        instructions='',
    )
    graph_config = AIAgentGraphConfig(
        key='g', root_config_key='no-track', edges=[], enabled=True
    )
    nodes = AgentGraphDefinition.build_nodes(graph_config, {'no-track': node_config_no_tracker})
    graph = AgentGraphDefinition(
        agent_graph=graph_config,
        nodes=nodes,
        context=context,
        enabled=True,
        create_tracker=lambda: None,
    )

    handler = LDMetricsCallbackHandler({'no-track'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='no-track')
    handler.on_llm_end(_llm_result(5, 3, 2), run_id=uuid4(), parent_run_id=node_run_id)
    await handler.flush(graph)  # should not raise

    mock_ld_client.track.assert_not_called()


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------

def test_path_property_returns_copy():
    """Mutating the returned path does not affect the handler's internal state."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    handler.on_chain_start({}, {}, run_id=uuid4(), name='root-agent')
    path = handler.path
    path.append('extra')
    assert handler.path == ['root-agent']


def test_node_tokens_property_returns_copy():
    """Mutating the returned dict does not affect the handler's internal state."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')
    handler.on_llm_end(_llm_result(5, 3, 2), run_id=uuid4(), parent_run_id=node_run_id)
    tokens = handler.node_tokens
    tokens['other'] = TokenUsage(total=1, input=1, output=0)
    assert 'other' not in handler.node_tokens
