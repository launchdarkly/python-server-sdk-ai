"""
Unit tests for LDMetricsCallbackHandler.

Tests the callback handler directly by simulating the events that LangChain
fires during a graph run — without needing a real or mock LangGraph execution.
"""

from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ldai_langchain.langgraph_callback_handler import LDMetricsCallbackHandler


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


# ---------------------------------------------------------------------------
# on_chain_start tests
# ---------------------------------------------------------------------------

def test_on_chain_start_records_agent_node():
    """Agent node name is recorded in path and run_to_node map."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, name='root-agent')
    assert handler.path == ['root-agent']


def test_on_chain_start_seeds_node_metrics():
    """Agent node gets an LDAIMetrics entry with success=False on first chain_start."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    handler.on_chain_start({}, {}, run_id=uuid4(), name='root-agent')
    metrics = handler.node_metrics
    assert 'root-agent' in metrics
    assert metrics['root-agent'].success is False


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
    assert handler.node_metrics['root-agent'].duration_ms is not None
    assert handler.node_metrics['root-agent'].success is True


def test_on_chain_end_accumulates_across_multiple_runs():
    """Duration accumulates (not overwritten) when a node runs multiple times."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})

    run1 = uuid4()
    handler.on_chain_start({}, {}, run_id=run1, name='root-agent')
    handler.on_chain_end({}, run_id=run1)
    duration_after_first = handler.node_metrics['root-agent'].duration_ms or 0

    run2 = uuid4()
    handler.on_chain_start({}, {}, run_id=run2, name='root-agent')
    handler.on_chain_end({}, run_id=run2)
    duration_after_second = handler.node_metrics['root-agent'].duration_ms or 0

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

    usage = handler.node_metrics['root-agent'].usage
    assert usage is not None
    assert usage.total == 15
    assert usage.input == 10
    assert usage.output == 5


def test_on_llm_end_accumulates_across_multiple_calls():
    """Multiple LLM calls for the same node accumulate token counts."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')

    result1 = _llm_result(total=10, prompt=7, completion=3)
    result2 = _llm_result(total=6, prompt=4, completion=2)
    handler.on_llm_end(result1, run_id=uuid4(), parent_run_id=node_run_id)
    handler.on_llm_end(result2, run_id=uuid4(), parent_run_id=node_run_id)

    usage = handler.node_metrics['root-agent'].usage
    assert usage.total == 16
    assert usage.input == 11
    assert usage.output == 5


def test_on_llm_end_none_parent_run_id_ignored():
    """LLM end with parent_run_id=None does not raise."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    result = _llm_result(total=5, prompt=3, completion=2)
    handler.on_llm_end(result, run_id=uuid4(), parent_run_id=None)
    assert handler.node_metrics == {}


def test_on_llm_end_unknown_parent_run_id_ignored():
    """LLM end for a run_id not in _run_to_node is silently ignored."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    result = _llm_result(total=5, prompt=3, completion=2)
    handler.on_llm_end(result, run_id=uuid4(), parent_run_id=uuid4())
    assert handler.node_metrics == {}


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

    usage = handler.node_metrics['root-agent'].usage
    assert usage is not None
    assert usage.total == 20
    assert usage.input == 12
    assert usage.output == 8


# ---------------------------------------------------------------------------
# on_tool_end tests
# ---------------------------------------------------------------------------

def test_on_tool_end_records_tool_call():
    """Tool end event records config key for the owning agent node."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {'fetch_weather': 'get_weather_open_meteo'})
    agent_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=agent_run_id, name='root-agent')
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    handler.on_tool_end('sunny', run_id=uuid4(), parent_run_id=tools_run_id, name='fetch_weather')
    assert handler.node_metrics['root-agent'].tool_calls == ['get_weather_open_meteo']


def test_on_tool_end_skips_unregistered_tools():
    """Tool end is ignored for tools not in the fn_name_to_config_key map (e.g. handoff tools)."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    agent_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=agent_run_id, name='root-agent')
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    handler.on_tool_end('result', run_id=uuid4(), parent_run_id=tools_run_id, name='transfer_to_child')
    assert handler.node_metrics['root-agent'].tool_calls is None


def test_on_tool_end_multiple_tools_accumulated():
    """Multiple tool calls are accumulated in order."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {'search': 'search', 'summarize': 'summarize'})
    agent_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=agent_run_id, name='root-agent')
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    handler.on_tool_end('r1', run_id=uuid4(), parent_run_id=tools_run_id, name='search')
    handler.on_tool_end('r2', run_id=uuid4(), parent_run_id=tools_run_id, name='summarize')
    assert handler.node_metrics['root-agent'].tool_calls == ['search', 'summarize']


def test_on_tool_end_none_parent_run_id_ignored():
    """Tool end with parent_run_id=None does not raise."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    handler.on_tool_end('result', run_id=uuid4(), parent_run_id=None, name='my_tool')
    assert handler.node_metrics == {}


def test_on_tool_end_none_name_ignored():
    """Tool end with name=None does not raise."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, name='root-agent')
    handler.on_tool_end('result', run_id=uuid4(), parent_run_id=run_id, name=None)
    assert handler.node_metrics['root-agent'].tool_calls is None


# ---------------------------------------------------------------------------
# node_metrics property tests
# ---------------------------------------------------------------------------

def test_node_metrics_includes_tokens():
    """node_metrics returns token usage for nodes that received LLM calls."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')
    handler.on_llm_end(_llm_result(15, 10, 5), run_id=uuid4(), parent_run_id=node_run_id)

    metrics = handler.node_metrics

    assert 'root-agent' in metrics
    node = metrics['root-agent']
    assert node.usage is not None
    assert node.usage.total == 15
    assert node.usage.input == 10
    assert node.usage.output == 5


def test_node_metrics_includes_duration():
    """node_metrics returns duration_ms for nodes that completed a chain run."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, name='root-agent')
    handler.on_chain_end({}, run_id=run_id)

    metrics = handler.node_metrics

    assert 'root-agent' in metrics
    assert metrics['root-agent'].duration_ms is not None
    assert metrics['root-agent'].success is True


def test_node_metrics_includes_tool_calls():
    """node_metrics returns tool_calls for nodes with recorded tool invocations."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {'fn_search': 'search'})
    agent_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=agent_run_id, name='root-agent')
    tools_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=tools_run_id, name='root-agent__tools')
    handler.on_tool_end('r', run_id=uuid4(), parent_run_id=tools_run_id, name='fn_search')

    metrics = handler.node_metrics

    assert 'root-agent' in metrics
    assert metrics['root-agent'].tool_calls == ['search']


def test_node_metrics_empty_when_no_nodes_executed():
    """node_metrics returns an empty dict when no nodes were executed."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})

    metrics = handler.node_metrics

    assert metrics == {}


def test_node_metrics_multiple_nodes():
    """node_metrics returns separate entries for each executed node."""
    handler = LDMetricsCallbackHandler({'root-agent', 'child-agent'}, {})

    root_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=root_run_id, name='root-agent')
    handler.on_llm_end(_llm_result(15, 10, 5), run_id=uuid4(), parent_run_id=root_run_id)

    child_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=child_run_id, name='child-agent')
    handler.on_llm_end(_llm_result(5, 3, 2), run_id=uuid4(), parent_run_id=child_run_id)

    metrics = handler.node_metrics

    assert 'root-agent' in metrics
    assert 'child-agent' in metrics
    assert metrics['root-agent'].usage.total == 15
    assert metrics['child-agent'].usage.total == 5


def test_node_metrics_no_tool_calls_returns_none():
    """node_metrics sets tool_calls to None for nodes with no tool invocations."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    node_run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=node_run_id, name='root-agent')
    handler.on_llm_end(_llm_result(5, 3, 2), run_id=uuid4(), parent_run_id=node_run_id)

    metrics = handler.node_metrics

    assert metrics['root-agent'].tool_calls is None


def test_node_metrics_no_usage_returns_none():
    """node_metrics sets usage to None for nodes with no LLM calls."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, name='root-agent')
    handler.on_chain_end({}, run_id=run_id)

    metrics = handler.node_metrics

    assert metrics['root-agent'].usage is None


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


def test_node_metrics_property_returns_copy():
    """Mutating the returned dict does not affect the handler's internal state."""
    handler = LDMetricsCallbackHandler({'root-agent'}, {})
    handler.on_chain_start({}, {}, run_id=uuid4(), name='root-agent')
    metrics = handler.node_metrics
    del metrics['root-agent']
    assert 'root-agent' in handler.node_metrics
