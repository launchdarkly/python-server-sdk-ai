"""Tests for OpenAIAgentGraphRunner and OpenAIRunnerFactory.create_agent_graph()."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ldai.agent_graph import AgentGraphDefinition
from ldai.models import (
    AIAgentConfig,
    AIAgentGraphConfig,
    Edge,
    JudgeConfiguration,
    ModelConfig,
    ProviderConfig,
)
from ldai.providers import ToolRegistry
from ldai.providers.types import AgentGraphRunnerResult, AIGraphMetrics, EvalRequest
from ldai_openai.openai_agent_graph_runner import OpenAIAgentGraphRunner
from ldai_openai.openai_runner_factory import OpenAIRunnerFactory
from ldai.evaluator import Evaluator


def _make_graph(enabled: bool = True) -> AgentGraphDefinition:
    """Build a minimal single-node AgentGraphDefinition for testing."""
    node_factory = MagicMock()
    graph_factory = MagicMock()
    root_config = AIAgentConfig(
        key='root-agent',
        enabled=enabled,
        evaluator=Evaluator.noop(),
        model=ModelConfig(name='gpt-4'),
        provider=ProviderConfig(name='openai'),
        instructions='You are a helpful assistant.',
        create_tracker=node_factory,
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
        create_tracker=graph_factory,
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
    """Import failure returns AgentGraphRunnerResult with success=False."""
    graph = _make_graph()
    runner = OpenAIAgentGraphRunner(graph, {})

    with patch.dict('sys.modules', {'agents': None}):
        result = await runner.run("test input")
        assert isinstance(result, AgentGraphRunnerResult)
        assert result.metrics.success is False


@pytest.mark.asyncio
async def test_openai_agent_graph_runner_run_failure_returns_metrics():
    """On import failure, returned AIGraphMetrics has success=False (no tracker needed)."""
    graph = _make_graph()
    runner = OpenAIAgentGraphRunner(graph, {})

    with patch.dict('sys.modules', {'agents': None}):
        result = await runner.run("fail")

    assert isinstance(result, AgentGraphRunnerResult)
    assert result.metrics.success is False
    assert result.metrics.duration_ms is not None
    # Import failure happens before node metrics are created
    assert len(result.metrics.node_metrics) == 0
    # Runner no longer calls graph tracker — graph.create_tracker should NOT be called
    graph.create_tracker.assert_not_called()


@pytest.mark.asyncio
async def test_openai_agent_graph_runner_run_failure_marks_node_not_success():
    """When Runner.run() raises, started nodes retain success=False."""
    graph = _make_graph()

    mock_agents = MagicMock()
    mock_agents.Runner.run = AsyncMock(side_effect=RuntimeError("boom"))
    mock_agents.Agent = MagicMock(return_value=MagicMock())
    mock_agents.Handoff = MagicMock()
    mock_agents.handoff = MagicMock()

    mock_agents_ext = MagicMock()
    mock_agents_ext.RECOMMENDED_PROMPT_PREFIX = '[PREFIX]'

    with patch.dict('sys.modules', {
        'agents': mock_agents,
        'agents.extensions': MagicMock(),
        'agents.extensions.handoff_prompt': mock_agents_ext,
        'agents.tool_context': MagicMock(),
    }):
        runner = OpenAIAgentGraphRunner(graph, {})
        result = await runner.run("test input")

    assert result.metrics.success is False
    assert 'root-agent' in result.metrics.node_metrics
    assert result.metrics.node_metrics['root-agent'].success is False


@pytest.mark.asyncio
async def test_openai_agent_graph_runner_run_success():
    """Successful run returns AgentGraphRunnerResult with populated AIGraphMetrics."""
    graph = _make_graph()

    mock_result = MagicMock()
    mock_result.final_output = "agent answer"
    mock_result.new_items = []
    mock_result.context_wrapper.usage.total_tokens = 10
    mock_result.context_wrapper.usage.input_tokens = 5
    mock_result.context_wrapper.usage.output_tokens = 5
    mock_result.context_wrapper.usage.request_usage_entries = []

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

    assert isinstance(result, AgentGraphRunnerResult)
    assert result.content == "agent answer"
    assert isinstance(result.metrics, AIGraphMetrics)
    assert result.metrics.success is True
    assert result.metrics.duration_ms is not None
    assert 'root-agent' in result.metrics.path

    # Runner no longer creates or calls the graph tracker
    graph.create_tracker.assert_not_called()

    # Runner no longer creates per-node LDAIConfigTracker instances
    node_factory = graph.get_node('root-agent').get_config().create_tracker
    node_factory.assert_not_called()

    # Runner accumulates per-node metrics in _node_metrics
    assert 'root-agent' in runner._node_metrics
    assert runner._node_metrics['root-agent'].success is True


@pytest.mark.asyncio
async def test_openai_agent_graph_runner_run_resets_node_metrics_between_runs():
    """Successive runs do not leak stale node metrics from a previous run."""
    graph = _make_graph()

    mock_result = MagicMock()
    mock_result.final_output = "answer"
    mock_result.new_items = []
    mock_result.context_wrapper.usage.request_usage_entries = []

    mock_agents = MagicMock()
    mock_agents.Runner.run = AsyncMock(
        side_effect=[RuntimeError("boom"), mock_result]
    )
    mock_agents.Agent = MagicMock(return_value=MagicMock())
    mock_agents.Handoff = MagicMock()
    mock_agents.handoff = MagicMock()

    mock_agents_ext = MagicMock()
    mock_agents_ext.RECOMMENDED_PROMPT_PREFIX = '[PREFIX]'

    with patch.dict('sys.modules', {
        'agents': mock_agents,
        'agents.extensions': MagicMock(),
        'agents.extensions.handoff_prompt': mock_agents_ext,
        'agents.tool_context': MagicMock(),
    }):
        runner = OpenAIAgentGraphRunner(graph, {})

        first = await runner.run("attempt 1")
        assert first.metrics.success is False
        assert first.metrics.node_metrics['root-agent'].success is False
        failed_metrics = first.metrics.node_metrics['root-agent']

        second = await runner.run("attempt 2")
        assert second.metrics.success is True
        assert second.metrics.node_metrics['root-agent'].success is True
        assert second.metrics.node_metrics['root-agent'] is not failed_metrics


# --- _extract_eval_requests unit tests ---


def _make_judge_graph(node_keys_with_judges=None) -> AgentGraphDefinition:
    """Build a 2-node graph (root -> specialist) with judges optionally configured."""
    if node_keys_with_judges is None:
        node_keys_with_judges = {'root-agent'}

    def _agent_config(key: str) -> AIAgentConfig:
        jc = (
            JudgeConfiguration(judges=[JudgeConfiguration.Judge(key='j1', sampling_rate=1.0)])
            if key in node_keys_with_judges
            else None
        )
        return AIAgentConfig(
            key=key,
            enabled=True,
            create_tracker=MagicMock(),
            model=ModelConfig(name='gpt-4'),
            provider=ProviderConfig(name='openai'),
            instructions=f'You are {key}.',
            evaluator=Evaluator.noop(),
            judge_configuration=jc,
        )

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
        create_tracker=MagicMock(),
    )


def _stub_message_item(agent_name: str, text: str):
    """Build a stand-in for a MessageOutputItem (just enough for _extract_eval_requests)."""
    from agents.items import MessageOutputItem
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    raw = ResponseOutputMessage(
        id='m',
        content=[ResponseOutputText(annotations=[], text=text, type='output_text')],
        role='assistant',
        status='completed',
        type='message',
    )
    agent = MagicMock()
    agent.name = agent_name
    return MessageOutputItem(agent=agent, raw_item=raw)


def _stub_handoff_item(source_name: str, target_name: str):
    """Build a stand-in for a HandoffOutputItem."""
    from agents.items import HandoffOutputItem

    src = MagicMock()
    src.name = source_name
    tgt = MagicMock()
    tgt.name = target_name
    return HandoffOutputItem(
        agent=src,
        raw_item={'type': 'function_call_output', 'call_id': 'c', 'output': ''},
        source_agent=src,
        target_agent=tgt,
    )


def test_extract_eval_requests_single_node_with_judges():
    graph = _make_judge_graph(node_keys_with_judges={'root-agent'})
    runner = OpenAIAgentGraphRunner(graph, {})
    runner._agent_name_map = {'root_agent': 'root-agent'}

    result = MagicMock()
    result.new_items = [
        _stub_message_item('root_agent', 'final answer'),
    ]
    result.final_output = 'final answer'

    requests = runner._extract_eval_requests(result, 'user input')
    assert len(requests) == 1
    assert requests[0].node_key == 'root-agent'
    assert requests[0].input == 'user input'
    assert requests[0].output == 'final answer'


def test_extract_eval_requests_handoff_chain():
    """Root hands off to specialist; both nodes have judges → two eval requests."""
    graph = _make_judge_graph(node_keys_with_judges={'root-agent', 'specialist-agent'})
    runner = OpenAIAgentGraphRunner(graph, {})
    runner._agent_name_map = {
        'root_agent': 'root-agent',
        'specialist_agent': 'specialist-agent',
    }

    result = MagicMock()
    result.new_items = [
        _stub_message_item('root_agent', 'I will hand this off'),
        _stub_handoff_item('root_agent', 'specialist_agent'),
        _stub_message_item('specialist_agent', 'specialist answer'),
    ]
    result.final_output = 'specialist answer'

    requests = runner._extract_eval_requests(result, 'user input')
    assert len(requests) == 2

    by_key = {r.node_key: r for r in requests}
    assert by_key['root-agent'].input == 'user input'
    assert by_key['root-agent'].output == 'I will hand this off'
    assert by_key['specialist-agent'].input == 'I will hand this off'
    assert by_key['specialist-agent'].output == 'specialist answer'


def test_extract_eval_requests_skips_nodes_without_judges():
    """Only the root has judges configured; specialist must contribute nothing."""
    graph = _make_judge_graph(node_keys_with_judges={'root-agent'})
    runner = OpenAIAgentGraphRunner(graph, {})
    runner._agent_name_map = {
        'root_agent': 'root-agent',
        'specialist_agent': 'specialist-agent',
    }

    result = MagicMock()
    result.new_items = [
        _stub_message_item('root_agent', 'hand off'),
        _stub_handoff_item('root_agent', 'specialist_agent'),
        _stub_message_item('specialist_agent', 'answer'),
    ]
    result.final_output = 'answer'

    requests = runner._extract_eval_requests(result, 'user')
    assert len(requests) == 1
    assert requests[0].node_key == 'root-agent'


def test_extract_eval_requests_empty_when_no_judges():
    graph = _make_judge_graph(node_keys_with_judges=set())
    runner = OpenAIAgentGraphRunner(graph, {})
    runner._agent_name_map = {'root_agent': 'root-agent'}

    result = MagicMock()
    result.new_items = [_stub_message_item('root_agent', 'answer')]
    result.final_output = 'answer'

    assert runner._extract_eval_requests(result, 'user') == []


def test_extract_eval_requests_empty_when_no_items():
    graph = _make_judge_graph()
    runner = OpenAIAgentGraphRunner(graph, {})
    runner._agent_name_map = {}

    result = MagicMock()
    result.new_items = []
    result.final_output = ''

    assert runner._extract_eval_requests(result, 'user') == []


@pytest.mark.asyncio
async def test_run_populates_eval_requests_for_judge_nodes():
    """End-to-end: run() must attach eval_requests to AgentGraphRunnerResult."""
    graph = _make_judge_graph(node_keys_with_judges={'root-agent'})

    mock_result = MagicMock()
    mock_result.final_output = "answer"
    mock_result.context_wrapper.usage.request_usage_entries = []

    # Build a single MessageOutputItem-like new_item.
    item = _stub_message_item('root_agent', 'answer')
    mock_result.new_items = [item]

    mock_agents = MagicMock()
    mock_agents.Runner.run = AsyncMock(return_value=mock_result)
    mock_agents.Agent = MagicMock(return_value=MagicMock())
    mock_agents.Handoff = MagicMock()
    mock_agents.handoff = MagicMock()

    mock_agents_ext = MagicMock()
    mock_agents_ext.RECOMMENDED_PROMPT_PREFIX = '[PREFIX]'

    with patch.dict('sys.modules', {
        'agents': mock_agents,
        'agents.extensions': MagicMock(),
        'agents.extensions.handoff_prompt': mock_agents_ext,
        'agents.tool_context': MagicMock(),
    }):
        runner = OpenAIAgentGraphRunner(graph, {})
        # Force the sanitized agent-name -> LD key mapping so extraction works
        # without re-running the full build path under heavy mocking.
        runner._agent_name_map = {'root_agent': 'root-agent'}
        result = await runner.run("user input")

    assert result.eval_requests is not None
    assert len(result.eval_requests) == 1
    assert result.eval_requests[0].node_key == 'root-agent'


@pytest.mark.asyncio
async def test_run_does_not_populate_eval_requests_without_judges():
    graph = _make_graph()  # default _make_graph has no judges

    mock_result = MagicMock()
    mock_result.final_output = "answer"
    mock_result.context_wrapper.usage.request_usage_entries = []
    mock_result.new_items = [_stub_message_item('root_agent', 'answer')]

    mock_agents = MagicMock()
    mock_agents.Runner.run = AsyncMock(return_value=mock_result)
    mock_agents.Agent = MagicMock(return_value=MagicMock())
    mock_agents.Handoff = MagicMock()
    mock_agents.handoff = MagicMock()

    mock_agents_ext = MagicMock()
    mock_agents_ext.RECOMMENDED_PROMPT_PREFIX = '[PREFIX]'

    with patch.dict('sys.modules', {
        'agents': mock_agents,
        'agents.extensions': MagicMock(),
        'agents.extensions.handoff_prompt': mock_agents_ext,
        'agents.tool_context': MagicMock(),
    }):
        runner = OpenAIAgentGraphRunner(graph, {})
        runner._agent_name_map = {'root_agent': 'root-agent'}
        result = await runner.run("user")

    assert result.eval_requests is None or result.eval_requests == []
