import pytest
from ldclient import Config, Context, LDClient
from ldclient.integrations.test_data import TestData

from ldai import (
    LDAIClient,
    AIAgentGraphConfig,
    AgentGraphDefinition,
    AIAgentConfig,
    Edge,
)


@pytest.fixture
def td() -> TestData:
    td = TestData.data_source()
    # Agent graph with depth of 1
    td.update(
        td.flag("test-agent-graph")
        .variations(
            {
                "root": "customer-support-agent",
                "edges": {
                    "customer-support-agent": [
                        {
                            "key": "personalized-agent",
                            "handoff": {"state": "from-root-to-personalized"},
                        },
                        {"key": "multi-context-agent", "handoff": {}},
                        {"key": "minimal-agent", "handoff": {}},
                    ]
                },
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "test-agent-graph",
                    "version": 1,
                },
            }
        )
        .variation_for_all(0)
    )
    # Agent graph with depth of 3
    td.update(
        td.flag("test-agent-graph-depth-3")
        .variations(
            {
                "root": "customer-support-agent",
                "edges": {
                    "customer-support-agent": [
                        {
                            "key": "personalized-agent",
                            "handoff": {"state": "from-root-to-personalized"},
                        },
                        {
                            "key": "minimal-agent",
                            "handoff": {"state": "from-root-to-minimal"},
                        },
                    ],
                    "personalized-agent": [
                        {"key": "multi-context-agent", "handoff": {}}
                    ],
                    "multi-context-agent": [
                        {
                            "key": "minimal-agent",
                            "handoff": {"state": "from-multi-context-to-minimal"},
                        }
                    ],
                },
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "test-agent-graph-depth-3",
                    "version": 1,
                },
            }
        )
        .variation_for_all(0)
    )

    # Agent graph with disabled agent included - invalid
    td.update(
        td.flag("test-agent-graph-disabled-agent")
        .variations(
            {
                "root": "customer-support-agent",
                "edges": {
                    "customer-support-agent": [{"key": "disabled-agent", "handoff": {}}]
                },
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "test-agent-graph-disabled-agent",
                    "version": 1,
                },
            }
        )
        .variation_for_all(0)
    )

    # Agent graph with no root key - invalid
    td.update(
        td.flag("test-agent-graph-no-root-key")
        .variations(
            {
                "edges": {},
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "test-agent-graph-no-root-key",
                    "version": 1,
                },
            }
        )
        .variation_for_all(0)
    )

    # Single agent with instructions
    td.update(
        td.flag("customer-support-agent")
        .variations(
            {
                "model": {
                    "name": "gpt-4",
                    "parameters": {"temperature": 0.3, "maxTokens": 2048},
                },
                "provider": {"name": "openai"},
                "instructions": "You are a helpful customer support agent for {{company_name}}. Always be polite and professional.",
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "agent-v1",
                    "version": 1,
                    "mode": "agent",
                },
            }
        )
        .variation_for_all(0)
    )

    # Agent with context interpolation
    td.update(
        td.flag("personalized-agent")
        .variations(
            {
                "model": {"name": "claude-3", "parameters": {"temperature": 0.5}},
                "instructions": "Hello {{ldctx.name}}! I am your personal assistant. Your user key is {{ldctx.key}}.",
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "personal-v1",
                    "version": 2,
                    "mode": "agent",
                },
            }
        )
        .variation_for_all(0)
    )

    # Agent with multi-context interpolation
    td.update(
        td.flag("multi-context-agent")
        .variations(
            {
                "model": {"name": "gpt-3.5-turbo"},
                "instructions": "Welcome {{ldctx.user.name}} from {{ldctx.org.name}}! Your organization tier is {{ldctx.org.tier}}.",
                "_ldMeta": {
                    "enabled": True,
                    "variationKey": "multi-v1",
                    "version": 1,
                    "mode": "agent",
                },
            }
        )
        .variation_for_all(0)
    )

    # Disabled agent
    td.update(
        td.flag("disabled-agent")
        .variations(
            {
                "model": {"name": "gpt-4"},
                "instructions": "This agent is disabled.",
                "_ldMeta": {
                    "enabled": False,
                    "variationKey": "disabled-v1",
                    "version": 1,
                    "mode": "agent",
                },
            }
        )
        .variation_for_all(0)
    )

    # Agent with minimal metadata
    td.update(
        td.flag("minimal-agent")
        .variations(
            {
                "instructions": "Minimal agent configuration.",
                "_ldMeta": {"enabled": True},
            }
        )
        .variation_for_all(0)
    )

    return td


@pytest.fixture
def client(td: TestData) -> LDClient:
    config = Config("sdk-key", update_processor_class=td, send_events=False)
    return LDClient(config=config)


@pytest.fixture
def ldai_client(client: LDClient) -> LDAIClient:
    return LDAIClient(client)


def test_agent_graph_method(ldai_client: LDAIClient):
    graph = ldai_client.agent_graph("test-agent-graph", Context.create("user-key"))

    assert graph.enabled is True
    assert graph is not None
    assert graph.root() is not None
    assert graph.root().get_key() == "customer-support-agent"
    assert len(graph.get_child_nodes("customer-support-agent")) == 3
    assert len(graph.get_child_nodes("personalized-agent")) == 0
    assert len(graph.get_child_nodes("multi-context-agent")) == 0
    assert len(graph.get_child_nodes("minimal-agent")) == 0


def test_agent_graph_method_disabled_agent(ldai_client: LDAIClient):
    graph = ldai_client.agent_graph(
        "test-agent-graph-disabled-agent", Context.create("user-key")
    )

    assert graph.enabled is False
    assert graph.root() is None


def test_agent_graph_method_no_root_key(ldai_client: LDAIClient):
    graph = ldai_client.agent_graph(
        "test-agent-graph-no-root-key", Context.create("user-key")
    )

    assert graph.enabled is False
    assert graph.root() is None


def test_agent_graph_build_nodes(ldai_client: LDAIClient):
    graph_config = ldai_client._client.variation(
        "test-agent-graph", Context.create("user-key"), {}
    )

    ai_graph_config = AIAgentGraphConfig(
        key="test-agent-graph",
        root_config_key=graph_config["root"],
        edges=[
            Edge(
                key=edge_key + "-" + edge.get("key", ""),
                source_config=edge_key,
                target_config=edge.get("key", ""),
                handoff=edge.get("handoff", {}),
            )
            for edge_key, edges in graph_config["edges"].items()
            for edge in edges
        ],
    )

    nodes = AgentGraphDefinition.build_nodes(
        ai_graph_config,
        {
            "customer-support-agent": AIAgentConfig(
                key="customer-support-agent", enabled=True
            ),
            "personalized-agent": AIAgentConfig(key="personalized-agent", enabled=True),
            "multi-context-agent": AIAgentConfig(
                key="multi-context-agent", enabled=True
            ),
            "minimal-agent": AIAgentConfig(key="minimal-agent", enabled=True),
        },
    )

    assert nodes["customer-support-agent"] is not None
    assert nodes["personalized-agent"] is not None
    assert nodes["multi-context-agent"] is not None
    assert nodes["minimal-agent"] is not None

    assert len(nodes["customer-support-agent"].get_edges()) == 3
    assert len(nodes["personalized-agent"].get_edges()) == 0
    assert len(nodes["multi-context-agent"].get_edges()) == 0
    assert len(nodes["minimal-agent"].get_edges()) == 0

    assert type(nodes["customer-support-agent"].get_config()) is AIAgentConfig
    assert type(nodes["personalized-agent"].get_config()) is AIAgentConfig
    assert type(nodes["multi-context-agent"].get_config()) is AIAgentConfig
    assert type(nodes["minimal-agent"].get_config()) is AIAgentConfig

    assert type(nodes["customer-support-agent"].get_edges()[0]) is Edge


def test_agent_graph_get_methods(ldai_client: LDAIClient):
    graph = ldai_client.agent_graph("test-agent-graph", Context.create("user-key"))

    assert graph.root() is not None
    assert graph.root().get_key() == "customer-support-agent"
    assert graph.get_node("customer-support-agent") is not None
    assert graph.get_node("personalized-agent") is not None
    assert graph.get_node("multi-context-agent") is not None

    children = graph.get_child_nodes("customer-support-agent")
    assert len(children) == 3
    assert children[0].get_key() == "personalized-agent"
    assert children[1].get_key() == "multi-context-agent"
    assert children[2].get_key() == "minimal-agent"

    parents = graph.get_parent_nodes("personalized-agent")
    assert len(parents) == 1
    assert parents[0].get_key() == "customer-support-agent"

    parents = graph.get_parent_nodes("multi-context-agent")
    assert len(parents) == 1
    assert parents[0].get_key() == "customer-support-agent"

    terminal = graph.terminal_nodes()
    assert len(terminal) == 3
    assert terminal[0].get_key() == "personalized-agent"
    assert terminal[1].get_key() == "multi-context-agent"
    assert terminal[2].get_key() == "minimal-agent"

    assert graph.root().is_terminal() is False
    assert graph.get_node("customer-support-agent").is_terminal() is False
    assert graph.get_node("personalized-agent").is_terminal() is True
    assert graph.get_node("multi-context-agent").is_terminal() is True
    assert graph.get_node("minimal-agent").is_terminal() is True


def test_agent_graph_traverse(ldai_client: LDAIClient):
    graph = ldai_client.agent_graph(
        "test-agent-graph-depth-3", Context.create("user-key")
    )

    context = {}
    order = []

    def handle_traverse(node, context):
        # Asserting that returned values are included in the context
        for previousKey in order:
            assert previousKey in context
            assert context[previousKey] == previousKey + "-test"
        order.append(node.get_key())
        return node.get_key() + "-test"

    graph.traverse(handle_traverse, context)
    # Asserting that we traverse in the expected order
    # This config specifically has nodes connecting from depth 2->3 and root->3 to ensure the root node is visited first
    # and minimal-agent is visited last
    assert order == [
        "customer-support-agent",
        "personalized-agent",
        "multi-context-agent",
        "minimal-agent",
    ]


def test_agent_graph_reverse_traverse(ldai_client: LDAIClient):
    graph = ldai_client.agent_graph(
        "test-agent-graph-depth-3", Context.create("user-key")
    )

    context = {}
    order = []

    def handle_reverse_traverse(node, context):
        # Asserting that returned values are included in the context
        for previousKey in order:
            assert previousKey in context
            assert context[previousKey] == previousKey + "-test"
        order.append(node.get_key())
        return node.get_key() + "-test"

    graph.reverse_traverse(handle_reverse_traverse, context)
    # Asserting that we traverse in the expected order
    # This config specifically has nodes connecting from depth 2->3 and root->3 to ensure the root node is visited last
    assert order == [
        "minimal-agent",
        "multi-context-agent",
        "personalized-agent",
        "customer-support-agent",
    ]


def test_agent_graph_handoff(ldai_client: LDAIClient):
    graph = ldai_client.agent_graph(
        "test-agent-graph-depth-3", Context.create("user-key")
    )

    context = {}

    def handle_traverse(node, context):
        if node.get_key() == "multi-context-agent":
            first_edge = node.get_edges()[0]
            assert first_edge.handoff == {"state": "from-multi-context-to-minimal"}
            assert first_edge.source_config == "multi-context-agent"
            assert first_edge.target_config == "minimal-agent"
            assert first_edge.key == "multi-context-agent-minimal-agent"
        if node.get_key() == "customer-support-agent":
            first_edge = node.get_edges()[0]
            second_edge = node.get_edges()[1]
            assert first_edge.handoff == {"state": "from-root-to-personalized"}
            assert second_edge.handoff == {"state": "from-root-to-minimal"}
            assert first_edge.source_config == "customer-support-agent"
            assert first_edge.target_config == "personalized-agent"
            assert first_edge.key == "customer-support-agent-personalized-agent"
            assert second_edge.source_config == "customer-support-agent"
            assert second_edge.target_config == "minimal-agent"
            assert second_edge.key == "customer-support-agent-minimal-agent"
        return None

    graph.traverse(handle_traverse, context)
