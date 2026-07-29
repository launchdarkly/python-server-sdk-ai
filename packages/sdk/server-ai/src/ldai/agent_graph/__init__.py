"""Graph implementation for managing AI agent graphs."""

from typing import Any, Callable, Dict, List, Optional, Set

from ldclient import Context

from ldai.models import AIAgentConfig, AIAgentGraphConfig, Edge
from ldai.tracker import AIGraphTracker


class AgentGraphNode:
    """
    Node in an agent graph.
    """

    def __init__(
        self,
        key: str,
        config: AIAgentConfig,
        children: List[Edge],
    ):
        self._key = key
        self._config = config
        self._children = children

    def get_key(self) -> str:
        """Get the key of the node."""
        return self._key

    def get_config(self) -> AIAgentConfig:
        """Get the config of the node."""
        return self._config

    def is_terminal(self) -> bool:
        """Check if the node is a terminal node."""
        return len(self._children) == 0

    def get_edges(self) -> List[Edge]:
        """Get the edges of the node."""
        return self._children


class AgentGraphDefinition:
    """
    Graph implementation for managing AI agent graphs.
    """
    enabled: bool

    def __init__(
        self,
        agent_graph: AIAgentGraphConfig,
        nodes: Dict[str, AgentGraphNode],
        context: Context,
        enabled: bool,
        create_tracker: Callable[[], AIGraphTracker],
    ):
        self._agent_graph = agent_graph
        self._context = context
        self._nodes = nodes
        self.enabled = enabled
        self.create_tracker = create_tracker

    def is_enabled(self) -> bool:
        """Check if the graph is enabled."""
        return self.enabled

    @staticmethod
    def build_nodes(
        agent_graph: AIAgentGraphConfig,
        graph_nodes: Dict[str, AIAgentConfig],
    ) -> Dict[str, "AgentGraphNode"]:
        """Build the nodes of the graph into AgentGraphNode objects."""
        nodes = {
            agent_graph.root_config_key: AgentGraphNode(
                agent_graph.root_config_key,
                graph_nodes[agent_graph.root_config_key],
                [
                    edge
                    for edge in agent_graph.edges
                    if edge.source_config == agent_graph.root_config_key
                ],
            ),
        }

        for edge in agent_graph.edges:
            nodes[edge.target_config] = AgentGraphNode(
                edge.target_config,
                graph_nodes[edge.target_config],
                [e for e in agent_graph.edges if e.source_config == edge.target_config],
            )

        return nodes

    def get_node(self, key: str) -> Optional[AgentGraphNode]:
        """Get a node by its key."""
        return self._nodes.get(key)

    def _get_child_edges(self, config_key: str) -> List[Edge]:
        """Get the child edges of the given config."""
        return [
            edge for edge in self._agent_graph.edges if edge.source_config == config_key
        ]

    def get_child_nodes(self, node_key: str) -> List[AgentGraphNode]:
        """Get the child nodes of the given node key as AgentGraphNode objects."""
        nodes: List[AgentGraphNode] = []
        for edge in self._agent_graph.edges:
            if edge.source_config == node_key:
                node = self.get_node(edge.target_config)
                if node is not None:
                    nodes.append(node)
        return nodes

    def get_parent_nodes(self, node_key: str) -> List[AgentGraphNode]:
        """Get the parent nodes of the given node key as AgentGraphNode objects."""
        nodes: List[AgentGraphNode] = []
        for edge in self._agent_graph.edges:
            if edge.target_config == node_key:
                node = self.get_node(edge.source_config)
                if node is not None:
                    nodes.append(node)
        return nodes

    def _reachable_and_discovery(
        self, root_key: str
    ) -> tuple[Set[str], List[str]]:
        """Return reachable node keys and BFS discovery order from root."""
        reachable: Set[str] = set()
        order: List[str] = []
        queue: List[str] = [root_key]
        reachable.add(root_key)
        order.append(root_key)
        while queue:
            key = queue.pop(0)
            node = self.get_node(key)
            if node is None:
                continue
            for edge in node.get_edges():
                if (
                    self.get_node(edge.target_config) is not None
                    and edge.target_config not in reachable
                ):
                    reachable.add(edge.target_config)
                    order.append(edge.target_config)
                    queue.append(edge.target_config)
        return reachable, order

    def terminal_nodes(self) -> List[AgentGraphNode]:
        """Get the terminal nodes of the graph, meaning any nodes without children."""
        return [
            node
            for node in self._nodes.values()
            if len(self.get_child_nodes(node.get_key())) == 0
        ]

    def root(self) -> Optional[AgentGraphNode]:
        """Get the root node of the graph."""
        return self._nodes.get(self._agent_graph.root_config_key)

    def traverse(
        self,
        fn: Callable[["AgentGraphNode", Dict[str, Any]], Any],
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Visit each reachable node in topological (predecessors-first) order.

        The root is visited first. A node is visited only after all of its
        reachable predecessors. When multiple nodes are ready, discovery order
        (BFS from root following declared edge order) breaks ties. Cycles are
        broken at the lowest remaining in-degree, also tie-broken by discovery
        order.

        Each node's ``fn`` receives a fresh context containing the caller-provided
        initial context plus the returned results of exactly that node's reachable
        predecessors — not unrelated parallel-branch nodes.
        """
        if execution_context is None:
            execution_context = {}

        root_node = self.root()
        if root_node is None:
            return

        reachable, order = self._reachable_and_discovery(root_node.get_key())

        indeg = {k: 0 for k in reachable}
        for k in reachable:
            node = self.get_node(k)
            if node is None:
                continue
            for edge in node.get_edges():
                if edge.target_config in reachable:
                    indeg[edge.target_config] += 1
        indeg[root_node.get_key()] = 0  # force root

        visited: Set[str] = set()
        results: Dict[str, Any] = {}
        ancestors: Dict[str, Set[str]] = {}

        def scoped(deps: Set[str]) -> Dict[str, Any]:
            c = dict(execution_context)
            for dep_key in deps:
                c[dep_key] = results[dep_key]
            return c

        while len(visited) < len(reachable):
            nxt = next(
                (k for k in order if k not in visited and indeg[k] == 0), None
            )
            if nxt is None:  # cycle break
                nxt = min(
                    (k for k in order if k not in visited), key=lambda k: indeg[k]
                )
            visited.add(nxt)
            anc: Set[str] = set()
            for parent in self.get_parent_nodes(nxt):
                pk = parent.get_key()
                if pk not in visited:
                    continue
                anc.add(pk)
                anc |= ancestors.get(pk, set())
            ancestors[nxt] = anc
            node = self.get_node(nxt)
            assert node is not None
            results[nxt] = fn(node, scoped(anc))
            for edge in node.get_edges():
                if edge.target_config in reachable:
                    indeg[edge.target_config] -= 1
        return results.get(self._agent_graph.root_config_key)

    def reverse_traverse(
        self,
        fn: Callable[["AgentGraphNode", Dict[str, Any]], Any],
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Visit each reachable node in reverse topological (descendants-first) order.

        The root is visited last. A node is visited only after all of its
        reachable descendants. When multiple nodes are ready, discovery order
        (BFS from root following declared edge order) breaks ties. Cycles among
        non-root nodes are broken at the lowest remaining out-degree, also
        tie-broken by discovery order. Graphs with no terminals (pure cycles)
        still visit every reachable node, root last.

        Each node's ``fn`` receives a fresh context containing the caller-provided
        initial context plus the returned results of exactly that node's reachable
        descendants — not unrelated parallel-branch nodes.
        """
        if execution_context is None:
            execution_context = {}

        root_node = self.root()
        if root_node is None:
            return

        root_key = self._agent_graph.root_config_key
        reachable, order = self._reachable_and_discovery(root_key)

        outdeg: Dict[str, int] = {}
        for k in reachable:
            node = self.get_node(k)
            assert node is not None
            outdeg[k] = sum(
                1 for e in node.get_edges() if e.target_config in reachable
            )

        visited: Set[str] = set()
        results: Dict[str, Any] = {}
        descendants: Dict[str, Set[str]] = {}

        def scoped(deps: Set[str]) -> Dict[str, Any]:
            c = dict(execution_context)
            for dep_key in deps:
                c[dep_key] = results[dep_key]
            return c

        def non_root_remaining() -> bool:
            return any(k != root_key and k not in visited for k in reachable)

        while non_root_remaining():
            nxt = next(
                (
                    k
                    for k in order
                    if k != root_key and k not in visited and outdeg[k] == 0
                ),
                None,
            )
            if nxt is None:  # cycle break
                nxt = min(
                    (k for k in order if k != root_key and k not in visited),
                    key=lambda k: outdeg[k],
                )
            visited.add(nxt)
            desc: Set[str] = set()
            node = self.get_node(nxt)
            assert node is not None
            for edge in node.get_edges():
                ck = edge.target_config
                if ck not in reachable or ck not in visited:
                    continue
                desc.add(ck)
                desc |= descendants.get(ck, set())
            descendants[nxt] = desc
            results[nxt] = fn(node, scoped(desc))
            for parent in self.get_parent_nodes(nxt):
                pk = parent.get_key()
                if pk != root_key and pk in reachable:
                    outdeg[pk] -= 1

        # root last, once; root depends on every reachable non-root node
        root_deps = {k for k in reachable if k != root_key}
        visited.add(root_key)
        results[root_key] = fn(root_node, scoped(root_deps))
        return results.get(root_key)
