"""Graph implementation for managing AI agent graphs."""

from typing import Any, Callable, Dict, List, Optional, Set
from ldai.models import AIAgentGraph, AIAgentConfig, AIAgentGraphEdge
from ldclient import Context


class AgentGraphNode:
    """
    Node in an agent graph.
    """

    default_false = AIAgentConfig(key="", enabled=False)

    def __init__(
        self,
        key: str,
        config: AIAgentConfig,
        children: List[AIAgentGraphEdge],
        parent_graph: "AgentGraph",
    ):
        self._key = key
        self._config = config
        self._children = children
        self._parent_graph = parent_graph

    def get_key(self) -> str:
        """Get the key of the node."""
        return self._key

    def get_config(self) -> AIAgentConfig:
        """Get the config of the node."""
        return self._config

    def get_edges(self) -> List[AIAgentGraphEdge]:
        """Get the edges of the node."""
        return self._children

    def get_child_nodes(self) -> List["AgentGraphNode"]:
        """Get the child nodes of the node as AgentGraphNode objects."""
        return [
            self._parent_graph.get_node(edge.targetConfig) for edge in self._children
        ]

    def is_terminal(self) -> bool:
        """Check if the node is a terminal node."""
        return len(self._children) == 0 

    def get_parent_nodes(self) -> List["AgentGraphNode"]:
        """Get the parent nodes of the node as AgentGraphNode objects."""
        return [
            self._parent_graph.get_node(edge.sourceConfig)
            for edge in self._parent_graph._get_parent_edges(self._key)
        ]

    def traverse(
        self, fn: Callable[["AgentGraphNode", Dict[str, Any]], None], execution_context: Dict[str, Any] = {}, visited: Optional[Set[str]] = None
    ) -> None:
        """Traverse the graph downwardly from this node, calling fn on each node."""
        if visited is None:
            visited = set()

        # Avoid cycles by tracking visited nodes
        if self._key in visited:
            return

        visited.add(self._key)
        fn(self, execution_context)

        for child in self._children:
            node = self._parent_graph.get_node(child.targetConfig)
            if node is not None:
                node.traverse(fn, execution_context, visited)

    def reverse_traverse(
        self,
        fn: Callable[["AgentGraphNode", Dict[str, Any]], None],
        execution_context: Dict[str, Any] = {},
        visited: Optional[Set[str]] = None,
    ) -> None:
        """Reverse traverse the graph upwardly from this node, calling fn on each node."""
        if visited is None:
            visited = set()

        # Avoid cycles by tracking visited nodes
        if self._key in visited:
            return

        visited.add(self._key)
        fn(self, execution_context)

        for parent in self._parent_graph._get_parent_edges(self._key):
            node = self._parent_graph.get_node(parent.sourceConfig)
            if node is not None:
                node.reverse_traverse(fn, execution_context, visited)


class AgentGraph:
    """
    Graph implementation for managing AI agent graphs.
    """

    default_false = AIAgentConfig(key="", enabled=False)

    def __init__(
        self,
        agent_graph: AIAgentGraph,
        context: Context,
        get_agent: Callable[[str, Context, dict], AIAgentConfig],
    ):
        self._agent_graph = agent_graph
        self._context = context
        self._get_agent = get_agent
        self._nodes = self._build_nodes()

    def _build_nodes(self) -> Dict[str, AgentGraphNode]:
        """Build the nodes of the graph into AgentGraphNode objects."""
        nodes = {
            self._agent_graph.rootConfigKey: AgentGraphNode(
                self._agent_graph.rootConfigKey,
                self._get_agent(
                    self._agent_graph.rootConfigKey, self._context, self.default_false
                ),
                self._get_child_edges(self._agent_graph.rootConfigKey),
                self,
            ),
        }

        for edge in self._agent_graph.edges:
            nodes[edge.targetConfig] = AgentGraphNode(
                edge.targetConfig,
                self._get_agent(edge.targetConfig, self._context, self.default_false),
                self._get_child_edges(edge.targetConfig),
                self,
            )

        return nodes

    def _get_child_edges(self, config_key: str) -> List[AIAgentGraphEdge]:
        """Get the child edges of the given config."""
        return [
            edge
            for edge in self._agent_graph.edges
            if edge.sourceConfig == config_key
        ]

    def _get_parent_edges(self, config_key: str) -> List[AIAgentGraphEdge]:
        """Get the parent edges of the given config."""
        return [
            edge
            for edge in self._agent_graph.edges
            if edge.targetConfig == config_key
        ]

    def _collect_nodes(
        self,
        node: AgentGraphNode,
        node_depths: Dict[str, int],
        nodes_by_depth: Dict[int, List[AgentGraphNode]],
        visited: Set[str],
    ) -> None:
        """Collect all reachable nodes from the given node and group them by depth."""
        node_key = node.get_key()
        if node_key in visited:
            return
        visited.add(node_key)

        node_depth = node_depths.get(node_key, 0)
        if node_depth not in nodes_by_depth:
            nodes_by_depth[node_depth] = []
        nodes_by_depth[node_depth].append(node)

        for child in node.get_child_nodes():
            self._collect_nodes(child, node_depths, nodes_by_depth, visited)

    def terminal_nodes(self) -> List[AgentGraphNode]:
        """Get the terminal nodes of the graph, meaning any nodes without children."""
        return [
            node for node in self._nodes.values() if len(node.get_child_nodes()) == 0
        ]

    def root(self) -> AgentGraphNode | None:
        """Get the root node of the graph."""
        config = self._get_agent(
            self._agent_graph.rootConfigKey, self._context, self.default_false
        )

        if config.enabled is False:
            return None

        children = [
            edge
            for edge in self._agent_graph.edges
            if edge.sourceConfig == self._agent_graph.rootConfigKey
        ]

        node = AgentGraphNode(self._agent_graph.rootConfigKey, config, children, self)

        return node

    def traverse(self, fn: Callable[["AgentGraphNode", Dict[str, Any]], None], execution_context: Dict[str, Any] = {}) -> None:
        """Traverse from the root down to terminal nodes, visiting nodes in order of depth.
        Nodes with the longest paths from the root (deepest nodes) will always be visited last."""
        root_node = self.root()
        if root_node is None:
            return

        node_depths: Dict[str, int] = {root_node.get_key(): 0}
        current_level: List[AgentGraphNode] = [root_node]
        depth = 0
        max_depth_limit = 10  # Infinite loop protection limit

        while current_level and depth < max_depth_limit:
            next_level: List[AgentGraphNode] = []
            depth += 1

            for node in current_level:
                for child in node.get_child_nodes():
                    child_key = child.get_key()
                    # Defer this child to the next level if it's at a longer path
                    if child_key not in node_depths or (
                        depth > node_depths[child_key] and depth < max_depth_limit
                    ):
                        node_depths[child_key] = depth
                        next_level.append(child)

            current_level = next_level

        # Group all nodes by depth
        nodes_by_depth: Dict[int, List[AgentGraphNode]] = {}
        visited: Set[str] = set()

        self._collect_nodes(root_node, node_depths, nodes_by_depth, visited)
        # Execute the lambda at this level for the nodes at this depth
        for depth_level in sorted(nodes_by_depth.keys()):
            for node in nodes_by_depth[depth_level]:
                execution_context[node.get_key()] = fn(node, execution_context)

        return execution_context[self._agent_graph.rootConfigKey]

    def reverse_traverse(self, fn: Callable[["AgentGraphNode", Dict[str, Any]], Any], execution_context: Dict[str, Any] = {}) -> None:
        """Traverse from terminal nodes up to the root, visiting nodes level by level.
        The root node will always be visited last, even if multiple paths converge at it."""
        terminal_nodes = self.terminal_nodes()
        if not terminal_nodes:
            return

        visited: Set[str] = set()
        current_level: List[AgentGraphNode] = terminal_nodes
        root_key = self._agent_graph.rootConfigKey
        root_node_seen = False

        while current_level:
            next_level: List[AgentGraphNode] = []

            for node in current_level:
                node_key = node.get_key()
                if node_key in visited:
                    continue

                visited.add(node_key)
                # Skip the root node if we reach a terminus, it will be visited last
                if node_key == root_key:
                    root_node_seen = True
                    continue

                execution_context[node_key] = fn(node, execution_context)
                
                for parent in node.get_parent_nodes():
                    parent_key = parent.get_key()
                    if parent_key not in visited:
                        next_level.append(parent)

            current_level = next_level

        # If we saw the root node, append it at the end as it'll always be the last node in a
        # reverse traversal (this should always happen, non-contiguous graphs are invalid)
        if root_node_seen:
            root_node = self.root()
            if root_node is not None:
                execution_context[root_node.get_key()] = fn(root_node, execution_context)

        return execution_context[self._agent_graph.rootConfigKey]

    def get_node(self, key: str) -> AgentGraphNode | None:
        """Get a node by its key."""
        return self._nodes.get(key)
