"""MCTS tree structure for routing."""

from typing import Optional, List
from .node import RoutingNode


class MCTSTree:
    """MCTS tree for routing search.

    Over time, the tree becomes a memory of successful routing patterns.
    Good routing decisions reinforce early denoising steps.
    """

    def __init__(self, root: RoutingNode):
        self.root = root

    def get_path_to_root(self, node: RoutingNode) -> List[RoutingNode]:
        """Get path from node to root."""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def get_best_terminal(self) -> Optional[RoutingNode]:
        """Get best terminal node by Q-value."""
        best_node = None
        best_q = float('-inf')

        def visit(node: RoutingNode):
            nonlocal best_node, best_q
            if node.is_terminal() and not node.pruned:
                if node.Q > best_q:
                    best_q = node.Q
                    best_node = node
            for child in node.children:
                visit(child)

        visit(self.root)
        return best_node

    def count_nodes(self) -> int:
        """Count total nodes."""
        count = 0
        def visit(node):
            nonlocal count
            count += 1
            for child in node.children:
                visit(child)
        visit(self.root)
        return count

    def count_pruned(self) -> int:
        """Count pruned nodes."""
        count = 0
        def visit(node):
            nonlocal count
            if node.pruned:
                count += 1
            for child in node.children:
                visit(child)
        visit(self.root)
        return count

    def count_terminal(self) -> int:
        """Count terminal (successful) nodes."""
        count = 0
        def visit(node):
            nonlocal count
            if node.is_terminal() and not node.pruned:
                count += 1
            for child in node.children:
                visit(child)
        visit(self.root)
        return count

    def get_statistics(self) -> dict:
        """Get tree statistics."""
        return {
            "total_nodes": self.count_nodes(),
            "pruned_nodes": self.count_pruned(),
            "terminal_nodes": self.count_terminal(),
            "root_visits": self.root.visit_count,
            "root_Q": self.root.Q
        }
