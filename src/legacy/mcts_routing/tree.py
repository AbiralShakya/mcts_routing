"""MCTS tree data structure (NO latent merging)."""

from typing import Dict, List, Optional
import torch

from .node import MCTSNode


class MCTSTree:
    """MCTS tree structure.
    
    Note: NO latent merging (unless value equivalence proven).
    Tree structure, not DAG.
    """
    
    def __init__(self, root: MCTSNode):
        self.root = root
        self.nodes: Dict[int, MCTSNode] = {id(root): root}
        self.children_map: Dict[int, List[int]] = {id(root): []}
    
    def add_node(self, parent: MCTSNode, child: MCTSNode) -> None:
        """Add a child node to the tree.
        
        Args:
            parent: Parent node
            child: Child node to add
        """
        parent_id = id(parent)
        child_id = id(child)
        
        # Store child
        self.nodes[child_id] = child
        
        # Add to parent's children
        if parent_id not in self.children_map:
            self.children_map[parent_id] = []
        self.children_map[parent_id].append(child_id)
    
    def get_children(self, node: MCTSNode) -> List[MCTSNode]:
        """Get children of a node."""
        node_id = id(node)
        child_ids = self.children_map.get(node_id, [])
        return [self.nodes[cid] for cid in child_ids]
    
    def get_node_count(self) -> int:
        """Get total number of nodes in tree."""
        return len(self.nodes)
    
    def is_leaf(self, node: MCTSNode) -> bool:
        """Check if node is a leaf."""
        node_id = id(node)
        return len(self.children_map.get(node_id, [])) == 0

