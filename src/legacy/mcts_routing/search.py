"""Main MCTS search algorithm."""

import torch
from typing import Optional, Callable
import random

from .node import MCTSNode
from .tree import MCTSTree
from .ucb import select_best_child, ucb_score
from .progressive_widening import should_expand, max_children_count
from .semantic_branching import should_branch_semantically
from .backpropagation import backpropagate
from .value_normalization import normalize_q_value
from ..diffusion.schedule import NoiseSchedule
from ..diffusion.sampler import reverse_step_ddpm, reverse_step_ddim
from ..diffusion.schedule import DDPMSchedule, DDIMSchedule


class MCTSSearch:
    """MCTS search for diffusion-guided routing."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        schedule: NoiseSchedule,
        decoder: Callable,
        solver: Callable,
        reward_fn: Callable,
        ucb_c: float = 1.41,
        k: float = 2.0,
        alpha: float = 0.5,
        max_depth: int = 50,
        semantic_branching: bool = True,
        value_normalization: bool = True,
        value_bootstrapping: bool = True
    ):
        self.model = model
        self.schedule = schedule
        self.decoder = decoder
        self.solver = solver
        self.reward_fn = reward_fn
        self.ucb_c = ucb_c
        self.k = k
        self.alpha = alpha
        self.max_depth = max_depth
        self.semantic_branching = semantic_branching
        self.value_normalization = value_normalization
        self.value_bootstrapping = value_bootstrapping
    
    def search(
        self,
        root_latent: torch.Tensor,
        T: int,
        num_simulations: int = 1000,
        seed: Optional[int] = None
    ) -> MCTSNode:
        """Run MCTS search.
        
        Args:
            root_latent: Root latent x_T [C, H, W]
            T: Total timesteps
            num_simulations: Number of MCTS simulations
            seed: Random seed
        
        Returns:
            Best node found
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Create root node
        root = MCTSNode(latent=root_latent, timestep=T)
        tree = MCTSTree(root)
        
        # Run simulations
        for _ in range(num_simulations):
            # Selection + Expansion
            node = self._tree_policy(root, tree)
            
            # Rollout
            reward = self._rollout(node, T)
            
            # Backpropagation
            backpropagate(
                node,
                reward,
                tree,
                schedule=self.schedule,
                T=T,
                time_aware=self.value_normalization
            )
        
        # Select best child
        best_child = self._select_best(root, tree)
        return best_child if best_child else root
    
    def _tree_policy(self, root: MCTSNode, tree: MCTSTree) -> MCTSNode:
        """Tree policy: selection + expansion."""
        node = root
        
        # Selection: traverse to leaf
        while not tree.is_leaf(node) and not node.is_terminal():
            children = tree.get_children(node)
            if children:
                # Select best child
                node = select_best_child(node, children, self.ucb_c)
            else:
                break
        
        # Expansion: add new child if allowed
        children = tree.get_children(node)
        if not node.is_terminal() and should_expand(node, len(children), self.k, self.alpha):
            child = self._expand(node)
            tree.add_node(node, child)
            node = child
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by sampling noise and denoising."""
        # Sample noise
        noise = torch.randn_like(node.latent)
        
        # Predict noise with model
        with torch.no_grad():
            t_tensor = torch.tensor([node.timestep])
            predicted_noise = self.model(node.latent.unsqueeze(0), t_tensor)
            predicted_noise = predicted_noise.squeeze(0)
        
        # Reverse step
        if isinstance(self.schedule, DDPMSchedule):
            next_latent = reverse_step_ddpm(
                node.latent.unsqueeze(0),
                predicted_noise.unsqueeze(0),
                t_tensor,
                self.schedule
            ).squeeze(0)
        elif isinstance(self.schedule, DDIMSchedule):
            t_prev_tensor = torch.tensor([max(0, node.timestep - 1)])
            next_latent = reverse_step_ddim(
                node.latent.unsqueeze(0),
                predicted_noise.unsqueeze(0),
                t_tensor,
                t_prev_tensor,
                self.schedule,
                eta=0.0
            ).squeeze(0)
        else:
            # Fallback
            next_latent = node.latent - 0.01 * predicted_noise
        
        # Create child node
        child = MCTSNode(
            latent=next_latent,
            timestep=max(0, node.timestep - 1)
        )
        
        return child
    
    def _rollout(self, node: MCTSNode, T: int) -> float:
        """Rollout from node to terminal and compute reward."""
        # If terminal, decode and compute reward
        if node.is_terminal():
            # Decode to potentials
            # Note: Need grid and netlist - should be passed to search
            # For now, return 0.0 as placeholder
            return 0.0
        
        # Otherwise, use proxy reward if bootstrapping enabled
        if self.value_bootstrapping:
            from .value_bootstrapping import compute_proxy_reward
            # Note: Need decoder, grid, netlist
            return compute_proxy_reward(node.latent, node.timestep, None, None)
        
        return 0.0
    
    def _select_best(self, root: MCTSNode, tree: MCTSTree) -> Optional[MCTSNode]:
        """Select best child of root."""
        children = tree.get_children(root)
        if not children:
            return None
        
        # Select child with highest Q-value
        best = max(children, key=lambda c: c.q_value)
        return best

