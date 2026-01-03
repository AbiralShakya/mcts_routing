"""GNN-based routing critic.

Predicts routing success probability from partial routing state.

Input:
- Partial routing state (which nets routed, which PIPs used)
- Current congestion map
- Remaining unrouted nets
- Timing slack estimates

Output:
- V(s) ≈ E[final routing score | s]

This is NOT a heuristic:
- Trained on (partial routing → final nextpnr score) pairs
- Learns global impossibility signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class RoutingGraph:
    """Graph representation of partial routing state.

    Nodes: Routing resources (CLBs, PIPs, wires)
    Edges: Connectivity between resources
    """
    node_features: torch.Tensor    # [N, node_dim]
    edge_index: torch.Tensor       # [2, E]
    edge_features: torch.Tensor    # [E, edge_dim]
    congestion: torch.Tensor       # [N] usage per resource
    unrouted_mask: torch.Tensor    # [num_nets] which nets unrouted
    batch: Optional[torch.Tensor] = None


class CongestionAwareMP(nn.Module):
    """Message passing layer that propagates congestion info.

    Key insight: Congestion in one region affects routability globally.
    This layer spreads congestion signals through the routing graph.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()

        # Message: combine source, dest, edge, and congestion
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + 1, hidden_dim),  # +1 for congestion
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        # Update: combine node with aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        congestion: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with congestion-aware message passing."""
        src, dst = edge_index

        # Include congestion in messages
        src_features = x[src]
        dst_features = x[dst]
        src_cong = congestion[src].unsqueeze(-1)

        messages = self.message_mlp(
            torch.cat([src_features, dst_features, edge_attr, src_cong], dim=-1)
        )

        # Aggregate
        aggregated = torch.zeros_like(x)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Update
        x_new = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        return x_new


class RoutingCritic(nn.Module):
    """GNN critic for routing state evaluation.

    Predicts:
    - V(s) ∈ [0, 1]: expected final routing quality
    - 1.0 = definitely routable with good quality
    - 0.0 = definitely will fail

    Architecture:
    1. Encode routing state as graph
    2. Message passing with congestion awareness
    3. Global pooling → value prediction
    """

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Input projections
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            CongestionAwareMP(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Unrouted net encoder
        self.unrouted_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Fraction unrouted
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Readout: graph embedding → value
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for graph + unrouted
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, graph: RoutingGraph) -> torch.Tensor:
        """Predict routing success probability.

        Args:
            graph: RoutingGraph with partial routing state

        Returns:
            Value in [0, 1]
        """
        x = self.node_encoder(graph.node_features)
        edge_attr = self.edge_encoder(graph.edge_features)

        # Message passing with congestion
        for mp, ln in zip(self.mp_layers, self.layer_norms):
            x_new = mp(x, graph.edge_index, edge_attr, graph.congestion)
            x = ln(x + self.dropout(x_new))

        # Global pooling
        if graph.batch is not None:
            num_graphs = graph.batch.max().item() + 1
            graph_embed = torch.zeros(num_graphs, x.size(-1), device=x.device)
            graph_embed.scatter_add_(0, graph.batch.unsqueeze(-1).expand_as(x), x)
            counts = torch.bincount(graph.batch, minlength=num_graphs).float()
            graph_embed = graph_embed / counts.unsqueeze(-1).clamp(min=1)
        else:
            graph_embed = x.mean(dim=0, keepdim=True)

        # Encode unrouted fraction
        unrouted_frac = graph.unrouted_mask.float().mean().unsqueeze(0).unsqueeze(0)
        unrouted_embed = self.unrouted_encoder(unrouted_frac)

        # Combine and predict
        combined = torch.cat([graph_embed, unrouted_embed], dim=-1)
        return self.readout(combined).squeeze(-1)

    def predict_routability(
        self,
        graph: RoutingGraph,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, bool]:
        """Predict with pruning decision.

        Args:
            graph: Routing state
            threshold: Pruning threshold

        Returns:
            (score, should_prune)
        """
        score = self.forward(graph)
        should_prune = score.item() < threshold
        return score, should_prune


class HierarchicalRoutingCritic(nn.Module):
    """Critic that operates at multiple levels.

    Level 1: Global congestion (coarse tiles)
    Level 2: Local routing channels
    Level 3: Individual net routability

    Combines all levels for final prediction.
    """

    def __init__(self, hidden_dim: int = 128, **kwargs):
        super().__init__()

        # Global critic (coarse grid)
        self.global_critic = RoutingCritic(
            node_dim=32, hidden_dim=hidden_dim // 2, num_layers=2, **kwargs
        )

        # Local critic (fine grid)
        self.local_critic = RoutingCritic(
            node_dim=64, hidden_dim=hidden_dim, num_layers=4, **kwargs
        )

        # Combine levels
        self.combiner = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        global_graph: RoutingGraph,
        local_graph: RoutingGraph
    ) -> torch.Tensor:
        """Multi-level prediction."""
        global_score = self.global_critic(global_graph)
        local_score = self.local_critic(local_graph)

        combined = torch.stack([global_score, local_score], dim=-1)
        return self.combiner(combined).squeeze(-1)


class TimingAwareCritic(RoutingCritic):
    """Critic that considers timing constraints.

    Adds timing slack as an input feature.
    Critical paths get special attention.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Timing encoder
        self.timing_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_dim),  # (slack, criticality)
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(
        self,
        graph: RoutingGraph,
        timing_info: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward with timing awareness."""
        x = self.node_encoder(graph.node_features)
        edge_attr = self.edge_encoder(graph.edge_features)

        # Add timing if available
        if timing_info is not None:
            timing_embed = self.timing_encoder(timing_info)
            x = x + timing_embed

        # Standard message passing
        for mp, ln in zip(self.mp_layers, self.layer_norms):
            x_new = mp(x, graph.edge_index, edge_attr, graph.congestion)
            x = ln(x + self.dropout(x_new))

        # Pooling and readout
        graph_embed = x.mean(dim=0, keepdim=True)
        unrouted_frac = graph.unrouted_mask.float().mean().unsqueeze(0).unsqueeze(0)
        unrouted_embed = self.unrouted_encoder(unrouted_frac)

        combined = torch.cat([graph_embed, unrouted_embed], dim=-1)
        return self.readout(combined).squeeze(-1)
