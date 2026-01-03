"""Feature extraction: partial routing state → graph representation.

Converts routing state into graph format for GNN critic.
"""

import torch
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from .gnn import RoutingGraph
from ..diffusion.model import RoutingState
from ..core.routing.grid import Grid
from ..core.routing.netlist import Netlist, Net


@dataclass
class RoutingResource:
    """A routing resource (PIP, wire, or node)."""
    resource_id: int
    x: int
    y: int
    layer: int
    resource_type: str  # "pip", "wire", "node"
    capacity: int = 1


class RoutingGraphBuilder:
    """Builds graph representation from partial routing state.

    Node features encode:
    - Resource type (PIP, wire, node)
    - Position (x, y, layer)
    - Current usage / capacity
    - Is part of routed net

    Edge features encode:
    - Connection type
    - Distance
    - Direction
    """

    def __init__(
        self,
        grid: Grid,
        node_dim: int = 64,
        edge_dim: int = 32
    ):
        self.grid = grid
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.width, self.height = grid.get_size()

    def build_graph(
        self,
        state: RoutingState,
        netlist: Netlist,
        resources: Optional[List[RoutingResource]] = None
    ) -> RoutingGraph:
        """Build routing graph from current state.

        Args:
            state: Current routing state
            netlist: Design netlist
            resources: Optional list of routing resources

        Returns:
            RoutingGraph for critic evaluation
        """
        # Build node features
        node_features, congestion = self._build_node_features(state, resources)

        # Build edges (resource connectivity)
        edge_index, edge_features = self._build_edge_features(resources)

        # Unrouted mask
        unrouted_mask = self._build_unrouted_mask(state, netlist)

        return RoutingGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            congestion=congestion,
            unrouted_mask=unrouted_mask
        )

    def _build_node_features(
        self,
        state: RoutingState,
        resources: Optional[List[RoutingResource]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build node feature matrix and congestion vector."""
        if resources is None:
            # Create default grid of resources
            num_nodes = self.width * self.height
            resources = [
                RoutingResource(
                    resource_id=i,
                    x=i % self.width,
                    y=i // self.width,
                    layer=0,
                    resource_type="node"
                )
                for i in range(num_nodes)
            ]

        num_nodes = len(resources)
        features = torch.zeros(num_nodes, self.node_dim)
        congestion = torch.zeros(num_nodes)

        # Compute congestion from routing state
        usage_map = self._compute_usage(state)

        for i, res in enumerate(resources):
            # Position features (normalized)
            features[i, 0] = res.x / max(self.width, 1)
            features[i, 1] = res.y / max(self.height, 1)
            features[i, 2] = res.layer / 10.0  # Assume max 10 layers

            # Type encoding
            type_idx = {"pip": 3, "wire": 4, "node": 5}.get(res.resource_type, 5)
            features[i, type_idx] = 1.0

            # Capacity
            features[i, 6] = res.capacity / 10.0

            # Usage / congestion
            key = (res.x, res.y, res.layer)
            usage = usage_map.get(key, 0)
            congestion[i] = usage / max(res.capacity, 1)
            features[i, 7] = congestion[i]

        return features, congestion

    def _compute_usage(self, state: RoutingState) -> Dict[Tuple[int, int, int], int]:
        """Compute resource usage from routing state."""
        usage = {}

        # Count usage from congestion map if available
        if state.congestion_map is not None:
            cmap = state.congestion_map
            if cmap.dim() == 2:
                H, W = cmap.shape
                for y in range(H):
                    for x in range(W):
                        if cmap[y, x] > 0:
                            usage[(x, y, 0)] = int(cmap[y, x].item())

        return usage

    def _build_edge_features(
        self,
        resources: Optional[List[RoutingResource]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edge index and features.

        Creates edges between adjacent routing resources.
        """
        if resources is None:
            num_nodes = self.width * self.height
            # Grid connectivity
            edges_src = []
            edges_dst = []

            for y in range(self.height):
                for x in range(self.width):
                    idx = y * self.width + x

                    # Right neighbor
                    if x < self.width - 1:
                        edges_src.extend([idx, idx + 1])
                        edges_dst.extend([idx + 1, idx])

                    # Down neighbor
                    if y < self.height - 1:
                        edges_src.extend([idx, idx + self.width])
                        edges_dst.extend([idx + self.width, idx])

            if not edges_src:
                return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, self.edge_dim)

            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            edge_features = torch.zeros(len(edges_src), self.edge_dim)

            # Simple edge features (direction, distance)
            for i in range(len(edges_src)):
                src_x = edges_src[i] % self.width
                src_y = edges_src[i] // self.width
                dst_x = edges_dst[i] % self.width
                dst_y = edges_dst[i] // self.width

                edge_features[i, 0] = (dst_x - src_x) / self.width  # dx
                edge_features[i, 1] = (dst_y - src_y) / self.height  # dy

            return edge_index, edge_features

        # Build from explicit resources
        # ... (simplified for now)
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, self.edge_dim)

    def _build_unrouted_mask(
        self,
        state: RoutingState,
        netlist: Netlist
    ) -> torch.Tensor:
        """Build mask indicating which nets are unrouted."""
        num_nets = len(netlist.nets)
        mask = torch.ones(num_nets, dtype=torch.bool)

        for i, net in enumerate(netlist.nets):
            if net.net_id in state.routed_nets:
                mask[i] = False

        return mask


def build_graph_from_routing_state(
    state: RoutingState,
    grid: Grid,
    netlist: Netlist
) -> RoutingGraph:
    """Convenience function to build graph from routing state."""
    builder = RoutingGraphBuilder(grid)
    return builder.build_graph(state, netlist)


class CongestionFeatureExtractor:
    """Extract congestion-focused features for critic.

    Computes:
    - Global congestion metrics (max, mean, hotspots)
    - Local congestion patterns
    - Critical region identification
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        self.width, self.height = grid.get_size()

    def extract_features(
        self,
        congestion_map: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Extract congestion features."""
        if congestion_map is None:
            return torch.zeros(10)  # 10 congestion features

        features = torch.zeros(10)

        # Global stats
        features[0] = congestion_map.mean()
        features[1] = congestion_map.max()
        features[2] = congestion_map.std()

        # Fraction of congested resources
        features[3] = (congestion_map > 0.5).float().mean()
        features[4] = (congestion_map > 0.8).float().mean()
        features[5] = (congestion_map > 1.0).float().mean()  # Overutilized

        # Spatial distribution
        if congestion_map.dim() >= 2:
            H, W = congestion_map.shape[-2:]
            # Center vs edge congestion
            center = congestion_map[..., H//4:3*H//4, W//4:3*W//4].mean()
            edge = congestion_map.mean() - center
            features[6] = center
            features[7] = edge

        return features
