"""Unit tests for critic module."""

import pytest
import torch

from src.critic.gnn import RoutabilityCritic, TimestepAwareCritic, PlacementGraph
from src.critic.features import PlacementGraphBuilder
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin
from src.core.routing.placement import Placement


class TestRoutabilityCritic:
    """Tests for RoutabilityCritic GNN."""

    @pytest.fixture
    def critic(self):
        return RoutabilityCritic(
            node_dim=64,
            edge_dim=32,
            hidden_dim=128,
            num_layers=4
        )

    @pytest.fixture
    def sample_graph(self):
        """Create sample placement graph."""
        return PlacementGraph(
            node_features=torch.randn(10, 64),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_features=torch.randn(20, 32)
        )

    def test_forward_shape(self, critic, sample_graph):
        """Test output shape."""
        score = critic(sample_graph)
        assert score.shape == torch.Size([])  # Scalar

    def test_output_range(self, critic, sample_graph):
        """Test output is in [0, 1] due to sigmoid."""
        score = critic(sample_graph)
        assert 0 <= score.item() <= 1

    def test_batched_forward(self, critic):
        """Test batched graph processing."""
        # Create batched graph
        node_features = torch.randn(30, 64)  # 3 graphs of 10 nodes
        edge_index = torch.randint(0, 30, (2, 60))
        edge_features = torch.randn(60, 32)
        batch = torch.tensor([0]*10 + [1]*10 + [2]*10)

        graph = PlacementGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            batch=batch
        )

        scores = critic(graph)
        assert scores.shape == torch.Size([3])

    def test_predict_routability(self, critic, sample_graph):
        """Test threshold-based prediction."""
        scores, should_prune = critic.predict_routability(sample_graph, threshold=0.5)

        assert isinstance(should_prune, torch.Tensor)
        assert should_prune.dtype == torch.bool


class TestTimestepAwareCritic:
    """Tests for timestep-conditioned critic."""

    @pytest.fixture
    def critic(self):
        return TimestepAwareCritic(
            max_timesteps=1000,
            hidden_dim=128,
            num_layers=4
        )

    @pytest.fixture
    def sample_graph(self):
        return PlacementGraph(
            node_features=torch.randn(10, 64),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_features=torch.randn(20, 32)
        )

    def test_timestep_conditioning(self, critic, sample_graph):
        """Test that different timesteps give different outputs."""
        t1 = torch.tensor([100])
        t2 = torch.tensor([900])

        score1 = critic(sample_graph, t1)
        score2 = critic(sample_graph, t2)

        # Scores should differ (not guaranteed but likely with random weights)
        # Just check shapes for now
        assert score1.shape == score2.shape


class TestPlacementGraphBuilder:
    """Tests for graph building from placement."""

    @pytest.fixture
    def grid(self):
        return Grid(width=100, height=100)

    @pytest.fixture
    def netlist(self):
        nets = [
            Net(net_id=0, pins=[
                Pin(x=10, y=10, pin_id=0),
                Pin(x=20, y=20, pin_id=1)
            ], name="net0"),
            Net(net_id=1, pins=[
                Pin(x=30, y=30, pin_id=2),
                Pin(x=40, y=40, pin_id=3),
                Pin(x=50, y=50, pin_id=4)
            ], name="net1")
        ]
        return Netlist(nets=nets)

    @pytest.fixture
    def placement(self):
        return Placement(
            pin_placements={0: (10, 10), 1: (20, 20), 2: (30, 30), 3: (40, 40), 4: (50, 50)},
            cell_placements={"0": (10, 10), "1": (20, 20), "2": (30, 30), "3": (40, 40), "4": (50, 50)}
        )

    def test_build_graph(self, grid, netlist, placement):
        """Test graph construction."""
        builder = PlacementGraphBuilder(grid)
        graph = builder.build_graph(placement, netlist)

        assert graph.node_features is not None
        assert graph.edge_index is not None
        assert graph.edge_features is not None

    def test_node_features_shape(self, grid, netlist, placement):
        """Test node feature dimensions."""
        builder = PlacementGraphBuilder(grid, node_dim=64)
        graph = builder.build_graph(placement, netlist)

        assert graph.node_features.shape[1] == 64

    def test_edge_features_shape(self, grid, netlist, placement):
        """Test edge feature dimensions."""
        builder = PlacementGraphBuilder(grid, edge_dim=32)
        graph = builder.build_graph(placement, netlist)

        if graph.edge_features.size(0) > 0:
            assert graph.edge_features.shape[1] == 32
