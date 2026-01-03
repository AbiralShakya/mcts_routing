"""Unit tests for placer module."""

import pytest
import torch

from src.placer.placement_diffusion import (
    PlacementDiffusion,
    PlacementState,
    NetlistEncoder,
    PlacementDenoiser
)
from src.placer.mcts_placer import MCTSPlacer, PlacerNode
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist, Net, Pin


class TestPlacementState:
    """Tests for PlacementState."""

    def test_creation(self):
        state = PlacementState(
            positions=torch.randn(10, 2),
            cell_types=torch.zeros(10, dtype=torch.long),
            timestep=100
        )
        assert state.positions.shape == (10, 2)
        assert state.timestep == 100


class TestNetlistEncoder:
    """Tests for NetlistEncoder."""

    @pytest.fixture
    def encoder(self):
        return NetlistEncoder(
            num_cell_types=16,
            hidden_dim=128,
            num_layers=3
        )

    def test_forward(self, encoder):
        cell_types = torch.randint(0, 16, (20,))
        edge_index = torch.randint(0, 20, (2, 40))

        embeddings = encoder(cell_types, edge_index)
        assert embeddings.shape == (20, 128)


class TestPlacementDenoiser:
    """Tests for PlacementDenoiser."""

    @pytest.fixture
    def denoiser(self):
        return PlacementDenoiser(
            hidden_dim=128,
            num_heads=4,
            num_layers=3
        )

    def test_forward(self, denoiser):
        B, N = 4, 20
        positions = torch.randn(B, N, 2)
        timestep = torch.randint(0, 1000, (B,))
        netlist_embed = torch.randn(B, N, 128)

        noise = denoiser(positions, timestep, netlist_embed)
        assert noise.shape == (B, N, 2)


class TestPlacementDiffusion:
    """Tests for PlacementDiffusion model."""

    @pytest.fixture
    def model(self):
        return PlacementDiffusion(
            hidden_dim=128,
            num_cell_types=8,
            num_heads=4,
            num_layers=3
        )

    @pytest.fixture
    def sample_input(self):
        B, N = 2, 10
        return {
            "positions": torch.randn(B, N, 2),
            "timestep": torch.randint(0, 1000, (B,)),
            "cell_types": torch.randint(0, 8, (B, N)),
            "edge_index": torch.randint(0, N, (2, 20))
        }

    def test_forward(self, model, sample_input):
        """Test noise prediction."""
        noise = model(
            sample_input["positions"],
            sample_input["timestep"],
            sample_input["cell_types"],
            sample_input["edge_index"]
        )
        assert noise.shape == sample_input["positions"].shape

    def test_q_sample(self, model):
        """Test forward diffusion."""
        x_0 = torch.randn(4, 10, 2)
        t = torch.randint(0, 1000, (4,))

        x_t, noise = model.q_sample(x_0, t)
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape

    def test_denoise_step(self, model):
        """Test single denoising step."""
        state = PlacementState(
            positions=torch.randn(10, 2),
            cell_types=torch.zeros(10, dtype=torch.long),
            timestep=100
        )
        cell_types = torch.zeros(10, dtype=torch.long)
        edge_index = torch.randint(0, 10, (2, 20))

        new_state = model.denoise_step(state, cell_types, edge_index)

        assert new_state.timestep == 99
        assert new_state.positions.shape == state.positions.shape

    def test_decode_placement(self, model):
        """Test decoding to discrete placement."""
        grid = Grid(width=50, height=50)
        state = PlacementState(
            positions=torch.rand(5, 2),  # [0, 1] positions
            cell_types=torch.zeros(5, dtype=torch.long),
            timestep=0
        )

        placement = model.decode_placement(state, grid)

        assert placement.cell_placements is not None
        assert len(placement.cell_placements) == 5

        # Check positions are within grid
        for name, (x, y) in placement.cell_placements.items():
            assert 0 <= x < 50
            assert 0 <= y < 50


class TestPlacerNode:
    """Tests for MCTS node."""

    def test_q_value(self):
        state = PlacementState(
            positions=torch.randn(10, 2),
            cell_types=torch.zeros(10, dtype=torch.long),
            timestep=50
        )
        node = PlacerNode(state=state)

        # Initial Q is 0
        assert node.q_value == 0.0

        # After updates
        node.visit_count = 10
        node.total_value = 5.0
        assert node.q_value == 0.5

    def test_is_terminal(self):
        state_terminal = PlacementState(
            positions=torch.randn(10, 2),
            cell_types=torch.zeros(10, dtype=torch.long),
            timestep=0
        )
        state_not_terminal = PlacementState(
            positions=torch.randn(10, 2),
            cell_types=torch.zeros(10, dtype=torch.long),
            timestep=50
        )

        assert PlacerNode(state=state_terminal).is_terminal()
        assert not PlacerNode(state=state_not_terminal).is_terminal()

    def test_hash(self):
        state = PlacementState(
            positions=torch.randn(10, 2),
            cell_types=torch.zeros(10, dtype=torch.long),
            timestep=50
        )
        node1 = PlacerNode(state=state)
        node2 = PlacerNode(state=state)

        # Same state should have same hash
        assert hash(node1) == hash(node2)


class TestMCTSPlacerIntegration:
    """Integration tests for MCTSPlacer (requires all components)."""

    @pytest.fixture
    def simple_netlist(self):
        nets = [
            Net(net_id=0, pins=[
                Pin(x=0, y=0, pin_id=0),
                Pin(x=10, y=10, pin_id=1)
            ], name="net0")
        ]
        return Netlist(nets=nets)

    @pytest.fixture
    def grid(self):
        return Grid(width=20, height=20)

    def test_placer_creation(self, simple_netlist, grid):
        """Test placer can be instantiated (smoke test)."""
        from src.placer.placement_diffusion import PlacementDiffusion
        from src.critic.gnn import RoutabilityCritic
        from src.bridge.router import NextPNRRouter

        diffusion = PlacementDiffusion(hidden_dim=64, num_layers=2)
        critic = RoutabilityCritic(hidden_dim=64, num_layers=2)
        router = NextPNRRouter()  # Will use subprocess mode

        # This should not raise
        placer = MCTSPlacer(
            diffusion=diffusion,
            critic=critic,
            router=router,
            grid=grid,
            netlist=simple_netlist,
            max_simulations=1,
            device="cpu"
        )

        assert placer is not None
