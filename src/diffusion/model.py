"""Routing diffusion model.

Generates routing assignments by denoising from noise.

State representation:
- Each net n has latent z_n ∈ Δ^|E_n| (distribution over feasible PIPs/edges)
- Denoising progressively commits routing decisions
- Terminal state: all nets have concentrated (near-deterministic) routing

Key insight: Diffusion handles *proposal*, search handles *verification*.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from .schedule import NoiseSchedule, DDPMSchedule
from ..core.routing.netlist import Netlist, Net
from ..core.routing.grid import Grid


@dataclass
class RoutingState:
    """Diffusion state for routing.

    Represents partial routing assignment during denoising.

    Attributes:
        net_latents: Dict[net_id -> latent tensor over PIPs]
        timestep: Current diffusion timestep
        routed_nets: Set of net IDs that are fully committed
        congestion_map: Current congestion on routing resources
    """
    net_latents: Dict[int, torch.Tensor]  # net_id -> [num_pips] logits
    timestep: int
    routed_nets: set  # Which nets are fully committed
    congestion_map: Optional[torch.Tensor] = None  # [H, W, layers] or sparse

    def is_terminal(self) -> bool:
        """All nets routed or timestep exhausted."""
        return self.timestep <= 0

    def get_unrouted_nets(self) -> List[int]:
        """Get list of nets not yet fully committed."""
        return [nid for nid in self.net_latents.keys() if nid not in self.routed_nets]


class NetEncoder(nn.Module):
    """Encode net structure for conditioning.

    Captures:
    - Source/sink positions
    - Net fanout
    - Timing criticality
    - Spatial span (bounding box)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encode net properties
        self.net_mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),  # fanout, bbox, criticality, etc.
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Encode source/sink positions
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # (x1,y1,x2,y2) normalized
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        net_features: torch.Tensor,  # [num_nets, 8]
        net_positions: torch.Tensor  # [num_nets, 4] bbox
    ) -> torch.Tensor:
        """Encode nets to embeddings."""
        net_emb = self.net_mlp(net_features)
        pos_emb = self.pos_encoder(net_positions)
        return net_emb + pos_emb


class CongestionEncoder(nn.Module):
    """Encode current congestion state.

    Input: congestion map (usage per routing resource)
    Output: embedding capturing global and local congestion
    """

    def __init__(self, hidden_dim: int = 128, grid_size: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim

        # CNN for spatial congestion
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)
        )

        self.fc = nn.Linear(64 * 8 * 8, hidden_dim)

    def forward(self, congestion: torch.Tensor) -> torch.Tensor:
        """Encode congestion map."""
        if congestion is None:
            return torch.zeros(1, self.hidden_dim)

        if congestion.dim() == 2:
            congestion = congestion.unsqueeze(0).unsqueeze(0)

        x = self.conv(congestion)
        x = x.flatten(1)
        return self.fc(x)


class RoutingDenoiser(nn.Module):
    """Denoising network for routing.

    Predicts noise in net routing latents conditioned on:
    - Current noisy routing assignments
    - Timestep
    - Net embeddings
    - Congestion state

    A denoising step corresponds to:
    - Fixing routing for one net, OR
    - Fixing a segment/Steiner subtree, OR
    - Committing a bundle of correlated nets
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_pips_per_net: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_pips = max_pips_per_net

        # Embed PIP logits
        self.pip_encoder = nn.Sequential(
            nn.Linear(max_pips_per_net, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Congestion encoder
        self.congestion_encoder = CongestionEncoder(hidden_dim)

        # Cross-attention between nets
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: predict noise per net's PIP logits
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_pips_per_net)
        )

    def forward(
        self,
        net_latents: torch.Tensor,  # [B, num_nets, max_pips]
        timestep: torch.Tensor,     # [B]
        net_embeds: torch.Tensor,   # [B, num_nets, hidden_dim]
        congestion_embed: torch.Tensor  # [B, hidden_dim]
    ) -> torch.Tensor:
        """Predict noise in PIP logits."""
        B, N, P = net_latents.shape

        # Encode current routing latents
        h = self.pip_encoder(net_latents)  # [B, N, hidden_dim]

        # Add net embeddings
        h = h + net_embeds

        # Add time embedding (broadcast to all nets)
        t = timestep.float().view(B, 1) / 1000.0
        t_embed = self.time_embed(t).unsqueeze(1).expand(-1, N, -1)
        h = h + t_embed

        # Add congestion embedding (broadcast)
        c_embed = congestion_embed.unsqueeze(1).expand(-1, N, -1)
        h = h + c_embed

        # Cross-attention between nets
        h = self.transformer(h)

        # Predict noise
        noise = self.output(h)

        return noise


class RoutingDiffusion(nn.Module):
    """Complete routing diffusion model.

    Diffusion provides:
    - A distribution over plausible routing paths
    - A policy prior for tree expansion

    Diffusion does NOT guarantee legality or optimality.
    That's what the search and real router are for.
    """

    def __init__(
        self,
        schedule: Optional[NoiseSchedule] = None,
        hidden_dim: int = 256,
        max_pips_per_net: int = 1000,
        **kwargs
    ):
        super().__init__()

        self.schedule = schedule or DDPMSchedule(num_timesteps=1000)
        self.num_timesteps = getattr(self.schedule, 'num_timesteps', 1000)
        self.max_pips = max_pips_per_net

        self.net_encoder = NetEncoder(hidden_dim=128)
        self.congestion_encoder = CongestionEncoder(hidden_dim=128)
        self.denoiser = RoutingDenoiser(
            hidden_dim=hidden_dim,
            max_pips_per_net=max_pips_per_net,
            **kwargs
        )

    def forward(
        self,
        net_latents: torch.Tensor,  # [B, num_nets, max_pips]
        timestep: torch.Tensor,
        net_features: torch.Tensor,
        net_positions: torch.Tensor,
        congestion: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict noise in routing latents."""
        B = net_latents.size(0)

        # Encode nets
        net_embeds = self.net_encoder(net_features, net_positions)
        net_embeds = net_embeds.unsqueeze(0).expand(B, -1, -1)

        # Encode congestion
        cong_embed = self.congestion_encoder(congestion)
        if cong_embed.size(0) == 1:
            cong_embed = cong_embed.expand(B, -1)

        return self.denoiser(net_latents, timestep, net_embeds, cong_embed)

    def denoise_step(
        self,
        state: RoutingState,
        net_features: torch.Tensor,
        net_positions: torch.Tensor
    ) -> RoutingState:
        """Single denoising step.

        Corresponds to committing routing decisions for one or more nets.
        Reduces routing entropy.
        """
        # Stack net latents into tensor
        net_ids = list(state.net_latents.keys())
        num_nets = len(net_ids)

        latents = torch.zeros(1, num_nets, self.max_pips)
        for i, nid in enumerate(net_ids):
            lat = state.net_latents[nid]
            latents[0, i, :len(lat)] = lat

        t = torch.tensor([state.timestep])

        # Predict noise
        with torch.no_grad():
            predicted_noise = self.forward(
                latents, t, net_features, net_positions, state.congestion_map
            )

        # Denoise (simplified DDPM step)
        alpha = 1 - 0.0001 - (0.02 - 0.0001) * state.timestep / self.num_timesteps
        new_latents = (latents - (1 - alpha) * predicted_noise) / (alpha ** 0.5)

        # Add noise for non-terminal steps
        if state.timestep > 1:
            noise = torch.randn_like(new_latents) * 0.1
            new_latents = new_latents + noise

        # Update net latents
        new_net_latents = {}
        for i, nid in enumerate(net_ids):
            orig_len = len(state.net_latents[nid])
            new_net_latents[nid] = new_latents[0, i, :orig_len]

        # Check if any nets became "committed" (low entropy)
        new_routed = set(state.routed_nets)
        for nid, lat in new_net_latents.items():
            probs = F.softmax(lat, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum()
            if entropy < 0.5:  # Low entropy = committed
                new_routed.add(nid)

        return RoutingState(
            net_latents=new_net_latents,
            timestep=max(0, state.timestep - 1),
            routed_nets=new_routed,
            congestion_map=state.congestion_map
        )

    def decode_routing(
        self,
        state: RoutingState,
        pip_indices: Dict[int, List[int]]  # net_id -> list of PIP indices
    ) -> Dict[int, List[int]]:
        """Decode latents to discrete routing assignments.

        For each net, select the highest-probability PIPs to form a route.
        """
        routing = {}
        for net_id, latent in state.net_latents.items():
            # Get top-k PIPs (greedy decoding)
            probs = F.softmax(latent, dim=-1)
            num_pips_needed = 10  # Rough estimate, should come from net
            top_pips = probs.topk(num_pips_needed).indices.tolist()

            # Map to actual PIP indices
            if net_id in pip_indices:
                routing[net_id] = [pip_indices[net_id][i] for i in top_pips
                                   if i < len(pip_indices[net_id])]
            else:
                routing[net_id] = top_pips

        return routing


def create_routing_diffusion(config: Dict[str, Any]) -> RoutingDiffusion:
    """Factory function."""
    schedule = DDPMSchedule(num_timesteps=config.get("num_timesteps", 1000))

    return RoutingDiffusion(
        schedule=schedule,
        hidden_dim=config.get("hidden_dim", 256),
        max_pips_per_net=config.get("max_pips_per_net", 1000),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 6),
        dropout=config.get("dropout", 0.1)
    )
