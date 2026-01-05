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

    def __init__(self, hidden_dim: int = 128, net_feat_dim: int = 7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net_feat_dim = net_feat_dim

        # Encode net properties
        self.net_mlp = nn.Sequential(
            nn.Linear(net_feat_dim, hidden_dim),  # fanout, bbox, criticality, etc.
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

    def forward(self, congestion: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """Encode congestion map."""
        if congestion is None:
            # Get device from model parameters
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.hidden_dim, device=device)

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
        net_feat_dim: int = 7,
        **kwargs
    ):
        super().__init__()

        self.schedule = schedule or DDPMSchedule(num_timesteps=1000)
        self.num_timesteps = getattr(self.schedule, 'num_timesteps', 1000)
        self.max_pips = max_pips_per_net
        self.hidden_dim = hidden_dim

        # All encoders use the same hidden_dim for consistency
        self.net_encoder = NetEncoder(hidden_dim=hidden_dim, net_feat_dim=net_feat_dim)
        self.congestion_encoder = CongestionEncoder(hidden_dim=hidden_dim)
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
        cong_embed = self.congestion_encoder(congestion, batch_size=B)
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
        # Get device from input tensors
        device = net_features.device

        # Stack net latents into tensor
        net_ids = list(state.net_latents.keys())
        num_nets = len(net_ids)

        latents = torch.zeros(1, num_nets, self.max_pips, device=device)
        for i, nid in enumerate(net_ids):
            lat = state.net_latents[nid].to(device)
            latents[0, i, :len(lat)] = lat

        t = torch.tensor([state.timestep], device=device)

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
        pip_indices: Dict[int, List[int]],  # net_id -> list of PIP indices
        netlist: Optional['Netlist'] = None  # For computing num_pips from topology
    ) -> Dict[int, List[int]]:
        """Decode latents to discrete routing assignments.

        For each net, select PIPs to form a connected route.
        Uses soft potential → edge cost conversion for better selection.

        Args:
            state: Current routing state with net latents
            pip_indices: Mapping from net_id to list of actual PIP indices
            netlist: Optional netlist for computing required path length

        Returns:
            Dict mapping net_id -> list of selected PIP indices
        """
        routing = {}

        for net_id, latent in state.net_latents.items():
            # Convert logits to probabilities
            probs = F.softmax(latent, dim=-1)

            # Estimate PIPs needed based on net topology
            # If netlist provided, use HPWL estimate; otherwise use latent size
            if netlist is not None:
                num_pips_needed = self._estimate_pips_for_net(net_id, netlist)
            else:
                # Heuristic: use ~20% of available PIPs, minimum 5
                num_pips_needed = max(5, latent.size(-1) // 5)

            # Ensure we don't request more PIPs than available
            num_pips_needed = min(num_pips_needed, latent.size(-1))

            # Select PIPs using temperature-scaled sampling for diversity
            # Higher probability PIPs are preferred, but allow some exploration
            temperature = 0.5  # Lower = more greedy
            scaled_probs = F.softmax(latent / temperature, dim=-1)

            # Get top-k with highest scaled probabilities
            top_values, top_indices = scaled_probs.topk(num_pips_needed)
            selected_pips = top_indices.tolist()

            # Map to actual PIP indices if mapping provided
            if net_id in pip_indices and pip_indices[net_id]:
                pip_list = pip_indices[net_id]
                routing[net_id] = [
                    pip_list[i] for i in selected_pips
                    if i < len(pip_list)
                ]
            else:
                routing[net_id] = selected_pips

            # Sort by probability (highest first) for deterministic ordering
            if routing[net_id]:
                pip_probs = [(p, probs[selected_pips[i]].item())
                             for i, p in enumerate(routing[net_id])]
                pip_probs.sort(key=lambda x: -x[1])
                routing[net_id] = [p for p, _ in pip_probs]

        return routing

    def _estimate_pips_for_net(
        self,
        net_id: int,
        netlist: 'Netlist'
    ) -> int:
        """Estimate number of PIPs needed to route a net.

        Uses HPWL (half-perimeter wirelength) as estimate.
        Each unit of wirelength typically needs 1-2 PIPs.

        Args:
            net_id: Net ID to estimate for
            netlist: Netlist containing the net

        Returns:
            Estimated number of PIPs needed
        """
        net = netlist.get_net(net_id)
        if net is None or len(net.pins) < 2:
            return 5  # Minimum

        # Compute HPWL
        xs = [p.x for p in net.pins]
        ys = [p.y for p in net.pins]
        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))

        # Steiner tree typically needs HPWL * factor PIPs
        # Factor depends on net fanout (more sinks = more branching)
        fanout = len(net.pins) - 1
        steiner_factor = 1.0 + 0.2 * min(fanout, 5)  # Cap at fanout 5

        estimated_pips = int(hpwl * steiner_factor) + fanout

        # Clamp to reasonable range
        return max(5, min(estimated_pips, 100))


def compute_congestion_from_latents(
    net_latents: Dict[int, torch.Tensor],
    netlist: Netlist,
    grid_size: Tuple[int, int],
    device: str = "cpu"
) -> torch.Tensor:
    """Compute congestion map from soft PIP probabilities.

    Estimates routing resource usage by spreading probability mass
    over the bounding box of each net, weighted by routing probability.

    Args:
        net_latents: Dict mapping net_id to latent tensor (probs over PIPs)
        netlist: Netlist with net topology
        grid_size: (width, height) of the grid
        device: Device for output tensor

    Returns:
        Congestion map tensor of shape [height, width]
    """
    width, height = grid_size
    congestion = torch.zeros(height, width, device=device)

    for net_id, latent in net_latents.items():
        net = netlist.get_net(net_id)
        if net is None or len(net.pins) < 2:
            continue

        # Compute net's bounding box
        xs = [p.x for p in net.pins]
        ys = [p.y for p in net.pins]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Compute routing probability (how "committed" is this net)
        probs = F.softmax(latent, dim=-1)
        max_prob = probs.max().item()
        entropy = -(probs * (probs + 1e-8).log()).sum().item()
        max_entropy = torch.log(torch.tensor(float(latent.size(-1)))).item()

        # Commitment score: 0 = uniform, 1 = fully committed
        commitment = 1.0 - (entropy / max(max_entropy, 1e-8))

        # Spread probability mass over bounding box
        # More committed = more concentrated usage
        bbox_area = max(1, (max_x - min_x + 1) * (max_y - min_y + 1))
        usage_per_cell = commitment / bbox_area

        # Add usage to congestion map (clamp to grid bounds)
        for y in range(max(0, min_y), min(height, max_y + 1)):
            for x in range(max(0, min_x), min(width, max_x + 1)):
                congestion[y, x] += usage_per_cell

    # Normalize to [0, 1] range
    if congestion.max() > 0:
        congestion = congestion / congestion.max()

    return congestion


def create_routing_diffusion(config: Dict[str, Any]) -> RoutingDiffusion:
    """Factory function."""
    schedule = DDPMSchedule(num_timesteps=config.get("num_timesteps", 1000))

    return RoutingDiffusion(
        schedule=schedule,
        hidden_dim=config.get("hidden_dim", 256),
        max_pips_per_net=config.get("max_pips_per_net", 1000),
        net_feat_dim=config.get("net_feat_dim", 7),  # Default 7 to match generated data
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 6),
        dropout=config.get("dropout", 0.1)
    )
