"""Value network: predicts expected reward from partially denoised latents.

V_φ(x_t, t) ≈ E[R(x_0) | x_t]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..core.routing.grid import Grid
from ..core.routing.netlist import Netlist


class ValueNetwork(nn.Module):
    """Value network for predicting expected reward from partial latents.
    
    Architecture:
    - CNN encoder for x_t and grid features
    - MLP for combined features (latent + grid + netlist + timestep)
    - Output: scalar reward estimate
    """
    
    def __init__(
        self,
        latent_channels: int = 1,
        grid_feature_dim: int = 4,
        netlist_embedding_dim: int = 64,
        time_emb_dim: int = 128,
        base_channels: int = 32
    ):
        """Initialize value network.
        
        Args:
            latent_channels: Number of channels in latent x_t
            grid_feature_dim: Dimension of grid features
            netlist_embedding_dim: Dimension of netlist embedding
            time_emb_dim: Dimension of time embedding
            base_channels: Base number of channels for CNN
        """
        super().__init__()
        
        # CNN encoder for latent
        self.latent_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten()
        )
        latent_enc_dim = base_channels * 2
        
        # CNN encoder for grid features (if provided)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(grid_feature_dim, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        grid_enc_dim = base_channels
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        time_enc_dim = 32
        
        # Combined MLP
        combined_dim = latent_enc_dim + grid_enc_dim + netlist_embedding_dim + time_enc_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # Scalar reward estimate
        )
    
    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        grid_features: Optional[torch.Tensor] = None,
        netlist_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x_t: Partially denoised latent [B, C, H, W] or [B, H, W, C]
            timestep: Timestep [B]
            grid_features: Grid features [B, F_grid, H, W] (optional)
            netlist_embedding: Netlist embedding [B, F_netlist] (optional)
        
        Returns:
            Value estimate [B, 1]
        """
        # Ensure x_t is [B, C, H, W]
        if x_t.dim() == 4 and x_t.shape[-1] < x_t.shape[1]:
            # Assume [B, H, W, C] -> [B, C, H, W]
            x_t = x_t.permute(0, 3, 1, 2)
        
        B = x_t.shape[0]
        
        # Encode latent
        h_latent = self.latent_encoder(x_t)  # [B, latent_enc_dim]
        
        # Encode grid features (or use zeros if not provided)
        if grid_features is not None:
            h_grid = self.grid_encoder(grid_features)  # [B, grid_enc_dim]
        else:
            h_grid = torch.zeros(B, self.grid_encoder[-2].out_features, device=x_t.device)
        
        # Time embedding
        time_emb = self.time_embed(timestep)  # [B, time_emb_dim]
        h_time = self.time_mlp(time_emb)  # [B, time_enc_dim]
        
        # Netlist embedding (or zeros if not provided)
        if netlist_embedding is not None:
            h_netlist = netlist_embedding  # [B, netlist_embedding_dim]
        else:
            h_netlist = torch.zeros(B, 64, device=x_t.device)  # Default dimension
        
        # Combine features
        h_combined = torch.cat([h_latent, h_grid, h_netlist, h_time], dim=1)  # [B, combined_dim]
        
        # MLP
        value = self.mlp(h_combined)  # [B, 1]
        
        return value


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Create time embeddings.
        
        Args:
            time: Timesteps [B]
        
        Returns:
            Embeddings [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


def create_grid_features(grid: Grid, H: int, W: int) -> torch.Tensor:
    """Create grid feature tensor.
    
    Args:
        grid: Grid structure
        H: Height
        W: Width
    
    Returns:
        Grid features [1, F_grid, H, W]
    """
    # Simple features: capacity, blockages, etc.
    # For now, create uniform features
    features = torch.ones(1, 4, H, W)
    
    # Feature channels:
    # 0: capacity (uniform for now)
    # 1: blockage mask (0 = blocked, 1 = free)
    # 2: x coordinate (normalized)
    # 3: y coordinate (normalized)
    
    features[0, 0] = 1.0  # Capacity
    features[0, 1] = 1.0  # No blockages
    
    # Coordinate features
    for y in range(H):
        for x in range(W):
            features[0, 2, y, x] = x / max(W - 1, 1)
            features[0, 3, y, x] = y / max(H - 1, 1)
    
    return features

