"""Simple UNet architecture for 2D routing diffusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TimeEmbedding(nn.Module):
    """Simple time embedding."""

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
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class UNet(nn.Module):
    """Simple UNet for 2D routing diffusion.

    A much simpler implementation to avoid bugs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        time_emb_dim: int = 128,
        cond_dim: int = 64
    ):
        super().__init__()

        # Time embedding (placeholder)
        self.time_embed = TimeEmbedding(time_emb_dim)

        # Conditioning (placeholder)
        self.cond_dim = cond_dim

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)

        # Decoder
        self.dec3 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1)
        self.dec2 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.dec1 = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        # Activations
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W]
            timestep: Timesteps [B]
            conditioning: Optional conditioning tensor

        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding (placeholder - not used)
        _ = self.time_embed(timestep)

        # Encoder
        h1 = self.act(self.enc1(x))
        h2 = self.act(self.enc2(F.max_pool2d(h1, 2)))

        h3 = self.act(self.enc3(F.max_pool2d(h2, 2)))

        # Decoder
        h = self.act(self.dec3(h3))
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)

        h = h + h2  # Skip connection
        h = self.act(self.dec2(h))
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)

        h = h + h1  # Skip connection
        h = self.dec1(h)

        return h