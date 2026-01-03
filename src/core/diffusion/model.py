"""Diffusion model wrapper."""

import torch
import torch.nn as nn
from typing import Optional

from ...models.unet import UNet
from .conditioning import ConditionEncoder


class DiffusionModel(nn.Module):
    """Diffusion model for routing potentials."""
    
    def __init__(
        self,
        unet: Optional[UNet] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_dim: int = 128,
        **unet_kwargs
    ):
        super().__init__()
        if unet is None:
            self.unet = UNet(
                in_channels=in_channels,
                out_channels=out_channels,
                cond_dim=cond_dim,
                **unet_kwargs
            )
        else:
            self.unet = unet
        
        self.cond_encoder = ConditionEncoder(embedding_dim=cond_dim)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict noise at timestep t.
        
        Args:
            x_t: Noisy data [B, C, H, W]
            t: Timesteps [B]
            conditioning: Optional conditioning tensor [B, cond_dim]
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        return self.unet(x_t, t, conditioning)
    
    def reverse_step(
        self,
        x_t: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Reverse diffusion step (interface for MCTS).
        
        Args:
            x_t: Current latent [B, C, H, W]
            eps: Noise realization [B, C, H, W]
            t: Timestep [B]
        
        Returns:
            x_{t-1}: Next latent [B, C, H, W]
        """
        # This is a placeholder - actual implementation depends on schedule
        # Will be implemented in integration layer
        raise NotImplementedError("Use sampler.reverse_step_* functions")

