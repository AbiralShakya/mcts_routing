"""Diffusion module: Denoising model for routing.

Generates routing assignments by denoising.
Each net has a latent z_n ∈ Δ^|E_n| over feasible PIPs.

Early timesteps: high-entropy, many routing possibilities
Later timesteps: concentrated paths through fabric
"""

from .model import RoutingDiffusion, RoutingState, create_routing_diffusion
from .schedule import NoiseSchedule, DDPMSchedule
from .sampler import denoise_step

__all__ = [
    "RoutingDiffusion",
    "RoutingState",
    "create_routing_diffusion",
    "NoiseSchedule",
    "DDPMSchedule",
    "denoise_step"
]
