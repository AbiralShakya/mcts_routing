"""Uncertainty-guided branching: branch when diffusion model is uncertain.

Computes variance across K perturbed predictions:
σ²(x_t, t) = (1/K) Σ_k (ε̂_k - ε̄)²
Branch if σ²(x_t, t) > τ_t
"""

import torch
from typing import Optional
from .node import MCTSNode


def compute_uncertainty(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    timestep: int,
    conditioning: Optional[torch.Tensor] = None,
    K: int = 5,
    eta: float = 0.01
) -> float:
    """Compute uncertainty via perturbation sensitivity.
    
    Args:
        model: Diffusion model
        x_t: Latent tensor [C, H, W] or [B, C, H, W]
        timestep: Current timestep
        conditioning: Conditioning tensor (optional)
        K: Number of perturbations
        eta: Perturbation scale
    
    Returns:
        Uncertainty variance σ²(x_t, t)
    """
    # Ensure batch dimension
    if x_t.dim() == 3:
        x_t = x_t.unsqueeze(0)
    
    B, C, H, W = x_t.shape
    device = x_t.device
    
    # Generate K perturbed versions
    perturbed_x = []
    for k in range(K):
        perturbation = torch.randn_like(x_t) * eta
        x_t_k = x_t + perturbation
        perturbed_x.append(x_t_k)
    
    # Stack for batch processing
    x_t_batch = torch.cat(perturbed_x, dim=0)  # [K*B, C, H, W]
    
    # Predict noise for all perturbations
    t_tensor = torch.full((K * B,), timestep, device=device, dtype=torch.long)
    
    with torch.no_grad():
        if conditioning is not None:
            # Repeat conditioning for all perturbations
            cond_batch = conditioning.repeat(K, *([1] * (conditioning.dim() - 1)))
            preds = model(x_t_batch, t_tensor, cond_batch)
        else:
            preds = model(x_t_batch, t_tensor)
    
    # Reshape to [K, B, C, H, W]
    preds = preds.view(K, B, *preds.shape[1:])
    
    # Compute variance across K predictions
    preds_mean = preds.mean(dim=0, keepdim=True)  # [1, B, C, H, W]
    variance = ((preds - preds_mean) ** 2).mean()  # Global variance
    
    return float(variance.item())


def should_branch_uncertainty(
    model: torch.nn.Module,
    node: MCTSNode,
    conditioning: Optional[torch.Tensor] = None,
    K: int = 5,
    eta: float = 0.01,
    tau: float = 0.05,
    tau_timestep_scale: bool = True,
    T: int = 1000
) -> bool:
    """Determine if should branch based on uncertainty.
    
    Args:
        model: Diffusion model
        node: Current MCTS node
        conditioning: Conditioning tensor (optional)
        K: Number of perturbations
        eta: Perturbation scale
        tau: Uncertainty threshold
        tau_timestep_scale: Whether to scale tau by timestep
        T: Total timesteps
    
    Returns:
        True if should branch (high uncertainty)
    """
    # Compute uncertainty
    variance = compute_uncertainty(
        model, node.latent, node.timestep, conditioning, K=K, eta=eta
    )
    
    # Scale threshold by timestep if requested
    if tau_timestep_scale:
        # Higher tolerance at early timesteps
        tau_scaled = tau * (1.0 - node.timestep / T)
    else:
        tau_scaled = tau
    
    # Branch if uncertainty exceeds threshold
    return variance > tau_scaled


def compute_uncertainty_batch(
    model: torch.nn.Module,
    x_t_batch: torch.Tensor,
    timesteps: torch.Tensor,
    conditioning: Optional[torch.Tensor] = None,
    K: int = 5,
    eta: float = 0.01
) -> torch.Tensor:
    """Compute uncertainty for batch of latents.
    
    Args:
        model: Diffusion model
        x_t_batch: Batch of latents [B, C, H, W]
        timesteps: Timesteps [B]
        conditioning: Conditioning tensor (optional)
        K: Number of perturbations
        eta: Perturbation scale
    
    Returns:
        Uncertainty variances [B]
    """
    B = x_t_batch.shape[0]
    device = x_t_batch.device
    uncertainties = []
    
    for i in range(B):
        x_t = x_t_batch[i:i+1]
        t = timesteps[i].item()
        cond = conditioning[i:i+1] if conditioning is not None else None
        
        uncertainty = compute_uncertainty(model, x_t, t, cond, K=K, eta=eta)
        uncertainties.append(uncertainty)
    
    return torch.tensor(uncertainties, device=device)

