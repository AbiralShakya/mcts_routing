"""Loss functions with Lipschitz, identifiability, and smoothness regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..core.decoding.decoder import RoutingPotentials, Decoder


class DiffusionLoss(nn.Module):
    """Complete diffusion loss with all regularization terms.
    
    L_total = L_diffusion + λ_L·L_lipschitz + λ_E·L_entropy + λ_S·L_smooth
    """
    
    def __init__(
        self,
        lambda_lipschitz: float = 0.1,
        lambda_entropy: float = 0.01,
        lambda_smooth: float = 0.001,
        lipschitz_target: float = 5.0,
        perturbation_scale: float = 0.01
    ):
        """Initialize loss function.
        
        Args:
            lambda_lipschitz: Weight for Lipschitz regularization
            lambda_entropy: Weight for entropy regularization
            lambda_smooth: Weight for smoothness regularization
            lipschitz_target: Target Lipschitz constant
            perturbation_scale: Scale for latent perturbations
        """
        super().__init__()
        self.lambda_lipschitz = lambda_lipschitz
        self.lambda_entropy = lambda_entropy
        self.lambda_smooth = lambda_smooth
        self.lipschitz_target = lipschitz_target
        self.perturbation_scale = perturbation_scale
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        decoder: Optional[Decoder] = None,
        latent: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute total loss.
        
        Args:
            predicted_noise: Predicted noise ε_θ [B, C, H, W]
            target_noise: Target noise ε [B, C, H, W]
            decoder: Decoder for regularization (optional)
            latent: Latent tensor for regularization (optional)
        
        Returns:
            Total loss
        """
        # Primary diffusion loss
        L_diff = F.mse_loss(predicted_noise, target_noise)
        
        # Regularization terms (only if decoder and latent provided)
        L_lip = torch.tensor(0.0, device=predicted_noise.device)
        L_ent = torch.tensor(0.0, device=predicted_noise.device)
        L_smooth = torch.tensor(0.0, device=predicted_noise.device)
        
        if decoder is not None and latent is not None:
            # Lipschitz regularization
            L_lip = self._lipschitz_penalty(decoder, latent)
            
            # Entropy regularization
            try:
                potentials = decoder.decode(latent, None, None)  # Grid/netlist not needed for entropy
                L_ent = self._entropy_penalty(potentials)
            except Exception:
                pass  # Skip if decode fails
            
            # Smoothness regularization (total variation)
            L_smooth = self._smoothness_penalty(decoder, latent)
        
        # Total loss
        L_total = (
            L_diff +
            self.lambda_lipschitz * L_lip +
            self.lambda_entropy * L_ent +
            self.lambda_smooth * L_smooth
        )
        
        return L_total
    
    def _lipschitz_penalty(
        self,
        decoder: Decoder,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """Compute Lipschitz regularization penalty.
        
        L_lipschitz = λ_L · E[max(0, ||D(x) - D(x')|| / ||x - x'|| - L_target)²]
        
        Args:
            decoder: Decoder
            latent: Latent tensor [B, C, H, W]
        
        Returns:
            Lipschitz penalty
        """
        # Create perturbed latent
        perturbation = torch.randn_like(latent) * self.perturbation_scale
        latent_perturbed = latent + perturbation
        
        # Decode both
        try:
            potentials_orig = decoder.decode(latent, None, None)
            potentials_pert = decoder.decode(latent_perturbed, None, None)
            
            # Compute Lipschitz ratio
            diff_potentials = torch.norm(potentials_orig.cost_field - potentials_pert.cost_field)
            diff_latent = torch.norm(perturbation)
            
            if diff_latent > 1e-8:
                lipschitz_ratio = diff_potentials / diff_latent
                # Penalty if exceeds target
                penalty = torch.clamp(lipschitz_ratio - self.lipschitz_target, min=0.0) ** 2
                return penalty
        except Exception:
            pass
        
        return torch.tensor(0.0, device=latent.device)
    
    def _entropy_penalty(self, potentials: RoutingPotentials) -> torch.Tensor:
        """Compute entropy regularization penalty.
        
        L_entropy = -λ_E · E[H(D(x))]
        where H(p) = -Σ p log p
        
        Args:
            potentials: Routing potentials
        
        Returns:
            Entropy penalty (negative because we maximize entropy)
        """
        cost_field = potentials.cost_field
        
        # Normalize to probabilities
        cost_sum = cost_field.sum()
        if cost_sum > 1e-8:
            probs = cost_field / cost_sum
            # Compute entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            # Negative because we want to maximize entropy
            return -self.lambda_entropy * entropy
        
        return torch.tensor(0.0, device=cost_field.device)
    
    def _smoothness_penalty(
        self,
        decoder: Decoder,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """Compute smoothness regularization (total variation).
        
        L_smooth = λ_S · E[||∇D(x)||²]
        
        Args:
            decoder: Decoder
            latent: Latent tensor [B, C, H, W]
        
        Returns:
            Smoothness penalty
        """
        latent.requires_grad_(True)
        
        try:
            potentials = decoder.decode(latent, None, None)
            cost_field = potentials.cost_field
            
            # Compute gradients
            grad_x = torch.autograd.grad(
                cost_field.sum(),
                latent,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Total variation: sum of squared gradients
            tv_penalty = (grad_x ** 2).sum()
            return self.lambda_smooth * tv_penalty
        except Exception:
            return torch.tensor(0.0, device=latent.device)


def diffusion_loss(predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    """Standard diffusion loss: ||ε - ε_θ||².
    
    Args:
        predicted_noise: Predicted noise
        target_noise: Target noise
    
    Returns:
        MSE loss
    """
    return F.mse_loss(predicted_noise, target_noise)


def lipschitz_regularization(
    decoder: Decoder,
    latent: torch.Tensor,
    lambda_l: float = 0.1,
    target_l: float = 5.0
) -> torch.Tensor:
    """Lipschitz regularization: λ_L * max(0, L - L_target)².
    
    Args:
        decoder: Decoder
        latent: Latent tensor
        lambda_l: Regularization weight
        target_l: Target Lipschitz constant
    
    Returns:
        Regularization penalty
    """
    # Create perturbed latent
    perturbation = torch.randn_like(latent) * 0.01
    latent_perturbed = latent + perturbation
    
    # Decode both
    try:
        potentials_orig = decoder.decode(latent, None, None)
        potentials_pert = decoder.decode(latent_perturbed, None, None)
        
        diff_potentials = torch.norm(potentials_orig.cost_field - potentials_pert.cost_field)
        diff_latent = torch.norm(perturbation)
        
        if diff_latent > 1e-8:
            lipschitz_ratio = diff_potentials / diff_latent
            penalty = torch.clamp(lipschitz_ratio - target_l, min=0.0) ** 2
            return lambda_l * penalty
    except Exception:
        pass
    
    return torch.tensor(0.0, device=latent.device)


def identifiability_regularization(
    potentials: RoutingPotentials,
    lambda_e: float = 0.01
) -> torch.Tensor:
    """Identifiability regularization: entropy maximization.
    
    L_entropy = -λ_E · E[H(D(x))]
    
    Args:
        potentials: Routing potentials
        lambda_e: Regularization weight
    
    Returns:
        Entropy penalty (negative because we maximize)
    """
    cost_field = potentials.cost_field
    cost_sum = cost_field.sum()
    
    if cost_sum > 1e-8:
        probs = cost_field / cost_sum
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        return -lambda_e * entropy  # Negative because we maximize
    
    return torch.tensor(0.0, device=cost_field.device)
