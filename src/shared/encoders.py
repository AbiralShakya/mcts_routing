"""Shared encoder components for diffusion and critic models.

These encoders are shared between the diffusion model (policy) and critic model (value)
to ensure consistent representation of routing states.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SharedNetEncoder(nn.Module):
    """Unified net encoder for both diffusion and critic.
    
    Encodes net structure information:
    - Net features (fanout, HPWL, bbox, etc.)
    - Net positions (bounding box)
    
    This ensures both models see nets the same way.
    """

    def __init__(self, hidden_dim: int = 128, net_feat_dim: int = 7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net_feat_dim = net_feat_dim

        # Encode net properties (fanout, bbox, criticality, etc.)
        self.net_mlp = nn.Sequential(
            nn.Linear(net_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Encode source/sink positions (bounding box)
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # (x1, y1, x2, y2) normalized
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        net_features: torch.Tensor,  # [B, num_nets, net_feat_dim] or [num_nets, net_feat_dim]
        net_positions: torch.Tensor   # [B, num_nets, 4] or [num_nets, 4]
    ) -> torch.Tensor:
        """Encode nets to embeddings.
        
        Args:
            net_features: Net feature vectors
            net_positions: Net bounding boxes [x1, y1, x2, y2]
        
        Returns:
            Net embeddings [B, num_nets, hidden_dim] or [num_nets, hidden_dim]
        """
        net_emb = self.net_mlp(net_features)
        pos_emb = self.pos_encoder(net_positions)
        return net_emb + pos_emb


class SharedCongestionEncoder(nn.Module):
    """Unified congestion encoder for both diffusion and critic.
    
    Encodes congestion maps using CNN to capture:
    - Global congestion patterns
    - Local hotspots
    - Spatial distribution
    
    Both models need to understand congestion the same way.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # CNN for spatial congestion encoding
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8)  # Fixed output size
        )

        # Fully connected layer to hidden_dim
        self.fc = nn.Linear(64 * 8 * 8, hidden_dim)

    def forward(
        self,
        congestion: Optional[torch.Tensor],
        batch_size: int = 1
    ) -> torch.Tensor:
        """Encode congestion map.
        
        Args:
            congestion: Congestion map [H, W] or [B, H, W] or None
            batch_size: Batch size (used when congestion is None)
        
        Returns:
            Congestion embedding [B, hidden_dim]
        """
        if congestion is None:
            # Return zero embedding if no congestion info
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.hidden_dim, device=device)

        # Handle different input shapes
        if congestion.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            congestion = congestion.unsqueeze(0).unsqueeze(0)
        elif congestion.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            congestion = congestion.unsqueeze(1)
        # If already 4D [B, 1, H, W], use as is

        # Forward through CNN
        x = self.conv(congestion)
        x = x.flatten(1)  # [B, 64*8*8]
        x = self.fc(x)    # [B, hidden_dim]

        return x


def create_shared_encoders(
    hidden_dim: int = 128,
    net_feat_dim: int = 7
) -> Tuple[SharedNetEncoder, SharedCongestionEncoder]:
    """Factory function to create shared encoders.
    
    Args:
        hidden_dim: Hidden dimension for embeddings
        net_feat_dim: Dimension of net feature vectors
    
    Returns:
        Tuple of (net_encoder, congestion_encoder)
    """
    net_encoder = SharedNetEncoder(hidden_dim=hidden_dim, net_feat_dim=net_feat_dim)
    congestion_encoder = SharedCongestionEncoder(hidden_dim=hidden_dim)
    
    return net_encoder, congestion_encoder

