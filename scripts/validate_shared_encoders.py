#!/usr/bin/env python3
"""Validate shared encoder architecture fix.

This script verifies that:
1. Duplicate CongestionEncoder has been removed from RoutingDenoiser
2. Shared encoders are properly connected
3. Gradients flow through shared encoders during training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.diffusion.model import RoutingDiffusion, RoutingDenoiser
from src.critic.gnn import RoutingCritic
from src.shared.encoders import create_shared_encoders


def test_denoiser_no_duplicate_encoder():
    """Test that RoutingDenoiser doesn't have duplicate congestion encoder."""
    print("=" * 70)
    print("TEST 1: RoutingDenoiser No Duplicate Encoder")
    print("=" * 70)
    
    denoiser = RoutingDenoiser(hidden_dim=128)
    
    # Check that congestion_encoder attribute doesn't exist
    has_cong_encoder = hasattr(denoiser, 'congestion_encoder')
    
    if has_cong_encoder:
        print("❌ FAIL: RoutingDenoiser still has self.congestion_encoder")
        print("   This creates redundancy with shared encoders!")
        return False
    else:
        print("✅ PASS: RoutingDenoiser has no duplicate congestion encoder")
        print("   Denoiser will use pre-computed embeddings from shared encoder")
        return True


def test_shared_encoder_connectivity():
    """Test that shared encoders are properly connected to both models."""
    print("\n" + "=" * 70)
    print("TEST 2: Shared Encoder Connectivity")
    print("=" * 70)
    
    # Create shared encoders
    net_encoder, cong_encoder = create_shared_encoders(hidden_dim=256)
    
    # Create models with shared encoders
    diffusion = RoutingDiffusion(
        hidden_dim=256,
        max_pips_per_net=50,
        shared_net_encoder=net_encoder,
        shared_congestion_encoder=cong_encoder,
        num_heads=4,
        num_layers=2
    )
    
    critic = RoutingCritic(
        hidden_dim=256,
        shared_net_encoder=net_encoder,
        shared_congestion_encoder=cong_encoder
    )
    
    # Verify they're using the same encoder instances
    same_net_encoder = diffusion.net_encoder is net_encoder and critic.shared_net_encoder is net_encoder
    same_cong_encoder = diffusion.congestion_encoder is cong_encoder and critic.shared_congestion_encoder is cong_encoder
    
    if same_net_encoder and same_cong_encoder:
        print("✅ PASS: Both models use the same shared encoder instances")
        print(f"   Net encoder ID: {id(net_encoder)}")
        print(f"   - Diffusion uses: {id(diffusion.net_encoder)}")
        print(f"   - Critic uses: {id(critic.shared_net_encoder)}")
        print(f"   Congestion encoder ID: {id(cong_encoder)}")
        print(f"   - Diffusion uses: {id(diffusion.congestion_encoder)}")
        print(f"   - Critic uses: {id(critic.shared_congestion_encoder)}")
        return True
    else:
        print("❌ FAIL: Models not using same shared encoder instances")
        return False


def test_gradient_flow():
    """Test that gradients flow through shared encoders."""
    print("\n" + "=" * 70)
    print("TEST 3: Gradient Flow Through Shared Encoders")
    print("=" * 70)
    
    # Create shared encoders
    net_encoder, cong_encoder = create_shared_encoders(hidden_dim=128)
    
    # Create diffusion model
    diffusion = RoutingDiffusion(
        hidden_dim=128,
        max_pips_per_net=20,
        shared_net_encoder=net_encoder,
        shared_congestion_encoder=cong_encoder,
        num_heads=4,
        num_layers=2
    )
    
    # Create dummy input
    B, num_nets, max_pips = 2, 3, 20
    net_latents = torch.randn(B, num_nets, max_pips)
    timestep = torch.randint(0, 100, (B,))
    net_features = torch.randn(B, num_nets, 7)
    net_positions = torch.randn(B, num_nets, 4)
    congestion = torch.randn(B, 16, 16)  # 16x16 grid
    
    # Forward pass
    output = diffusion(net_latents, timestep, net_features, net_positions, congestion)
    
    # Compute loss and backward
    loss = output.mean()
    loss.backward()
    
    # Check gradients on shared encoders
    net_encoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                                for p in net_encoder.parameters())
    cong_encoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                                 for p in cong_encoder.parameters())
    
    if net_encoder_has_grad and cong_encoder_has_grad:
        print("✅ PASS: Gradients flow through both shared encoders")
        
        # Count parameters with gradients
        net_params_with_grad = sum(1 for p in net_encoder.parameters() 
                                    if p.grad is not None and p.grad.abs().sum() > 0)
        cong_params_with_grad = sum(1 for p in cong_encoder.parameters() 
                                     if p.grad is not None and p.grad.abs().sum() > 0)
        
        print(f"   Net encoder: {net_params_with_grad}/{len(list(net_encoder.parameters()))} params have gradients")
        print(f"   Congestion encoder: {cong_params_with_grad}/{len(list(cong_encoder.parameters()))} params have gradients")
        return True
    else:
        print("❌ FAIL: Gradients not flowing through shared encoders")
        if not net_encoder_has_grad:
            print("   Net encoder has no gradients!")
        if not cong_encoder_has_grad:
            print("   Congestion encoder has no gradients!")
        return False


def test_parameter_count_reduction():
    """Test that using shared encoders reduces total parameters."""
    print("\n" + "=" * 70)
    print("TEST 4: Parameter Count Reduction")
    print("=" * 70)
    
    hidden_dim = 256
    
    # Create models WITHOUT shared encoders
    diffusion_independent = RoutingDiffusion(
        hidden_dim=hidden_dim,
        max_pips_per_net=50,
        num_heads=4,
        num_layers=2
    )
    
    critic_independent = RoutingCritic(
        hidden_dim=hidden_dim
    )
    
    params_independent = (
        sum(p.numel() for p in diffusion_independent.parameters()) +
        sum(p.numel() for p in critic_independent.parameters())
    )
    
    # Create models WITH shared encoders
    net_encoder, cong_encoder = create_shared_encoders(hidden_dim=hidden_dim)
    
    diffusion_shared = RoutingDiffusion(
        hidden_dim=hidden_dim,
        max_pips_per_net=50,
        shared_net_encoder=net_encoder,
        shared_congestion_encoder=cong_encoder,
        num_heads=4,
        num_layers=2
    )
    
    critic_shared = RoutingCritic(
        hidden_dim=hidden_dim,
        shared_net_encoder=net_encoder,
        shared_congestion_encoder=cong_encoder
    )
    
    # Count parameters (encoders counted once)
    params_shared = (
        sum(p.numel() for p in diffusion_shared.parameters()) +
        sum(p.numel() for p in critic_shared.parameters()) +
        sum(p.numel() for p in net_encoder.parameters()) +
        sum(p.numel() for p in cong_encoder.parameters())
    )
    
    # Encoders are shared, so actual trainable params is less
    encoder_params = (
        sum(p.numel() for p in net_encoder.parameters()) +
        sum(p.numel() for p in cong_encoder.parameters())
    )
    
    params_shared_actual = params_shared - encoder_params  # Don't double count
    
    reduction = params_independent - params_shared_actual
    reduction_pct = (reduction / params_independent) * 100
    
    print(f"Independent encoders: {params_independent:,} parameters")
    print(f"Shared encoders: {params_shared_actual:,} parameters")
    print(f"Reduction: {reduction:,} parameters ({reduction_pct:.1f}%)")
    
    if reduction > 0:
        print("✅ PASS: Shared encoders reduce parameter count")
        return True
    else:
        print("❌ FAIL: No parameter reduction (something wrong)")
        return False


def main():
    """Run all validation tests."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  Shared Encoder Architecture Validation".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")
    
    tests = [
        test_denoiser_no_duplicate_encoder,
        test_shared_encoder_connectivity,
        test_gradient_flow,
        test_parameter_count_reduction
    ]
    
    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR in {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("\nShared encoder fix is working correctly.")
        print("Benefits:")
        print("  1. No duplicate encoders (saves memory)")
        print("  2. Consistent representations between diffusion & critic")
        print("  3. Gradients flow through shared components")
        print("  4. Reduced total parameter count")
        print("\nNext steps:")
        print("  - Train critic with shared encoders enabled")
        print("  - Monitor if loss improves from 0.88 baseline")
        print("  - Consider joint training for further improvements")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("Please review the failures above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

