"""Run MCTS inference with trained diffusion and critic models."""

import argparse
import yaml
import torch
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json

from src.mcts.search import MCTSRouter, RouterConfig, iterate
from src.mcts.node import RoutingNode
from src.mcts.tree import MCTSTree
from src.diffusion.model import RoutingDiffusion, create_routing_diffusion
from src.diffusion.sampler import initialize_routing_state
from src.critic.gnn import RoutingCritic
from src.bridge.router import NextPNRRouter
from src.integration.nextpnr.reader import NextPNRReader
from src.core.routing.grid import Grid
from src.core.routing.netlist import Netlist
from src.utils.seed import set_seed
from src.utils.logging import setup_logging


def load_models(
    diffusion_checkpoint: str,
    critic_checkpoint: Optional[str],
    config: Dict[str, Any],
    device: str = "cuda"
) -> tuple:
    """Load diffusion and critic models.
    
    Args:
        diffusion_checkpoint: Path to diffusion model checkpoint
        critic_checkpoint: Path to critic model checkpoint (optional)
        config: Configuration dictionary
        device: Device to load models on
    
    Returns:
        Tuple of (diffusion_model, critic_model)
    """
    # Load diffusion model
    print(f"Loading diffusion model from {diffusion_checkpoint}")
    diff_checkpoint = torch.load(diffusion_checkpoint, map_location=device)
    diff_config = config.get('diffusion', {}).get('model', {})
    
    # Create model
    diffusion_model = create_routing_diffusion(diff_config)
    if 'model_state_dict' in diff_checkpoint:
        diffusion_model.load_state_dict(diff_checkpoint['model_state_dict'])
    elif 'model' in diff_checkpoint:
        diffusion_model.load_state_dict(diff_checkpoint['model'])
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()
    print("Diffusion model loaded")
    
    # Load critic model (optional)
    critic_model = None
    if critic_checkpoint and Path(critic_checkpoint).exists():
        print(f"Loading critic model from {critic_checkpoint}")
        crit_checkpoint = torch.load(critic_checkpoint, map_location=device)
        crit_config = config.get('critic', {}).get('model', {})
        
        critic_model = RoutingCritic(
            node_dim=crit_config.get('node_dim', 64),
            edge_dim=crit_config.get('edge_dim', 32),
            hidden_dim=crit_config.get('hidden_dim', 128),
            num_layers=crit_config.get('num_layers', 4),
            dropout=crit_config.get('dropout', 0.1),
            num_timesteps=crit_config.get('num_timesteps', 1000)  # For time-aware evaluation
        )
        
        if 'model_state_dict' in crit_checkpoint:
            critic_model.load_state_dict(crit_checkpoint['model_state_dict'])
        elif 'critic' in crit_checkpoint:
            critic_model.load_state_dict(crit_checkpoint['critic'])
        critic_model = critic_model.to(device)
        critic_model.eval()
        print("Critic model loaded")
    else:
        print("Warning: No critic model provided, MCTS will run without pruning")
    
    return diffusion_model, critic_model


def run_mcts_inference(
    diffusion_model: RoutingDiffusion,
    critic_model: Optional[RoutingCritic],
    netlist: Netlist,
    grid: Grid,
    config: Dict[str, Any],
    device: str = "cuda"
) -> Dict[str, Any]:
    """Run MCTS inference on a single netlist.
    
    Args:
        diffusion_model: Trained diffusion model
        critic_model: Trained critic model (optional)
        netlist: Netlist to route
        grid: Grid structure
        config: MCTS configuration
        device: Device to run on
    
    Returns:
        Dictionary with results
    """
    # Create router config
    router_config = RouterConfig(
        num_timesteps=config.get('num_timesteps', 1000),
        hidden_dim=config.get('hidden_dim', 256),
        ucb_c=config.get('ucb_c', 1.41),
        max_iterations=config.get('max_iterations', 1000),
        critic_threshold=config.get('critic_threshold', 0.3),
        device=device
    )
    
    # Create nextpnr router
    nextpnr_path = config.get('nextpnr_path', 'nextpnr-xilinx')
    router = NextPNRRouter(nextpnr_path=nextpnr_path)
    
    # Create MCTS router
    mcts_router = MCTSRouter(
        diffusion=diffusion_model,
        critic=critic_model,
        router=router,
        grid=grid,
        netlist=netlist,
        config=router_config,
        device=device
    )
    
    # Run MCTS
    max_iterations = config.get('max_iterations', 1000)
    routing_result = mcts_router.route(max_iterations=max_iterations)
    
    # Get statistics
    stats = mcts_router.get_statistics()
    
    return {
        'routing': routing_result,
        'statistics': stats,
        'num_nets': len(netlist.nets),
        'grid_size': grid.get_size()
    }


def main():
    parser = argparse.ArgumentParser(description="Run MCTS inference")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--diffusion_checkpoint", type=str, required=True, help="Diffusion model checkpoint")
    parser.add_argument("--critic_checkpoint", type=str, default=None, help="Critic model checkpoint (optional)")
    parser.add_argument("--netlist_file", type=str, required=True, help="Netlist file to route")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line args
    if args.device:
        config['device'] = args.device
    
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    
    # Load models
    diffusion_model, critic_model = load_models(
        args.diffusion_checkpoint,
        args.critic_checkpoint,
        config,
        device=device
    )
    
    # Load netlist
    logger.info(f"Loading netlist from {args.netlist_file}")
    reader = NextPNRReader()
    grid, netlist, placement, routing_state = reader.read_all(args.netlist_file)
    
    if not netlist or len(netlist.nets) == 0:
        logger.error("Failed to load netlist or netlist is empty")
        return
    
    logger.info(f"Netlist loaded: {len(netlist.nets)} nets, grid size: {grid.get_size()}")
    
    # Run MCTS inference
    logger.info("Running MCTS inference...")
    mcts_config = config.get('mcts', {})
    results = run_mcts_inference(
        diffusion_model,
        critic_model,
        netlist,
        grid,
        mcts_config,
        device=device
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"mcts_result_{Path(args.netlist_file).stem}.json"
    
    # Convert results to JSON-serializable format
    serializable_results = {
        'statistics': results['statistics'],
        'num_nets': results['num_nets'],
        'grid_size': results['grid_size'],
        'routing_success': len(results['routing']) > 0,
        'num_routed_nets': len(results['routing'])
    }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Statistics: {results['statistics']}")
    logger.info(f"Routed {len(results['routing'])}/{results['num_nets']} nets")


if __name__ == "__main__":
    main()

