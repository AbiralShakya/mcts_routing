"""End-to-end pipeline for MCTS routing with diffusion and critic.

This script orchestrates the complete workflow:
1. Generate/load training data
2. Train diffusion model
3. Generate critic data from diffusion trajectories
4. Train critic model
5. Run MCTS inference
6. Evaluate results
"""

import argparse
import yaml
import torch
from pathlib import Path
import logging
import subprocess
import sys

from src.utils.seed import set_seed
from src.utils.logging import setup_logging


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    success = result.returncode == 0
    
    if success:
        print(f"\n✓ {description} completed successfully")
    else:
        print(f"\n✗ {description} failed with return code {result.returncode}")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="End-to-end MCTS routing pipeline")
    parser.add_argument("--config", type=str, required=True, help="Pipeline config file")
    parser.add_argument("--skip_data_gen", action="store_true", help="Skip data generation")
    parser.add_argument("--skip_diffusion_train", action="store_true", help="Skip diffusion training")
    parser.add_argument("--skip_critic_data", action="store_true", help="Skip critic data generation")
    parser.add_argument("--skip_critic_train", action="store_true", help="Skip critic training")
    parser.add_argument("--skip_inference", action="store_true", help="Skip MCTS inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    project_root = Path(config.get('project_root', '.'))
    python_cmd = config.get('python_cmd', sys.executable)
    
    # Step 1: Generate diffusion training data
    if not args.skip_data_gen:
        data_config = config.get('data_generation', {})
        if data_config.get('enabled', True):
            cmd = [
                python_cmd,
                str(project_root / "scripts" / "generate_synthetic_diffusion_data.py"),
                "--output_dir", str(project_root / data_config.get('output_dir', 'data/routing_states')),
                "--num_samples", str(data_config.get('num_samples', 2000)),
                "--grid_size", str(data_config.get('grid_size', 20))
            ]
            if not run_command(cmd, "Generate diffusion training data"):
                logger.error("Data generation failed")
                return
    
    # Step 2: Train diffusion model
    if not args.skip_diffusion_train:
        diff_config = config.get('diffusion_training', {})
        if diff_config.get('enabled', True):
            # Check if using SLURM
            use_slurm = diff_config.get('use_slurm', False)
            if use_slurm:
                cmd = [
                    "sbatch",
                    str(project_root / "scripts" / "slurm" / "train_diffusion.sh")
                ]
                logger.info("Submitting diffusion training job to SLURM")
                logger.info("Check logs/ directory for output")
            else:
                cmd = [
                    python_cmd,
                    str(project_root / "experiments" / "train_routing.py"),
                    "--config", str(project_root / diff_config.get('config', 'configs/training/della_diffusion.yaml')),
                    "--data_dir", str(project_root / diff_config.get('data_dir', 'data/routing_states')),
                    "--checkpoint_dir", str(project_root / diff_config.get('checkpoint_dir', 'checkpoints')),
                    "--seed", str(args.seed)
                ]
            
            if not run_command(cmd, "Train diffusion model"):
                logger.error("Diffusion training failed")
                return
    
    # Step 3: Generate critic training data
    if not args.skip_critic_data:
        critic_data_config = config.get('critic_data_generation', {})
        if critic_data_config.get('enabled', True):
            diffusion_checkpoint = project_root / diff_config.get('checkpoint_dir', 'checkpoints') / critic_data_config.get('diffusion_checkpoint', 'checkpoint_epoch_200.pt')
            
            cmd = [
                python_cmd,
                str(project_root / "scripts" / "generate_critic_data.py"),
                "--model_path", str(diffusion_checkpoint),
                "--output_dir", str(project_root / critic_data_config.get('output_dir', 'data/critic_data')),
                "--num_samples", str(critic_data_config.get('num_samples', 1000)),
                "--use_synthetic" if critic_data_config.get('use_synthetic', False) else ""
            ]
            # Remove empty string if use_synthetic is False
            cmd = [c for c in cmd if c]
            
            if not run_command(cmd, "Generate critic training data"):
                logger.error("Critic data generation failed")
                return
    
    # Step 4: Train critic model
    if not args.skip_critic_train:
        critic_config = config.get('critic_training', {})
        if critic_config.get('enabled', True):
            use_slurm = critic_config.get('use_slurm', False)
            if use_slurm:
                cmd = [
                    "sbatch",
                    str(project_root / "scripts" / "slurm" / "train_critic.sh")
                ]
                logger.info("Submitting critic training job to SLURM")
            else:
                cmd = [
                    python_cmd,
                    str(project_root / "experiments" / "train_critic.py"),
                    "--config", str(project_root / critic_config.get('config', 'configs/training/della_critic.yaml')),
                    "--data_dir", str(project_root / critic_config.get('data_dir', 'data/critic_data')),
                    "--checkpoint_dir", str(project_root / critic_config.get('checkpoint_dir', 'checkpoints/critic')),
                    "--seed", str(args.seed)
                ]
                
                # Add diffusion checkpoint if using shared encoders
                if critic_config.get('use_shared_encoders', False):
                    diff_checkpoint = project_root / diff_config.get('checkpoint_dir', 'checkpoints') / critic_config.get('diffusion_checkpoint', 'checkpoint_epoch_200.pt')
                    # This would need to be passed via config file modification
                    logger.info(f"Note: Update config to set diffusion_checkpoint: {diff_checkpoint}")
            
            if not run_command(cmd, "Train critic model"):
                logger.error("Critic training failed")
                return
    
    # Step 5: Run MCTS inference
    if not args.skip_inference:
        inference_config = config.get('inference', {})
        if inference_config.get('enabled', True):
            diffusion_checkpoint = project_root / diff_config.get('checkpoint_dir', 'checkpoints') / inference_config.get('diffusion_checkpoint', 'checkpoint_epoch_200.pt')
            critic_checkpoint = project_root / critic_config.get('checkpoint_dir', 'checkpoints/critic') / inference_config.get('critic_checkpoint', 'critic_epoch_100.pt')
            
            # Create MCTS config if it doesn't exist
            mcts_config_path = project_root / inference_config.get('mcts_config', 'configs/mcts/base.yaml')
            if not mcts_config_path.exists():
                logger.warning(f"MCTS config not found at {mcts_config_path}, using defaults")
            
            cmd = [
                python_cmd,
                str(project_root / "experiments" / "run_mcts_inference.py"),
                "--config", str(mcts_config_path),
                "--diffusion_checkpoint", str(diffusion_checkpoint),
                "--critic_checkpoint", str(critic_checkpoint) if critic_checkpoint.exists() else "",
                "--netlist_file", str(project_root / inference_config.get('netlist_file', 'data/test_netlist.json')),
                "--output_dir", str(project_root / inference_config.get('output_dir', 'results')),
                "--seed", str(args.seed)
            ]
            # Remove empty strings
            cmd = [c for c in cmd if c]
            
            if not run_command(cmd, "Run MCTS inference"):
                logger.error("MCTS inference failed")
                return
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

