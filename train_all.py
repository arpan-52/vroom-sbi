#!/usr/bin/env python
"""
CLI script to train all RM component models.

Usage:
    python train_all.py [--config CONFIG] [--max-components N] [--n-sims N] [--device DEVICE]
"""

import argparse
import yaml
import torch

from src.train import train_all_models


def main():
    parser = argparse.ArgumentParser(
        description="Train neural posterior estimators for RM synthesis"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--max-components',
        type=int,
        default=5,
        help='Maximum number of components to train (default: 5)'
    )
    
    parser.add_argument(
        '--n-sims',
        type=int,
        default=None,
        help='Number of simulations per model (default: from config or 10000)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use: cuda or cpu (default: from config or cuda if available)'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device
    if args.device is not None:
        device = args.device
    elif config.get('training', {}).get('device'):
        device = config['training']['device']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Determine number of simulations
    n_simulations = args.n_sims
    if n_simulations is None:
        n_simulations = config.get('training', {}).get('n_simulations', 10000)
    
    # Get frequency file
    freq_file = config.get('freq_file', 'freq.txt')
    
    print(f"\nTraining settings:")
    print(f"  Frequency file: {freq_file}")
    print(f"  Max components: {args.max_components}")
    print(f"  Simulations per model: {n_simulations}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Device: {device}")
    print()
    
    # Train all models
    posteriors = train_all_models(
        freq_file=freq_file,
        max_components=args.max_components,
        n_simulations=n_simulations,
        config=config,
        device=device,
        save_dir=args.save_dir
    )
    
    print("\nTraining complete! Models saved to:", args.save_dir)
    print("\nYou can now run inference with:")
    print("  python infer.py --q <q_values> --u <u_values> --output results.png")


if __name__ == '__main__':
    main()
