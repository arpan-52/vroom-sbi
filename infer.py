#!/usr/bin/env python
"""
CLI script for RM inference.

Usage:
    python infer.py --q Q1,Q2,... --u U1,U2,... [--output results.png]
"""

import argparse
import yaml
import numpy as np
import torch

from src.inference import RMInference
from src.plots import plot_result


def parse_values(value_string):
    """Parse comma-separated values into numpy array."""
    values = [float(x.strip()) for x in value_string.split(',')]
    return np.array(values)


def main():
    parser = argparse.ArgumentParser(
        description="Run RM inference on observed Q and U data"
    )
    
    parser.add_argument(
        '--q',
        type=str,
        required=True,
        help='Comma-separated Q values (e.g., "0.5,0.3,-0.1,...")'
    )
    
    parser.add_argument(
        '--u',
        type=str,
        required=True,
        help='Comma-separated U values (e.g., "0.1,0.2,0.3,...")'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results.png',
        help='Output figure path (default: results.png)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    parser.add_argument(
        '--max-components',
        type=int,
        default=5,
        help='Maximum number of components to consider (default: 5)'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help='Number of posterior samples (default: 10000)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use: cuda or cpu (default: cuda if available)'
    )
    
    args = parser.parse_args()
    
    # Parse Q and U values
    Q = parse_values(args.q)
    U = parse_values(args.u)
    
    print(f"Q values: {Q}")
    print(f"U values: {U}")
    
    # Check dimensions match
    if len(Q) != len(U):
        raise ValueError(f"Q and U must have same length! Got Q: {len(Q)}, U: {len(U)}")
    
    # Concatenate Q and U
    qu_obs = np.concatenate([Q, U])
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Initialize inference
    print(f"\nInitializing inference...")
    inference = RMInference(model_dir=args.model_dir, device=device)
    
    # Load models
    inference.load_models(max_components=args.max_components)
    
    # Run inference
    print("\nRunning inference...")
    best_result, all_results = inference.infer(qu_obs, n_samples=args.n_samples)
    
    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nBest model: {best_result.n_components} component(s)")
    print(f"Log evidence: {best_result.log_evidence:.2f}")
    print(f"\nInferred parameters:")
    
    for i, comp in enumerate(best_result.components):
        print(f"\n  Component {i+1}:")
        print(f"    RM = {comp.rm_mean:.2f} ± {comp.rm_std:.2f} rad/m²")
        print(f"    q  = {comp.q_mean:.3f} ± {comp.q_std:.3f}")
        print(f"    u  = {comp.u_mean:.3f} ± {comp.u_std:.3f}")
    
    print(f"\n  Noise: σ = {best_result.noise_mean:.4f} ± {best_result.noise_std:.4f}")
    
    print("\n" + "="*60)
    
    # Create visualization
    print(f"\nCreating visualization...")
    freq_file = config.get('freq_file', 'freq.txt')
    
    phi_config = config.get('phi', {})
    phi_range = (phi_config.get('min', -1000), phi_config.get('max', 1000))
    
    plot_result(
        best_result,
        qu_obs,
        freq_file,
        phi_range=phi_range,
        output_path=args.output
    )
    
    print(f"\nInference complete! Results saved to {args.output}")


if __name__ == '__main__':
    main()
