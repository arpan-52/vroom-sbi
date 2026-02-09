#!/usr/bin/env python3
"""
Run RM inference on observed Q and U data.

Usage:
    python infer.py --q Q1,Q2,... --u U1,U2,...
"""
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Configuration
from src.inference import InferenceEngine


def parse_values(value_string):
    """Parse comma-separated values into numpy array."""
    return np.array([float(x.strip()) for x in value_string.split(',')])


def main():
    parser = argparse.ArgumentParser(description="Run RM inference")
    parser.add_argument('--q', type=str, required=True, help='Comma-separated Q values')
    parser.add_argument('--u', type=str, required=True, help='Comma-separated U values')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--max-components', type=int, default=5)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output', type=str, default='results.png')
    args = parser.parse_args()
    
    Q = parse_values(args.q)
    U = parse_values(args.u)
    
    if len(Q) != len(U):
        raise ValueError(f"Q and U must have same length: {len(Q)} vs {len(U)}")
    
    qu_obs = np.concatenate([Q, U])
    
    config = None
    if Path(args.config).exists():
        config = Configuration.from_yaml(args.config)
    
    device = args.device or ('cuda' if config is None else config.training.device)
    
    engine = InferenceEngine(config=config, model_dir=args.model_dir, device=device)
    engine.load_models(max_components=args.max_components)
    
    best_result, all_results = engine.infer(qu_obs, n_samples=args.n_samples)
    
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"\nBest model: {best_result.model_type} with {best_result.n_components} component(s)")
    print(f"Log evidence: {best_result.log_evidence:.2f}")
    
    for i, comp in enumerate(best_result.components):
        print(f"\nComponent {i+1}:")
        print(f"  RM = {comp.rm_mean:.2f} ± {comp.rm_std:.2f} rad/m²")
        if comp.sigma_phi_mean is not None:
            print(f"  σ_φ = {comp.sigma_phi_mean:.2f} ± {comp.sigma_phi_std:.2f} rad/m²")
        if comp.delta_phi_mean is not None:
            print(f"  Δφ = {comp.delta_phi_mean:.2f} ± {comp.delta_phi_std:.2f} rad/m²")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
