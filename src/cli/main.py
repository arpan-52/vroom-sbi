"""
Main CLI for VROOM-SBI.

Provides commands for training and inference.
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_command(args):
    """Train models command."""
    from ..config import Configuration, validate_config, print_config_summary
    from ..config import auto_configure, detect_hardware
    from ..training import train_all_models
    
    # Load config
    config = Configuration.from_yaml(args.config)
    
    # Validate
    warnings = validate_config(config)
    for w in warnings:
        logger.warning(w)
    
    # Auto-optimize based on hardware (unless --no-auto-optimize)
    if not args.no_auto_optimize:
        print("\n" + "=" * 60)
        print("AUTO-DETECTING HARDWARE AND OPTIMIZING SETTINGS")
        print("=" * 60)
        config = auto_configure(config, verbose=True)
    else:
        print("\nUsing config file settings (auto-optimization disabled)")
    
    # Override device if specified via CLI
    if args.device:
        config.training.device = args.device
    
    # Print final config summary
    print_config_summary(config)
    
    # Train
    results = train_all_models(config, classifier_only=args.classifier_only)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


def infer_command(args):
    """Run inference command."""
    import numpy as np
    from ..config import Configuration
    from ..inference import InferenceEngine
    
    # Parse Q and U values
    Q = np.array([float(x.strip()) for x in args.q.split(',')])
    U = np.array([float(x.strip()) for x in args.u.split(',')])
    
    if len(Q) != len(U):
        logger.error(f"Q and U must have same length! Got Q: {len(Q)}, U: {len(U)}")
        sys.exit(1)
    
    qu_obs = np.concatenate([Q, U])
    
    # Load config
    config = Configuration.from_yaml(args.config) if args.config else None
    
    # Determine device
    device = args.device or (config.training.device if config else 'cuda')
    
    # Initialize inference
    engine = InferenceEngine(
        config=config,
        model_dir=args.model_dir,
        device=device,
    )
    
    # Load models
    engine.load_models(max_components=args.max_components)
    
    # Run inference
    best_result, all_results = engine.infer(qu_obs, n_samples=args.n_samples)
    
    # Print results
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"\nBest model: {best_result.model_type} with {best_result.n_components} component(s)")
    print(f"Log evidence: {best_result.log_evidence:.2f}")
    
    for i, comp in enumerate(best_result.components):
        print(f"\n  Component {i+1}:")
        print(f"    RM = {comp.rm_mean:.2f} ± {comp.rm_std:.2f} rad/m²")
        print(f"    q  = {comp.q_mean:.3f} ± {comp.q_std:.3f}")
        print(f"    u  = {comp.u_mean:.3f} ± {comp.u_std:.3f}")
    
    print("\n" + "=" * 60)
    
    # Create visualization if requested
    if args.output:
        from ..utils.plots import plot_inference_result
        plot_inference_result(
            best_result, qu_obs,
            freq_file=config.freq_file if config else "freq.txt",
            output_path=args.output,
        )
        print(f"\nVisualization saved to {args.output}")


def validate_command(args):
    """Comprehensive validation with publication-quality plots."""
    from ..validation.validator import run_comprehensive_validation
    
    # Determine if we should run RM-Tools
    run_rmtools = args.run_rmtools and not args.no_rmtools
    
    print("\n" + "=" * 60)
    print("VROOM-SBI COMPREHENSIVE VALIDATION")
    print("=" * 60)
    print(f"Posterior: {args.posterior}")
    print(f"Output: {args.output_dir}")
    print(f"Parameter sweep points: {args.n_param_points}")
    print(f"Noise: {args.noise_min} to {args.noise_max} ({args.noise_steps} steps)")
    print(f"Missing: {args.missing_min} to {args.missing_max} ({args.missing_steps} steps)")
    print(f"Grid repeats: {args.n_grid_repeats}")
    print(f"Individual cases: {args.n_cases}")
    print(f"RM-Tools model: {args.rmtools_model}")
    print(f"Run RM-Tools: {run_rmtools}")
    print(f"Device: {args.device}")
    print("=" * 60 + "\n")
    
    # Run validation
    validator = run_comprehensive_validation(
        posterior_path=args.posterior,
        output_dir=args.output_dir,
        rmtools_model=args.rmtools_model,
        n_param_points=args.n_param_points,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        noise_steps=args.noise_steps,
        missing_min=args.missing_min,
        missing_max=args.missing_max,
        missing_steps=args.missing_steps,
        n_grid_repeats=args.n_grid_repeats,
        n_individual_cases=args.n_cases,
        n_samples=args.n_samples,
        run_rmtools=run_rmtools,
        device=args.device,
        seed=args.seed,
    )
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print("\nOutput folders:")
    print("  parameter_sweeps/      - Parameter recovery plots")
    print("  noise_missing_analysis/ - RMSE heatmaps")
    print("  individual_cases/      - Detailed case studies")
    print("  posteriors/            - Corner plots")
    print("  timing/                - Speed comparison")
    print("  data/                  - Raw results JSON")
    print("=" * 60)


def push_command(args):
    """Push models to HuggingFace Hub."""
    import os
    from ..utils.huggingface import push_to_hub
    
    # Get token from environment or argument
    token = args.token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    
    if not token:
        logger.error("HuggingFace token required. Set HF_TOKEN environment variable or use --token")
        sys.exit(1)
    
    push_to_hub(
        model_dir=Path(args.model_dir),
        repo_id=args.repo_id,
        token=token,
        private=args.private,
    )


def cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VROOM-SBI: Simulation-Based Inference for RM Synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--config', type=str, default='config.yaml',
                             help='Configuration file')
    train_parser.add_argument('--device', type=str, default=None,
                             help='Device (cuda/cpu)')
    train_parser.add_argument('--classifier-only', action='store_true',
                             help='Only train classifier')
    train_parser.add_argument('--no-auto-optimize', action='store_true',
                             help='Disable automatic hardware optimization (use config values)')
    train_parser.set_defaults(func=train_command)
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--q', type=str, required=True,
                             help='Comma-separated Q values')
    infer_parser.add_argument('--u', type=str, required=True,
                             help='Comma-separated U values')
    infer_parser.add_argument('--config', type=str, default=None,
                             help='Configuration file')
    infer_parser.add_argument('--model-dir', type=str, default='models',
                             help='Models directory')
    infer_parser.add_argument('--max-components', type=int, default=5,
                             help='Maximum components')
    infer_parser.add_argument('--n-samples', type=int, default=10000,
                             help='Posterior samples')
    infer_parser.add_argument('--device', type=str, default=None,
                             help='Device')
    infer_parser.add_argument('--output', type=str, default=None,
                             help='Output figure path')
    infer_parser.set_defaults(func=infer_command)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Comprehensive validation with VROOM vs RM-Tools comparison')
    validate_parser.add_argument('--posterior', type=str, required=True,
                                help='Path to posterior .pt file')
    validate_parser.add_argument('--output-dir', type=str, default='validation_results',
                                help='Output directory for results and plots')
    validate_parser.add_argument('--n-param-points', type=int, default=20,
                                help='Number of points for parameter sweeps')
    validate_parser.add_argument('--n-samples', type=int, default=5000,
                                help='Number of posterior samples per test case')
    validate_parser.add_argument('--noise-min', type=float, default=0.001,
                                help='Minimum noise level')
    validate_parser.add_argument('--noise-max', type=float, default=0.5,
                                help='Maximum noise level')
    validate_parser.add_argument('--noise-steps', type=int, default=10,
                                help='Number of noise levels')
    validate_parser.add_argument('--missing-min', type=float, default=0.0,
                                help='Minimum missing fraction')
    validate_parser.add_argument('--missing-max', type=float, default=0.5,
                                help='Maximum missing fraction')
    validate_parser.add_argument('--missing-steps', type=int, default=10,
                                help='Number of missing fraction levels')
    validate_parser.add_argument('--n-grid-repeats', type=int, default=5,
                                help='Repeats per grid cell')
    validate_parser.add_argument('--n-cases', type=int, default=10,
                                help='Number of individual cases for deep dive')
    validate_parser.add_argument('--rmtools-model', type=str, default='1',
                                help='RM-Tools model number (e.g., 1, 11, 111)')
    validate_parser.add_argument('--run-rmtools', action='store_true',
                                help='Run RM-Tools QUfit comparison')
    validate_parser.add_argument('--no-rmtools', action='store_true',
                                help='Skip RM-Tools comparison')
    validate_parser.add_argument('--device', type=str, default='auto',
                                help='Device for inference (auto, cuda, or cpu)')
    validate_parser.add_argument('--seed', type=int, default=42,
                                help='Random seed for reproducibility')
    validate_parser.set_defaults(func=validate_command)
    
    # Push to HuggingFace command
    push_parser = subparsers.add_parser('push', help='Push models to HuggingFace')
    push_parser.add_argument('--model-dir', type=str, default='models')
    push_parser.add_argument('--repo-id', type=str, required=True,
                            help='HuggingFace repo ID (e.g., username/repo-name)')
    push_parser.add_argument('--token', type=str, default=None,
                            help='HuggingFace token (or set HF_TOKEN env var)')
    push_parser.add_argument('--private', action='store_true',
                            help='Make repo private')
    push_parser.set_defaults(func=push_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
