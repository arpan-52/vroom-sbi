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
    """Validate posteriors with publication-quality analysis."""
    from ..validation import run_validation
    
    # Parse noise levels
    noise_levels = [float(x.strip()) for x in args.noise_levels.split(',')]
    
    # Parse missing fractions
    missing_fractions = [float(x.strip()) for x in args.missing_fractions.split(',')]
    
    # Build prior overrides from CLI args
    prior_overrides = {}
    if args.rm_min is not None:
        prior_overrides['rm_min'] = args.rm_min
    if args.rm_max is not None:
        prior_overrides['rm_max'] = args.rm_max
    if args.amp_min is not None:
        prior_overrides['amp_min'] = args.amp_min
    if args.amp_max is not None:
        prior_overrides['amp_max'] = args.amp_max
    if args.chi0_min is not None:
        prior_overrides['chi0_min'] = args.chi0_min
    if args.chi0_max is not None:
        prior_overrides['chi0_max'] = args.chi0_max
    
    print("\n" + "=" * 60)
    print("VROOM-SBI VALIDATION")
    print("=" * 60)
    print(f"Posterior: {args.posterior}")
    print(f"Output: {args.output_dir}")
    print(f"Tests per condition: {args.n_tests}")
    print(f"Noise levels: {noise_levels}")
    print(f"Missing fractions: {missing_fractions}")
    if prior_overrides:
        print(f"Prior overrides: {prior_overrides}")
    print("=" * 60 + "\n")
    
    # Run validation
    validator = run_validation(
        posterior_path=args.posterior,
        output_dir=args.output_dir,
        n_tests=args.n_tests,
        noise_levels=noise_levels,
        missing_fractions=missing_fractions,
        n_posterior_samples=args.n_samples,
        prior_overrides=prior_overrides if prior_overrides else None,
        compare_rmtools=args.compare_rmtools,
        device=args.device,
        seed=args.seed,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Model: {validator.model_type} (N={validator.n_components})")
    print(f"Test cases: {validator.metrics.n_tests}")
    print(f"Mean inference time: {validator.metrics.mean_inference_time * 1000:.1f} ms")
    print(f"\nResults saved to: {args.output_dir}")
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
    validate_parser = subparsers.add_parser('validate', help='Validate posteriors with publication-quality analysis')
    validate_parser.add_argument('--posterior', type=str, required=True,
                                help='Path to posterior .pt file')
    validate_parser.add_argument('--output-dir', type=str, default='validation_results',
                                help='Output directory for results and plots')
    validate_parser.add_argument('--n-tests', type=int, default=100,
                                help='Number of test cases per noise/missing combination')
    validate_parser.add_argument('--n-samples', type=int, default=5000,
                                help='Number of posterior samples per test case')
    validate_parser.add_argument('--noise-levels', type=str, default='0.01,0.05,0.1',
                                help='Comma-separated noise levels to test')
    validate_parser.add_argument('--missing-fractions', type=str, default='0.0,0.1,0.3',
                                help='Comma-separated missing channel fractions')
    validate_parser.add_argument('--rm-min', type=float, default=None,
                                help='Override RM prior minimum')
    validate_parser.add_argument('--rm-max', type=float, default=None,
                                help='Override RM prior maximum')
    validate_parser.add_argument('--amp-min', type=float, default=None,
                                help='Override amplitude prior minimum')
    validate_parser.add_argument('--amp-max', type=float, default=None,
                                help='Override amplitude prior maximum')
    validate_parser.add_argument('--chi0-min', type=float, default=None,
                                help='Override chi0 prior minimum')
    validate_parser.add_argument('--chi0-max', type=float, default=None,
                                help='Override chi0 prior maximum')
    validate_parser.add_argument('--compare-rmtools', action='store_true',
                                help='Compare with RM-Tools QUfit')
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
