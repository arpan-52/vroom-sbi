"""
VROOM-SBI Command Line Interface.
"""

import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def train_command(args):
    """Train VROOM-SBI models."""
    from ..config import Configuration
    from ..training import train_all_models
    
    config = Configuration.from_yaml(args.config)
    
    if args.device:
        config.training.device = args.device
    
    train_all_models(config, classifier_only=args.classifier_only, 
                    auto_optimize=not args.no_auto_optimize)


def infer_command(args):
    """Run inference on Q/U data."""
    from ..inference import InferenceEngine
    import numpy as np
    
    Q = np.array([float(x) for x in args.q.split(',')])
    U = np.array([float(x) for x in args.u.split(',')])
    
    engine = InferenceEngine(
        model_dir=args.model_dir,
        config_path=args.config,
        max_components=args.max_components,
        device=args.device
    )
    
    result = engine.infer(Q, U, n_samples=args.n_samples)
    print(result)
    
    if args.output:
        result.plot(args.output)


def validate_command(args):
    """Run VROOM-SBI validation."""
    from ..validation.validator import run_validation
    
    print("\n" + "=" * 50)
    print("VROOM-SBI Validation")
    print("=" * 50)
    print(f"Posterior: {args.posterior}")
    print(f"Output: {args.output_dir}")
    print(f"Cases: {args.n_cases}")
    print(f"Device: {args.device}")
    print("=" * 50 + "\n")
    
    run_validation(
        posterior_path=args.posterior,
        output_dir=args.output_dir,
        n_cases=args.n_cases,
        n_samples=args.n_samples,
        device=args.device,
        seed=args.seed,
    )


def push_command(args):
    """Push models to HuggingFace."""
    from ..utils import push_to_huggingface
    
    push_to_huggingface(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private
    )


def main():
    parser = argparse.ArgumentParser(
        description="VROOM-SBI: Simulation-Based Inference for RM Synthesis",
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--config', type=str, default='config.yaml')
    train_parser.add_argument('--device', type=str, default=None)
    train_parser.add_argument('--classifier-only', action='store_true')
    train_parser.add_argument('--no-auto-optimize', action='store_true')
    train_parser.set_defaults(func=train_command)
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--q', type=str, required=True)
    infer_parser.add_argument('--u', type=str, required=True)
    infer_parser.add_argument('--config', type=str, default=None)
    infer_parser.add_argument('--model-dir', type=str, default='models')
    infer_parser.add_argument('--max-components', type=int, default=5)
    infer_parser.add_argument('--n-samples', type=int, default=10000)
    infer_parser.add_argument('--device', type=str, default=None)
    infer_parser.add_argument('--output', type=str, default=None)
    infer_parser.set_defaults(func=infer_command)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Run validation')
    validate_parser.add_argument('--posterior', type=str, required=True,
                                help='Path to posterior .pt file')
    validate_parser.add_argument('--output-dir', type=str, default='validation_results',
                                help='Output directory')
    validate_parser.add_argument('--n-cases', type=int, default=20,
                                help='Number of test cases')
    validate_parser.add_argument('--n-samples', type=int, default=5000,
                                help='Posterior samples per case')
    validate_parser.add_argument('--device', type=str, default='auto')
    validate_parser.add_argument('--seed', type=int, default=42)
    validate_parser.set_defaults(func=validate_command)
    
    # Push command
    push_parser = subparsers.add_parser('push', help='Push to HuggingFace')
    push_parser.add_argument('--model-dir', type=str, default='models')
    push_parser.add_argument('--repo-id', type=str, required=True)
    push_parser.add_argument('--token', type=str, default=None)
    push_parser.add_argument('--private', action='store_true')
    push_parser.set_defaults(func=push_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == '__main__':
    main()
