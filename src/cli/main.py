"""VROOM-SBI CLI."""

import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def train_command(args):
    from ..config import Configuration
    from ..training import train_all_models
    config = Configuration.from_yaml(args.config)
    if args.device:
        config.training.device = args.device
    train_all_models(config, classifier_only=args.classifier_only, 
                    auto_optimize=not args.no_auto_optimize)


def infer_command(args):
    from ..inference import InferenceEngine
    import numpy as np
    Q = np.array([float(x) for x in args.q.split(',')])
    U = np.array([float(x) for x in args.u.split(',')])
    engine = InferenceEngine(model_dir=args.model_dir, config_path=args.config,
                            max_components=args.max_components, device=args.device)
    result = engine.infer(Q, U, n_samples=args.n_samples)
    print(result)
    if args.output:
        result.plot(args.output)


def validate_command(args):
    from ..validation.validator import run_validation
    
    print("\n" + "=" * 50)
    print("VROOM-SBI Validation")
    print("=" * 50)
    print(f"Posterior: {args.posterior}")
    print(f"Output: {args.output_dir}")
    print(f"Noise: {args.noise_percent}% of signal")
    print(f"Missing: {args.missing_fraction*100:.0f}% flagged")
    print(f"Sweep points: {args.n_sweep_points}")
    print(f"Individual cases: {args.n_cases}")
    print(f"Device: {args.device}")
    print("=" * 50 + "\n")
    
    run_validation(
        posterior_path=args.posterior,
        output_dir=args.output_dir,
        n_sweep_points=args.n_sweep_points,
        n_cases=args.n_cases,
        noise_percent=args.noise_percent,
        missing_fraction=args.missing_fraction,
        n_samples=args.n_samples,
        device=args.device,
        seed=args.seed,
    )


def push_command(args):
    from ..utils import push_to_huggingface
    push_to_huggingface(model_dir=args.model_dir, repo_id=args.repo_id,
                       token=args.token, private=args.private)


def main():
    parser = argparse.ArgumentParser(description="VROOM-SBI")
    subparsers = parser.add_subparsers(dest='command')
    
    # Train
    train_p = subparsers.add_parser('train')
    train_p.add_argument('--config', default='config.yaml')
    train_p.add_argument('--device', default=None)
    train_p.add_argument('--classifier-only', action='store_true')
    train_p.add_argument('--no-auto-optimize', action='store_true')
    train_p.set_defaults(func=train_command)
    
    # Infer
    infer_p = subparsers.add_parser('infer')
    infer_p.add_argument('--q', required=True)
    infer_p.add_argument('--u', required=True)
    infer_p.add_argument('--config', default=None)
    infer_p.add_argument('--model-dir', default='models')
    infer_p.add_argument('--max-components', type=int, default=5)
    infer_p.add_argument('--n-samples', type=int, default=10000)
    infer_p.add_argument('--device', default=None)
    infer_p.add_argument('--output', default=None)
    infer_p.set_defaults(func=infer_command)
    
    # Validate
    val_p = subparsers.add_parser('validate', help='Run validation with parameter sweeps and case plots')
    val_p.add_argument('--posterior', required=True, help='Path to posterior .pt file')
    val_p.add_argument('--output-dir', default='validation_results', help='Output directory')
    val_p.add_argument('--noise-percent', type=float, default=10.0, 
                      help='Noise as %% of signal amplitude (default: 10)')
    val_p.add_argument('--missing-fraction', type=float, default=0.1,
                      help='Fraction of channels to flag (default: 0.1)')
    val_p.add_argument('--n-sweep-points', type=int, default=20,
                      help='Points per parameter sweep (default: 20)')
    val_p.add_argument('--n-cases', type=int, default=10,
                      help='Number of individual test cases (default: 10)')
    val_p.add_argument('--n-samples', type=int, default=5000,
                      help='Posterior samples per inference (default: 5000)')
    val_p.add_argument('--device', default='auto')
    val_p.add_argument('--seed', type=int, default=42)
    val_p.set_defaults(func=validate_command)
    
    # Push
    push_p = subparsers.add_parser('push')
    push_p.add_argument('--model-dir', default='models')
    push_p.add_argument('--repo-id', required=True)
    push_p.add_argument('--token', default=None)
    push_p.add_argument('--private', action='store_true')
    push_p.set_defaults(func=push_command)
    
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == '__main__':
    main()
