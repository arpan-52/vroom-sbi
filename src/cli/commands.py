"""
CLI commands for VROOM-SBI.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_command(args):
    """Train SBI models."""
    from ..config import Configuration, print_config_summary, validate_config
    from ..training import train_all_models

    # Load config
    config = Configuration.from_yaml(args.config)

    # Validate
    warnings = validate_config(config)
    for w in warnings:
        logger.warning(w)

    print_config_summary(config)

    # Override device if specified
    if args.device:
        config.training.device = args.device

    # Train
    results = train_all_models(config, classifier_only=args.classifier_only)

    logger.info("Training complete!")
    return results


def infer_command(args):
    """Run inference on data."""
    import numpy as np

    from ..config import Configuration
    from ..inference import InferenceEngine

    # Parse Q and U
    Q = np.array([float(x.strip()) for x in args.q.split(",")])
    U = np.array([float(x.strip()) for x in args.u.split(",")])

    if len(Q) != len(U):
        raise ValueError(f"Q and U must have same length: {len(Q)} vs {len(U)}")

    qu_obs = np.concatenate([Q, U])

    # Load config if provided
    config = None
    if args.config and Path(args.config).exists():
        config = Configuration.from_yaml(args.config)

    # Create inference engine
    device = args.device or ("cuda" if config is None else config.training.device)
    engine = InferenceEngine(config=config, model_dir=args.model_dir, device=device)

    # Load models
    engine.load_models(max_components=args.max_components)

    # Run inference
    best_result, all_results = engine.infer(qu_obs, n_samples=args.n_samples)

    # Print results
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(
        f"\nBest model: {best_result.model_type} with {best_result.n_components} component(s)"
    )
    print(f"Log evidence: {best_result.log_evidence:.2f}")

    for i, comp in enumerate(best_result.components):
        print(f"\nComponent {i + 1}:")
        print(f"  RM = {comp.rm_mean:.2f} ± {comp.rm_std:.2f} rad/m²")
        if comp.sigma_phi_mean is not None:
            print(
                f"  σ_φ = {comp.sigma_phi_mean:.2f} ± {comp.sigma_phi_std:.2f} rad/m²"
            )
        if comp.delta_phi_mean is not None:
            print(f"  Δφ = {comp.delta_phi_mean:.2f} ± {comp.delta_phi_std:.2f} rad/m²")

    return best_result


def validate_command(args):
    """Validate trained models."""

    logger.info("Validation not yet implemented in refactored version")
    # TODO: Implement validation


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VROOM-SBI: Simulation-Based Inference for RM Synthesis"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train SBI models")
    train_parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    train_parser.add_argument(
        "--device", type=str, default=None, help="Override device (cuda/cpu)"
    )
    train_parser.add_argument(
        "--classifier-only",
        action="store_true",
        help="Only train classifier (requires existing simulations)",
    )

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--q", type=str, required=True, help="Comma-separated Q values"
    )
    infer_parser.add_argument(
        "--u", type=str, required=True, help="Comma-separated U values"
    )
    infer_parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    infer_parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained models",
    )
    infer_parser.add_argument(
        "--max-components", type=int, default=5, help="Maximum number of components"
    )
    infer_parser.add_argument(
        "--n-samples", type=int, default=10000, help="Number of posterior samples"
    )
    infer_parser.add_argument("--device", type=str, default=None, help="Device to use")
    infer_parser.add_argument(
        "--output", type=str, default="results.png", help="Output figure path"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate trained models")
    validate_parser.add_argument("--config", type=str, default="config.yaml")
    validate_parser.add_argument("--model-dir", type=str, default="models")
    validate_parser.add_argument("--n-tests", type=int, default=20)

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "validate":
        validate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
