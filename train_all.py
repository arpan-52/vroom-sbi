#!/usr/bin/env python3
"""
Entrypoint to train all SBI models using settings from config.yaml.

This version matches the signature used in src/train.py:
    train_all_models(config: dict) -> dict

It loads config.yaml, prints a short summary, ensures the save directory exists,
and calls train_all_models(config).  Any exceptions are reported with a short
hint to help debugging.
"""
import sys
from pathlib import Path
import yaml

from src.train import train_all_models

def main():
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        print("Error: config.yaml not found in current directory.", file=sys.stderr)
        sys.exit(1)

    print("Loading configuration from config.yaml...")
    with cfg_path.open() as f:
        config = yaml.safe_load(f)

    # Friendly defaults if keys are missing
    training_cfg = config.get("training", {})
    device = training_cfg.get("device", "cpu")
    freq_file = config.get("freq_file", "freq.txt")
    max_comp = training_cfg.get("max_components", 5)
    n_sim = training_cfg.get("n_simulations", training_cfg.get("n_sim", 10000))
    save_dir = training_cfg.get("save_dir", training_cfg.get("save_directory", "models"))

    print(f"Using device: {device}\n")
    print("Training settings:")
    print(f"  Frequency file: {freq_file}")
    print(f"  Max components: {max_comp}")
    print(f"  Simulations per model: {n_sim}")
    print(f"  Save directory: {save_dir}")
    print(f"  Device: {device}\n")

    # Ensure the save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    try:
        # train_all_models expects the full config dict
        posteriors = train_all_models(config)
    except TypeError as e:
        print("TypeError while calling train_all_models(config):", file=sys.stderr)
        print(e, file=sys.stderr)
        print("\nHint: ensure src/train.py defines train_all_models(config: dict).", file=sys.stderr)
        sys.exit(2)
    except AssertionError as e:
        print("AssertionError during training:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print("An unexpected error occurred during training:", file=sys.stderr)
        raise

    print("\nTraining finished.")
    if isinstance(posteriors, dict):
        print(f"Saved models for N = {sorted(posteriors.keys())}")
    else:
        print("train_all_models returned:", type(posteriors), posteriors)

if __name__ == "__main__":
    main()