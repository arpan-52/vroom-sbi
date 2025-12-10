#!/usr/bin/env python3
"""
Entrypoint to train all SBI models using settings from config.yaml.

This trains:
1. Posterior models (1-comp and 2-comp) using SBI
2. Model selection classifier (direct classification, no AIC/BIC)

The classifier reuses the same simulations as posterior training,
so there's no extra simulation cost.
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

    # Extract settings for display
    training_cfg = config.get("training", {})
    device = training_cfg.get("device", "cpu")
    freq_file = config.get("freq_file", "freq.txt")
    max_comp = config.get("model_selection", {}).get("max_components", 2)
    n_sim = training_cfg.get("n_simulations", 10000)
    save_dir = training_cfg.get("save_dir", "models")
    
    # Check if classifier-only mode
    classifier_only = config.get("model_selection", {}).get("classifier_only", False)
    
    # Classifier settings
    classifier_cfg = config.get("classifier", {})

    print(f"\nUsing device: {device}")
    print("\nTraining settings:")
    print(f"  Frequency file: {freq_file}")
    print(f"  Max components: {max_comp}")
    print(f"  Simulations per model: {n_sim}")
    print(f"  Save directory: {save_dir}")
    print(f"  Classifier only: {classifier_only}")
    print(f"\nClassifier settings (1D CNN):")
    print(f"  Conv channels: {classifier_cfg.get('conv_channels', [32, 64, 128])}")
    print(f"  Kernel sizes: {classifier_cfg.get('kernel_sizes', [7, 5, 3])}")
    print(f"  Epochs: {classifier_cfg.get('n_epochs', 50)}")
    print(f"  Batch size: {classifier_cfg.get('batch_size', 128)}")

    # Ensure the save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    try:
        # train_all_models with classifier
        results = train_all_models(config, decision_layer_only=classifier_only)
    except TypeError as e:
        print("TypeError while calling train_all_models(config):", file=sys.stderr)
        print(e, file=sys.stderr)
        print("\nHint: ensure src/train.py defines train_all_models(config: dict).", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print("FileNotFoundError:", file=sys.stderr)
        print(e, file=sys.stderr)
        print("\nHint: If using classifier_only=true, simulations must already exist.", file=sys.stderr)
        sys.exit(4)
    except AssertionError as e:
        print("AssertionError during training:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print("An unexpected error occurred during training:", file=sys.stderr)
        raise

    print("\nTraining finished.")
    
    if isinstance(results, dict):
        # Filter only integer keys (model N values) for display
        model_keys = [k for k in results.keys() if isinstance(k, int)]
        if model_keys:
            print(f"Saved models for N = {sorted(model_keys)}")
        
        if "classifier" in results:
            acc = results['classifier'].get('final_val_accuracy', 'N/A')
            if isinstance(acc, (int, float)):
                print(f"Classifier trained with validation accuracy: {acc:.2f}%")
            else:
                print(f"Classifier trained with validation accuracy: {acc}")
    else:
        print("train_all_models returned:", type(results), results)


if __name__ == "__main__":
    main()