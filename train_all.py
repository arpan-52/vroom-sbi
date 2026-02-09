#!/usr/bin/env python3
"""
Train all VROOM-SBI models.

Usage:
    python train_all.py
    python train_all.py --config config.yaml
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Configuration, validate_config, print_config_summary
from src.training import train_all_models


def main():
    parser = argparse.ArgumentParser(description="Train all VROOM-SBI models")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--classifier-only', action='store_true')
    args = parser.parse_args()
    
    config = Configuration.from_yaml(args.config)
    if args.device:
        config.training.device = args.device
    
    warnings = validate_config(config)
    for w in warnings:
        print(f"Warning: {w}")
    
    print_config_summary(config)
    results = train_all_models(config, classifier_only=args.classifier_only)
    
    print("\nTraining complete!")
    print(f"Models saved to: {config.training.save_dir}")


if __name__ == "__main__":
    main()
