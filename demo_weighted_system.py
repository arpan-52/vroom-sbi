#!/usr/bin/env python3
"""
Demo script showing how to use the weighted two-layer neural network system.

This script demonstrates:
1. Creating synthetic observations with different weight patterns
2. Running inference with the two-layer system
3. Comparing results across different scenarios

Note: Run train_all.py first to train the models.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from src.simulator import RMSimulator
from src.inference import RMInference
from src.augmentation import (
    augment_weights_scattered,
    augment_weights_contiguous_gap,
    augment_weights_noise_variation
)


def create_demo_observations(freq_file: str):
    """
    Create synthetic observations with different weight scenarios.
    
    Returns
    -------
    dict
        Dictionary with scenario name -> (qu_obs, weights, true_params)
    """
    # Create simulators
    sim_1comp = RMSimulator(freq_file, 1)
    sim_2comp = RMSimulator(freq_file, 2)
    
    observations = {}
    
    # Scenario 1: 1-component, clean data
    theta_1 = np.array([120.0, 0.6, 0.5, 0.002])  # RM, amp, chi0, noise
    weights_clean = np.ones(sim_1comp.n_freq)
    qu_1_clean = sim_1comp(theta_1, weights=weights_clean)
    observations["1-comp Clean"] = {
        "qu_obs": qu_1_clean,
        "weights": weights_clean,
        "true_params": theta_1,
        "n_components": 1,
        "description": "Single component with perfect data quality"
    }
    
    # Scenario 2: 1-component with RFI gap
    weights_gap = augment_weights_contiguous_gap(weights_clean, gap_prob=1.0, min_gap=8, max_gap=12)
    qu_1_gap = sim_1comp(theta_1, weights=weights_gap)
    observations["1-comp RFI Gap"] = {
        "qu_obs": qu_1_gap,
        "weights": weights_gap,
        "true_params": theta_1,
        "n_components": 1,
        "description": "Single component with RFI causing contiguous gap"
    }
    
    # Scenario 3: 2-component, clean data
    theta_2 = np.array([180.0, 0.5, 0.7, -60.0, 0.4, 1.1, 0.002])  # RM1, amp1, chi01, RM2, amp2, chi02, noise
    qu_2_clean = sim_2comp(theta_2, weights=weights_clean)
    observations["2-comp Clean"] = {
        "qu_obs": qu_2_clean,
        "weights": weights_clean,
        "true_params": theta_2,
        "n_components": 2,
        "description": "Two components with perfect data quality"
    }
    
    # Scenario 4: 2-component with scattered missing
    weights_scattered = augment_weights_scattered(weights_clean, missing_prob=0.2)
    qu_2_scattered = sim_2comp(theta_2, weights=weights_scattered)
    observations["2-comp Scattered Missing"] = {
        "qu_obs": qu_2_scattered,
        "weights": weights_scattered,
        "true_params": theta_2,
        "n_components": 2,
        "description": "Two components with randomly scattered missing channels"
    }
    
    # Scenario 5: 1-component with varying noise
    weights_varied = augment_weights_noise_variation(weights_clean, variation_scale=0.4)
    qu_1_varied = sim_1comp(theta_1, weights=weights_varied)
    observations["1-comp Varying Noise"] = {
        "qu_obs": qu_1_varied,
        "weights": weights_varied,
        "true_params": theta_1,
        "n_components": 1,
        "description": "Single component with varying channel quality"
    }
    
    return observations


def plot_weight_patterns(observations, output_path="weight_patterns.png"):
    """
    Visualize the different weight patterns.
    """
    n_scenarios = len(observations)
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(12, 2*n_scenarios))
    
    if n_scenarios == 1:
        axes = [axes]
    
    for idx, (name, obs) in enumerate(observations.items()):
        ax = axes[idx]
        weights = obs['weights']
        channels = np.arange(len(weights))
        
        # Plot weights as bars
        colors = ['green' if w > 0.8 else 'orange' if w > 0.3 else 'red' if w > 0 else 'black' 
                  for w in weights]
        ax.bar(channels, weights, color=colors, alpha=0.7, width=1.0)
        
        ax.set_ylabel('Weight')
        ax.set_ylim([0, 1.1])
        ax.set_title(f"{name}: {obs['description']}")
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Good (w>0.8)'),
            Patch(facecolor='orange', alpha=0.7, label='Moderate (0.3<w<0.8)'),
            Patch(facecolor='red', alpha=0.7, label='Poor (0<w<0.3)'),
            Patch(facecolor='black', alpha=0.7, label='Missing (w=0)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Channel Index')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nWeight pattern plot saved to: {output_path}")


def run_demo(config_path="config.yaml", model_dir="models"):
    """
    Run the demo.
    """
    print("="*80)
    print("WEIGHTED TWO-LAYER SYSTEM DEMO")
    print("="*80)
    
    # Check if models exist
    if not Path(model_dir).exists():
        print(f"\nError: Model directory '{model_dir}' not found!")
        print("Please train the models first using: python train_all.py")
        return
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    freq_file = config.get("freq_file", "freq.txt")
    device = config.get("training", {}).get("device", "cpu")
    
    # Create demo observations
    print("\nCreating demo observations with different weight patterns...")
    observations = create_demo_observations(freq_file)
    
    # Plot weight patterns
    plot_weight_patterns(observations)
    
    # Initialize inference
    print("\nInitializing two-layer inference system...")
    inference = RMInference(model_dir=model_dir, device=device, use_decision_layer=True)
    inference.load_models(max_components=2)
    
    # Run inference on all scenarios
    print("\n" + "="*80)
    print("RUNNING INFERENCE ON ALL SCENARIOS")
    print("="*80)
    
    results = []
    
    for name, obs in observations.items():
        print(f"\n{'='*80}")
        print(f"Scenario: {name}")
        print(f"{'='*80}")
        print(f"Description: {obs['description']}")
        print(f"True n_components: {obs['n_components']}")
        
        # Show weight statistics
        weights = obs['weights']
        n_missing = np.sum(weights == 0)
        print(f"Missing channels: {n_missing}/{len(weights)} ({100*n_missing/len(weights):.1f}%)")
        print(f"Weight range: [{np.min(weights):.3f}, {np.max(weights):.3f}]")
        
        # Run inference
        best_result, all_results = inference.infer(
            obs['qu_obs'],
            weights=obs['weights'],
            n_samples=5000
        )
        
        # Compare with truth
        predicted_n = best_result.n_components
        true_n = obs['n_components']
        
        print(f"\nResult:")
        print(f"  Predicted: {predicted_n} component(s)")
        print(f"  Ground Truth: {true_n} component(s)")
        print(f"  Decision: {'✓ CORRECT' if predicted_n == true_n else '✗ INCORRECT'}")
        
        if predicted_n == true_n:
            if true_n == 1:
                true_rm = obs['true_params'][0]
                pred_rm = best_result.components[0].rm_mean
                error = abs(pred_rm - true_rm)
                print(f"\n  RM Recovery:")
                print(f"    True:      {true_rm:.1f} rad/m²")
                print(f"    Predicted: {pred_rm:.1f} ± {best_result.components[0].rm_std:.1f} rad/m²")
                print(f"    Error:     {error:.1f} rad/m² ({100*error/abs(true_rm):.1f}%)")
            elif true_n == 2:
                true_rm1 = obs['true_params'][0]
                true_rm2 = obs['true_params'][3]
                pred_rm1 = best_result.components[0].rm_mean
                pred_rm2 = best_result.components[1].rm_mean
                
                # Match components
                err1 = abs(pred_rm1 - true_rm1) + abs(pred_rm2 - true_rm2)
                err2 = abs(pred_rm1 - true_rm2) + abs(pred_rm2 - true_rm1)
                
                if err1 < err2:
                    print(f"\n  RM Recovery:")
                    print(f"    Component 1:")
                    print(f"      True:      {true_rm1:.1f} rad/m²")
                    print(f"      Predicted: {pred_rm1:.1f} ± {best_result.components[0].rm_std:.1f} rad/m²")
                    print(f"    Component 2:")
                    print(f"      True:      {true_rm2:.1f} rad/m²")
                    print(f"      Predicted: {pred_rm2:.1f} ± {best_result.components[1].rm_std:.1f} rad/m²")
                else:
                    print(f"\n  RM Recovery:")
                    print(f"    Component 1:")
                    print(f"      True:      {true_rm1:.1f} rad/m²")
                    print(f"      Predicted: {pred_rm2:.1f} ± {best_result.components[1].rm_std:.1f} rad/m²")
                    print(f"    Component 2:")
                    print(f"      True:      {true_rm2:.1f} rad/m²")
                    print(f"      Predicted: {pred_rm1:.1f} ± {best_result.components[0].rm_std:.1f} rad/m²")
        
        results.append({
            "scenario": name,
            "true_n": true_n,
            "predicted_n": predicted_n,
            "correct": predicted_n == true_n,
            "result": best_result
        })
    
    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    
    n_correct = sum(r['correct'] for r in results)
    accuracy = 100 * n_correct / len(results)
    
    print(f"\nDecision Layer Performance: {n_correct}/{len(results)} correct ({accuracy:.1f}%)")
    
    print(f"\nDetailed Results:")
    for r in results:
        status = "✓ PASS" if r['correct'] else "✗ FAIL"
        print(f"  {r['scenario']:<30} True: {r['true_n']}  Pred: {r['predicted_n']}  {status}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Decision layer successfully classifies between 1-comp and 2-comp")
    print("2. System handles missing channels (weight=0) through interpolation")
    print("3. Varying noise levels (different weights) are properly accounted for")
    print("4. RFI gaps and scattered missing data don't prevent accurate inference")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demo of the weighted two-layer neural network system"
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
    
    args = parser.parse_args()
    
    run_demo(config_path=args.config, model_dir=args.model_dir)


if __name__ == '__main__':
    main()
