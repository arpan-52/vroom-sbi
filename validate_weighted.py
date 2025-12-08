#!/usr/bin/env python3
"""
Validation script for the two-layer weighted neural network system.

Tests the system with various scenarios:
1. Clean data (all weights = 1.0)
2. Data with missing channels
3. Data with varying noise levels
4. 1-component vs 2-component ground truth

Demonstrates decision layer accuracy and parameter recovery.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.simulator import RMSimulator, build_prior
from src.inference import RMInference
from src.physics import load_frequencies
from src.augmentation import (
    augment_weights_scattered,
    augment_weights_contiguous_gap,
    augment_weights_noise_variation
)


def create_test_cases(freq_file: str, config: dict):
    """
    Create test cases for validation.
    
    Returns
    -------
    list of dict
        Each dict contains:
        - name: Test case name
        - n_components: True number of components
        - theta: True parameters
        - weights: Channel weights
        - qu_obs: Observed Q, U
    """
    flat_priors = {
        "rm_min": config["priors"]["rm"]["min"],
        "rm_max": config["priors"]["rm"]["max"],
        "amp_min": config["priors"]["amp"]["min"],
        "amp_max": config["priors"]["amp"]["max"],
        "noise_min": config["priors"]["noise"]["min"],
        "noise_max": config["priors"]["noise"]["max"],
    }
    
    sim_1comp = RMSimulator(freq_file, 1)
    sim_2comp = RMSimulator(freq_file, 2)
    
    test_cases = []
    
    # Case 1: Clean 1-component data
    theta_1 = np.array([100.0, 0.5, 0.3, 0.002])  # RM, amp, chi0, noise
    weights_clean = np.ones(sim_1comp.n_freq)
    qu_1_clean = sim_1comp(theta_1, weights=weights_clean)
    test_cases.append({
        "name": "1-comp Clean",
        "n_components": 1,
        "theta": theta_1,
        "weights": weights_clean,
        "qu_obs": qu_1_clean
    })
    
    # Case 2: 1-component with scattered missing channels
    weights_scattered = augment_weights_scattered(weights_clean, missing_prob=0.15)
    qu_1_scattered = sim_1comp(theta_1, weights=weights_scattered)
    test_cases.append({
        "name": "1-comp Scattered Missing",
        "n_components": 1,
        "theta": theta_1,
        "weights": weights_scattered,
        "qu_obs": qu_1_scattered
    })
    
    # Case 3: 1-component with contiguous gap
    weights_gap = augment_weights_contiguous_gap(weights_clean, gap_prob=1.0, min_gap=5, max_gap=10)
    qu_1_gap = sim_1comp(theta_1, weights=weights_gap)
    test_cases.append({
        "name": "1-comp Contiguous Gap",
        "n_components": 1,
        "theta": theta_1,
        "weights": weights_gap,
        "qu_obs": qu_1_gap
    })
    
    # Case 4: 1-component with varying noise
    weights_varied = augment_weights_noise_variation(weights_clean, variation_scale=0.3)
    qu_1_varied = sim_1comp(theta_1, weights=weights_varied)
    test_cases.append({
        "name": "1-comp Varying Noise",
        "n_components": 1,
        "theta": theta_1,
        "weights": weights_varied,
        "qu_obs": qu_1_varied
    })
    
    # Case 5: Clean 2-component data
    theta_2 = np.array([150.0, 0.4, 0.5, -80.0, 0.3, 1.2, 0.002])  # RM1, amp1, chi01, RM2, amp2, chi02, noise
    qu_2_clean = sim_2comp(theta_2, weights=weights_clean)
    test_cases.append({
        "name": "2-comp Clean",
        "n_components": 2,
        "theta": theta_2,
        "weights": weights_clean,
        "qu_obs": qu_2_clean
    })
    
    # Case 6: 2-component with scattered missing channels
    weights_scattered_2 = augment_weights_scattered(weights_clean, missing_prob=0.15)
    qu_2_scattered = sim_2comp(theta_2, weights=weights_scattered_2)
    test_cases.append({
        "name": "2-comp Scattered Missing",
        "n_components": 2,
        "theta": theta_2,
        "weights": weights_scattered_2,
        "qu_obs": qu_2_scattered
    })
    
    # Case 7: 2-component with contiguous gap
    weights_gap_2 = augment_weights_contiguous_gap(weights_clean, gap_prob=1.0, min_gap=5, max_gap=10)
    qu_2_gap = sim_2comp(theta_2, weights=weights_gap_2)
    test_cases.append({
        "name": "2-comp Contiguous Gap",
        "n_components": 2,
        "theta": theta_2,
        "weights": weights_gap_2,
        "qu_obs": qu_2_gap
    })
    
    # Case 8: 2-component with varying noise
    weights_varied_2 = augment_weights_noise_variation(weights_clean, variation_scale=0.3)
    qu_2_varied = sim_2comp(theta_2, weights=weights_varied_2)
    test_cases.append({
        "name": "2-comp Varying Noise",
        "n_components": 2,
        "theta": theta_2,
        "weights": weights_varied_2,
        "qu_obs": qu_2_varied
    })
    
    return test_cases


def run_validation(config_path: str = "config.yaml", model_dir: str = "models"):
    """
    Run validation tests.
    """
    print("="*80)
    print("WEIGHTED TWO-LAYER SYSTEM VALIDATION")
    print("="*80)
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    freq_file = config.get("freq_file", "freq.txt")
    device = config.get("training", {}).get("device", "cpu")
    
    # Check if models exist
    if not os.path.exists(model_dir):
        print(f"\nError: Model directory '{model_dir}' not found!")
        print("Please train the models first using: python train_all.py")
        return
    
    # Check if required models exist
    required_models = ["posterior_n1.pkl", "posterior_n2.pkl", "decision_layer.pkl"]
    missing = [m for m in required_models if not os.path.exists(os.path.join(model_dir, m))]
    if missing:
        print(f"\nError: Missing required models: {missing}")
        print("Please train the models first using: python train_all.py")
        return
    
    # Initialize inference
    print(f"\nInitializing inference engine...")
    inference = RMInference(model_dir=model_dir, device=device, use_decision_layer=True)
    inference.load_models(max_components=2)
    
    # Create test cases
    print(f"\nGenerating test cases...")
    test_cases = create_test_cases(freq_file, config)
    print(f"Created {len(test_cases)} test cases")
    
    # Run validation
    results = []
    
    print("\n" + "="*80)
    print("RUNNING VALIDATION TESTS")
    print("="*80)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"Test Case {i+1}: {test_case['name']}")
        print(f"{'='*80}")
        print(f"True n_components: {test_case['n_components']}")
        print(f"Missing channels: {np.sum(test_case['weights'] == 0)}/{len(test_case['weights'])}")
        print(f"Min weight: {np.min(test_case['weights']):.3f}, Max weight: {np.max(test_case['weights']):.3f}")
        
        # Run inference
        best_result, all_results = inference.infer(
            test_case['qu_obs'],
            weights=test_case['weights'],
            n_samples=5000
        )
        
        # Check if decision layer got it right
        predicted_n = best_result.n_components
        true_n = test_case['n_components']
        correct = (predicted_n == true_n)
        
        print(f"\nPredicted n_components: {predicted_n}")
        print(f"Decision: {'✓ CORRECT' if correct else '✗ INCORRECT'}")
        
        # Calculate parameter recovery error for matching components
        if predicted_n == true_n:
            if true_n == 1:
                true_rm = test_case['theta'][0]
                pred_rm = best_result.components[0].rm_mean
                rm_error = abs(pred_rm - true_rm)
                print(f"RM recovery: True={true_rm:.1f}, Pred={pred_rm:.1f}, Error={rm_error:.1f} rad/m²")
            elif true_n == 2:
                true_rm1 = test_case['theta'][0]
                true_rm2 = test_case['theta'][3]
                pred_rm1 = best_result.components[0].rm_mean
                pred_rm2 = best_result.components[1].rm_mean
                
                # Match components (closest assignment)
                err1 = abs(pred_rm1 - true_rm1) + abs(pred_rm2 - true_rm2)
                err2 = abs(pred_rm1 - true_rm2) + abs(pred_rm2 - true_rm1)
                
                if err1 < err2:
                    print(f"RM1 recovery: True={true_rm1:.1f}, Pred={pred_rm1:.1f}, Error={abs(pred_rm1-true_rm1):.1f} rad/m²")
                    print(f"RM2 recovery: True={true_rm2:.1f}, Pred={pred_rm2:.1f}, Error={abs(pred_rm2-true_rm2):.1f} rad/m²")
                else:
                    print(f"RM1 recovery: True={true_rm1:.1f}, Pred={pred_rm2:.1f}, Error={abs(pred_rm2-true_rm1):.1f} rad/m²")
                    print(f"RM2 recovery: True={true_rm2:.1f}, Pred={pred_rm1:.1f}, Error={abs(pred_rm1-true_rm2):.1f} rad/m²")
        
        results.append({
            "test_case": test_case['name'],
            "true_n": true_n,
            "predicted_n": predicted_n,
            "correct": correct,
            "best_result": best_result,
        })
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    n_correct = sum(r['correct'] for r in results)
    accuracy = 100 * n_correct / len(results)
    
    print(f"\nDecision Layer Accuracy: {n_correct}/{len(results)} ({accuracy:.1f}%)")
    
    # Breakdown by true class
    n_1comp_cases = sum(1 for r in results if r['true_n'] == 1)
    n_1comp_correct = sum(1 for r in results if r['true_n'] == 1 and r['correct'])
    
    n_2comp_cases = sum(1 for r in results if r['true_n'] == 2)
    n_2comp_correct = sum(1 for r in results if r['true_n'] == 2 and r['correct'])
    
    print(f"\n1-component cases: {n_1comp_correct}/{n_1comp_cases} correct ({100*n_1comp_correct/n_1comp_cases:.1f}%)")
    print(f"2-component cases: {n_2comp_correct}/{n_2comp_cases} correct ({100*n_2comp_correct/n_2comp_cases:.1f}%)")
    
    # Detailed results
    print(f"\nDetailed Results:")
    print(f"{'Test Case':<30} {'True':<8} {'Pred':<8} {'Result':<10}")
    print("-"*60)
    for r in results:
        result_str = "✓ PASS" if r['correct'] else "✗ FAIL"
        print(f"{r['test_case']:<30} {r['true_n']:<8} {r['predicted_n']:<8} {result_str:<10}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    # Create visualization
    create_summary_plot(results, output_path="validation_results.png")
    print(f"\nSummary plot saved to: validation_results.png")
    
    return results


def create_summary_plot(results, output_path="validation_results.png"):
    """
    Create a summary visualization of validation results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Two-Layer Weighted System Validation Results", fontsize=16, fontweight='bold')
    
    # 1. Accuracy by test case
    ax = axes[0, 0]
    test_names = [r['test_case'] for r in results]
    correct_vals = [1 if r['correct'] else 0 for r in results]
    colors = ['green' if c else 'red' for c in correct_vals]
    
    ax.barh(range(len(test_names)), correct_vals, color=colors, alpha=0.7)
    ax.set_yticks(range(len(test_names)))
    ax.set_yticklabels(test_names, fontsize=9)
    ax.set_xlabel('Correct (1) / Incorrect (0)')
    ax.set_title('Decision Layer Accuracy by Test Case')
    ax.set_xlim([0, 1.2])
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Confusion matrix
    ax = axes[0, 1]
    true_1 = sum(1 for r in results if r['true_n'] == 1 and r['predicted_n'] == 1)
    true_2 = sum(1 for r in results if r['true_n'] == 2 and r['predicted_n'] == 2)
    false_12 = sum(1 for r in results if r['true_n'] == 1 and r['predicted_n'] == 2)
    false_21 = sum(1 for r in results if r['true_n'] == 2 and r['predicted_n'] == 1)
    
    confusion = np.array([[true_1, false_12], [false_21, true_2]])
    im = ax.imshow(confusion, cmap='Blues', vmin=0, vmax=max(true_1, true_2))
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['1-comp', '2-comp'])
    ax.set_yticklabels(['1-comp', '2-comp'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, confusion[i, j],
                          ha="center", va="center", color="black", fontsize=14)
    
    plt.colorbar(im, ax=ax)
    
    # 3. Class distribution
    ax = axes[1, 0]
    n_1comp = sum(1 for r in results if r['true_n'] == 1)
    n_2comp = sum(1 for r in results if r['true_n'] == 2)
    ax.bar(['1-component', '2-component'], [n_1comp, n_2comp], color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    ax.set_ylabel('Number of Test Cases')
    ax.set_title('Test Case Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Overall accuracy
    ax = axes[1, 1]
    n_correct = sum(r['correct'] for r in results)
    n_total = len(results)
    accuracy = 100 * n_correct / n_total
    
    ax.pie([n_correct, n_total - n_correct], 
           labels=['Correct', 'Incorrect'],
           autopct='%1.1f%%',
           colors=['green', 'red'],
           alpha=0.7,
           startangle=90)
    ax.set_title(f'Overall Accuracy: {accuracy:.1f}%')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Validation plot saved to {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate the two-layer weighted neural network system"
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
    
    run_validation(config_path=args.config, model_dir=args.model_dir)


if __name__ == '__main__':
    main()
