#!/usr/bin/env python3
"""
Validate the Quality Prediction Decision Layer.

Tests the decision layer's ability to:
1. Predict quality metrics (AIC, BIC, log_evidence) for both models
2. Correctly select the true model via ensemble voting
3. Handle various scenarios (1-comp, 2-comp, edge cases)
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from src.simulator import RMSimulator, sample_prior
from src.physics import load_frequencies
from src.decision import QualityPredictionTrainer
from src.augmentation import augment_weights_combined


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_flat_priors(config: Dict) -> Dict[str, float]:
    """Extract flat priors from config."""
    pri = config.get("priors", {})
    rm = pri.get("rm", {})
    amp = pri.get("amp", {})

    return {
        "rm_min": float(rm.get("min", -500.0)),
        "rm_max": float(rm.get("max", 500.0)),
        "amp_min": max(float(amp.get("min", 0.0)), 1e-6),
        "amp_max": float(amp.get("max", 1.0)),
    }


def get_base_noise_level(config: Dict) -> float:
    """Extract base noise level from config."""
    noise_cfg = config.get("noise", {})
    return float(noise_cfg.get("base_level", 0.01))


def load_models(model_dir: str = "models"):
    """Load trained posteriors and decision layer."""
    model_dir = Path(model_dir)

    # Load worker models
    with open(model_dir / "posterior_n1.pkl", "rb") as f:
        data_1 = pickle.load(f)
        posterior_1 = data_1["posterior"]

    with open(model_dir / "posterior_n2.pkl", "rb") as f:
        data_2 = pickle.load(f)
        posterior_2 = data_2["posterior"]

    # Load decision layer
    decision_path = model_dir / "decision_layer.pkl"
    decision_trainer = QualityPredictionTrainer.load(str(decision_path))

    return posterior_1, posterior_2, decision_trainer, data_1["n_freq"]


def generate_test_cases(
    n_tests_per_model: int,
    freq_file: str,
    flat_priors: Dict[str, float],
    base_noise_level: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate balanced test cases from both 1-comp and 2-comp models.

    Returns
    -------
    X_test : np.ndarray
        Test inputs [Q, U, weights] of shape (2*n_tests, 3*n_freq)
    true_labels : np.ndarray
        True model labels (1 or 2) of shape (2*n_tests,)
    true_params : list
        True parameters for each test case
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    frequencies, weights = load_frequencies(freq_file)
    n_freq = len(frequencies)

    X_test = []
    true_labels = []
    true_params = []

    # Generate from both models
    for n_comp in [1, 2]:
        print(f"\nGenerating {n_tests_per_model} test cases from {n_comp}-component model...")

        sim = RMSimulator(freq_file, n_comp, base_noise_level=base_noise_level)
        theta_samples = sample_prior(n_tests_per_model, n_comp, flat_priors)

        for i in tqdm(range(n_tests_per_model), desc=f"{n_comp}-comp"):
            # Simulate with augmented weights
            aug_weights = augment_weights_combined(sim.weights)
            qu_obs = sim(theta_samples[i:i+1], weights=aug_weights).flatten()

            # Input: [Q, U, weights]
            x_input = np.concatenate([qu_obs, aug_weights])

            X_test.append(x_input)
            true_labels.append(n_comp)
            true_params.append(theta_samples[i])

    X_test = np.array(X_test)
    true_labels = np.array(true_labels)

    return X_test, true_labels, true_params


def evaluate_decision_layer(
    X_test: np.ndarray,
    true_labels: np.ndarray,
    decision_trainer: QualityPredictionTrainer,
    device: str = "cpu"
):
    """
    Evaluate decision layer performance.

    Returns
    -------
    predictions : dict
        Dictionary with predicted quality metrics
    accuracies : dict
        Accuracies for each metric and ensemble
    """
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    print("\nRunning decision layer predictions...")
    with torch.no_grad():
        predictions = decision_trainer.predict(X_test_t)

    # Extract predictions
    log_ev_pred = predictions['log_evidence'].cpu().numpy()  # (n_tests, 2)
    aic_pred = predictions['aic'].cpu().numpy()
    bic_pred = predictions['bic'].cpu().numpy()
    ensemble_pred = predictions['ensemble_decision'].cpu().numpy()  # (n_tests,)

    # Individual metric decisions (higher log_ev is better, lower AIC/BIC is better)
    log_ev_decision = np.argmax(log_ev_pred, axis=1) + 1  # Convert to 1 or 2
    aic_decision = np.argmin(aic_pred, axis=1) + 1
    bic_decision = np.argmin(bic_pred, axis=1) + 1

    # Calculate accuracies
    accuracies = {
        'log_evidence': np.mean(log_ev_decision == true_labels) * 100,
        'aic': np.mean(aic_decision == true_labels) * 100,
        'bic': np.mean(bic_decision == true_labels) * 100,
        'ensemble': np.mean(ensemble_pred == true_labels) * 100,
    }

    # Per-class accuracy
    mask_1comp = true_labels == 1
    mask_2comp = true_labels == 2

    accuracies['1comp_ensemble'] = np.mean(ensemble_pred[mask_1comp] == 1) * 100
    accuracies['2comp_ensemble'] = np.mean(ensemble_pred[mask_2comp] == 2) * 100

    predictions_dict = {
        'log_evidence': log_ev_pred,
        'aic': aic_pred,
        'bic': bic_pred,
        'log_ev_decision': log_ev_decision,
        'aic_decision': aic_decision,
        'bic_decision': bic_decision,
        'ensemble_decision': ensemble_pred,
    }

    return predictions_dict, accuracies


def plot_confusion_matrices(
    true_labels: np.ndarray,
    predictions: dict,
    output_dir: Path
):
    """Plot confusion matrices for each decision metric."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    metrics = [
        ('log_ev_decision', 'Log Evidence'),
        ('aic_decision', 'AIC'),
        ('bic_decision', 'BIC'),
        ('ensemble_decision', 'Ensemble')
    ]

    for ax, (key, name) in zip(axes.flat, metrics):
        pred = predictions[key]

        # Compute confusion matrix
        cm = np.zeros((2, 2), dtype=int)
        for true_val, pred_val in zip(true_labels, pred):
            cm[true_val - 1, pred_val - 1] += 1

        # Plot
        im = ax.imshow(cm, cmap='Blues', aspect='auto')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, str(cm[i, j]),
                             ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white",
                             fontsize=20, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['1-comp', '2-comp'])
        ax.set_yticklabels(['1-comp', '2-comp'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)

        # Calculate accuracy
        acc = np.trace(cm) / cm.sum() * 100
        ax.set_title(f'{name}\nAccuracy: {acc:.1f}%', fontsize=14, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'confusion_matrices.png'}")


def plot_metric_distributions(
    true_labels: np.ndarray,
    predictions: dict,
    output_dir: Path
):
    """Plot distributions of predicted quality metrics."""
    log_ev = predictions['log_evidence']
    aic = predictions['aic']
    bic = predictions['bic']

    mask_1comp = true_labels == 1
    mask_2comp = true_labels == 2

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Log Evidence
    ax = axes[0, 0]
    ax.hist(log_ev[mask_1comp, 0], bins=30, alpha=0.7, label='True: 1-comp', color='blue')
    ax.hist(log_ev[mask_2comp, 0], bins=30, alpha=0.7, label='True: 2-comp', color='red')
    ax.set_xlabel('Predicted Log Evidence (1-comp model)')
    ax.set_ylabel('Count')
    ax.set_title('Log Evidence Predictions for 1-comp Model')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(log_ev[mask_1comp, 1], bins=30, alpha=0.7, label='True: 1-comp', color='blue')
    ax.hist(log_ev[mask_2comp, 1], bins=30, alpha=0.7, label='True: 2-comp', color='red')
    ax.set_xlabel('Predicted Log Evidence (2-comp model)')
    ax.set_ylabel('Count')
    ax.set_title('Log Evidence Predictions for 2-comp Model')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AIC
    ax = axes[1, 0]
    ax.hist(aic[mask_1comp, 0], bins=30, alpha=0.7, label='True: 1-comp', color='blue')
    ax.hist(aic[mask_2comp, 0], bins=30, alpha=0.7, label='True: 2-comp', color='red')
    ax.set_xlabel('Predicted AIC (1-comp model)')
    ax.set_ylabel('Count')
    ax.set_title('AIC Predictions for 1-comp Model (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(aic[mask_1comp, 1], bins=30, alpha=0.7, label='True: 1-comp', color='blue')
    ax.hist(aic[mask_2comp, 1], bins=30, alpha=0.7, label='True: 2-comp', color='red')
    ax.set_xlabel('Predicted AIC (2-comp model)')
    ax.set_ylabel('Count')
    ax.set_title('AIC Predictions for 2-comp Model (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # BIC
    ax = axes[2, 0]
    ax.hist(bic[mask_1comp, 0], bins=30, alpha=0.7, label='True: 1-comp', color='blue')
    ax.hist(bic[mask_2comp, 0], bins=30, alpha=0.7, label='True: 2-comp', color='red')
    ax.set_xlabel('Predicted BIC (1-comp model)')
    ax.set_ylabel('Count')
    ax.set_title('BIC Predictions for 1-comp Model (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.hist(bic[mask_1comp, 1], bins=30, alpha=0.7, label='True: 1-comp', color='blue')
    ax.hist(bic[mask_2comp, 1], bins=30, alpha=0.7, label='True: 2-comp', color='red')
    ax.set_xlabel('Predicted BIC (2-comp model)')
    ax.set_ylabel('Count')
    ax.set_title('BIC Predictions for 2-comp Model (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metric_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'metric_distributions.png'}")


def print_summary(accuracies: dict, n_tests_per_model: int):
    """Print validation summary."""
    print("\n" + "="*70)
    print("DECISION LAYER VALIDATION SUMMARY")
    print("="*70)
    print(f"Total test cases: {2 * n_tests_per_model}")
    print(f"  - 1-component: {n_tests_per_model}")
    print(f"  - 2-component: {n_tests_per_model}")
    print("\n" + "-"*70)
    print("Model Selection Accuracy:")
    print("-"*70)
    print(f"  Log Evidence:  {accuracies['log_evidence']:6.2f}%")
    print(f"  AIC:           {accuracies['aic']:6.2f}%")
    print(f"  BIC:           {accuracies['bic']:6.2f}%")
    print(f"  Ensemble:      {accuracies['ensemble']:6.2f}%  ⭐")
    print("\n" + "-"*70)
    print("Per-Class Accuracy (Ensemble):")
    print("-"*70)
    print(f"  1-component:   {accuracies['1comp_ensemble']:6.2f}%")
    print(f"  2-component:   {accuracies['2comp_ensemble']:6.2f}%")
    print("="*70)

    # Interpretation
    print("\nInterpretation:")
    if accuracies['ensemble'] >= 90:
        print("  ✅ EXCELLENT - Decision layer is highly accurate!")
    elif accuracies['ensemble'] >= 75:
        print("  ✓  GOOD - Decision layer performs well")
    elif accuracies['ensemble'] >= 60:
        print("  ⚠  FAIR - Decision layer needs improvement")
    else:
        print("  ✗  POOR - Decision layer needs retraining")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Quality Prediction Decision Layer")
    parser.add_argument("--n-tests", type=int, default=100,
                       help="Number of test cases per model (total = 2 * n_tests)")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory containing trained models")
    parser.add_argument("--output-dir", type=str, default="validation_decision",
                       help="Output directory for plots")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    print("="*70)
    print("VALIDATING QUALITY PREDICTION DECISION LAYER")
    print("="*70)

    # Load configuration
    config = load_config()
    flat_priors = get_flat_priors(config)
    base_noise_level = get_base_noise_level(config)
    freq_file = config.get("freq_file", "freq.txt")

    # Load models
    print("\nLoading models...")
    posterior_1, posterior_2, decision_trainer, n_freq = load_models(args.model_dir)
    print(f"✓ Loaded 1-comp and 2-comp posteriors")
    print(f"✓ Loaded decision layer")
    print(f"✓ n_freq: {n_freq}")

    # Generate test cases
    X_test, true_labels, true_params = generate_test_cases(
        args.n_tests, freq_file, flat_priors, base_noise_level, args.seed
    )

    print(f"\nGenerated {len(X_test)} test cases:")
    print(f"  - Shape: {X_test.shape}")
    print(f"  - 1-comp: {np.sum(true_labels == 1)}")
    print(f"  - 2-comp: {np.sum(true_labels == 2)}")

    # Evaluate decision layer
    predictions, accuracies = evaluate_decision_layer(
        X_test, true_labels, decision_trainer, device
    )

    # Generate plots
    print("\nGenerating plots...")
    plot_confusion_matrices(true_labels, predictions, output_dir)
    plot_metric_distributions(true_labels, predictions, output_dir)

    # Print summary
    print_summary(accuracies, args.n_tests)

    print(f"\n✅ Validation complete! Results saved to: {output_dir}")
