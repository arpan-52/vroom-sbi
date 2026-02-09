"""
Plotting utilities for VROOM-SBI.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_inference_result(
    result,
    qu_obs: np.ndarray,
    freq_file: str,
    phi_range: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
):
    """
    Create comprehensive visualization of inference results.
    
    Parameters
    ----------
    result : InferenceResult
        Inference result
    qu_obs : np.ndarray
        Observed [Q, U] spectrum
    freq_file : str
        Path to frequency file
    phi_range : tuple, optional
        Faraday depth range
    output_path : str, optional
        Where to save figure
    figsize : tuple
        Figure size
    """
    from ..simulator.physics import load_frequencies, freq_to_lambda_sq, compute_rmsf
    
    try:
        import corner
    except ImportError:
        logger.warning("corner package not available, skipping corner plot")
        corner = None
    
    # Load frequencies
    frequencies, _ = load_frequencies(freq_file)
    lambda_sq = freq_to_lambda_sq(frequencies)
    n_channels = len(frequencies)
    
    # Split Q and U
    Q_obs = qu_obs[:n_channels]
    U_obs = qu_obs[n_channels:]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Q and U vs lambda^2
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(lambda_sq * 1000, Q_obs, 'ro-', label='Q', markersize=4, alpha=0.7)
    ax1.plot(lambda_sq * 1000, U_obs, 'bs-', label='U', markersize=4, alpha=0.7)
    ax1.set_xlabel('λ² (×10⁻³ m²)', fontsize=11)
    ax1.set_ylabel('Stokes Q, U', fontsize=11)
    ax1.set_title('Observed Polarization', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSF
    ax2 = fig.add_subplot(gs[0, 1])
    
    if phi_range is None:
        rm_values = [comp.rm_mean for comp in result.components]
        if rm_values:
            rm_range = max(abs(max(rm_values)), abs(min(rm_values)))
            phi_range = (-rm_range * 2, rm_range * 2)
        else:
            phi_range = (-1000, 1000)
    
    phi = np.linspace(phi_range[0], phi_range[1], 500)
    rmsf = compute_rmsf(lambda_sq, phi)
    
    ax2.plot(phi, np.abs(rmsf), 'k-', linewidth=1.5)
    ax2.set_xlabel('Faraday Depth (rad/m²)', fontsize=11)
    ax2.set_ylabel('|RMSF|', fontsize=11)
    ax2.set_title('Rotation Measure Spread Function', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark inferred RM components
    colors = plt.cm.tab10(np.linspace(0, 1, len(result.components)))
    for i, comp in enumerate(result.components):
        ax2.axvline(comp.rm_mean, color=colors[i], linestyle='--', linewidth=2,
                   label=f'RM{i+1}={comp.rm_mean:.1f}±{comp.rm_std:.1f}')
    ax2.legend(fontsize=9)
    
    # 3. RM posterior histograms
    ax3 = fig.add_subplot(gs[1, 0])
    
    for i, comp in enumerate(result.components):
        rm_samples = comp.samples[:, 0]
        ax3.hist(rm_samples, bins=50, alpha=0.6, color=colors[i],
                label=f'Component {i+1}', density=True)
    
    ax3.set_xlabel('Rotation Measure (rad/m²)', fontsize=11)
    ax3.set_ylabel('Probability Density', fontsize=11)
    ax3.set_title(f'RM Posteriors ({result.model_type}, N={result.n_components})',
                  fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary text or corner plot
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Summary text
    summary_text = f"Model: {result.model_type}\n"
    summary_text += f"Components: {result.n_components}\n"
    summary_text += f"Log Evidence: {result.log_evidence:.2f}\n\n"
    
    for i, comp in enumerate(result.components):
        summary_text += f"Component {i+1}:\n"
        summary_text += f"  RM = {comp.rm_mean:.2f} ± {comp.rm_std:.2f}\n"
        summary_text += f"  q = {comp.q_mean:.4f} ± {comp.q_std:.4f}\n"
        summary_text += f"  u = {comp.u_mean:.4f} ± {comp.u_std:.4f}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Summary', fontsize=12, fontweight='bold')
    
    # Main title
    fig.suptitle(
        f'VROOM-SBI Results: {result.n_components} Component(s), '
        f'Log Evidence = {result.log_evidence:.2f}',
        fontsize=13, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")
    else:
        plt.show()
    
    plt.close(fig)
    return fig


def plot_training_summary(
    results: dict,
    output_dir: Path,
    model_types: list,
    max_components: int,
):
    """
    Generate training summary plots.
    
    Parameters
    ----------
    results : dict
        Training results dictionary
    output_dir : Path
        Output directory
    model_types : list
        Model types trained
    max_components : int
        Maximum components
    """
    model_names = []
    val_losses = []
    n_samples = []
    
    for model_type in model_types:
        for n in range(1, max_components + 1):
            key = f"{model_type}_n{n}"
            if key in results:
                result = results[key]
                model_names.append(f"{model_type}\nN={n}")
                
                if hasattr(result, 'final_val_loss'):
                    val_losses.append(result.final_val_loss or np.nan)
                else:
                    val_losses.append(np.nan)
                
                if hasattr(result, 'n_simulations'):
                    n_samples.append(result.n_simulations)
                else:
                    n_samples.append(0)
    
    if not model_names:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    x_pos = np.arange(len(model_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))
    
    # Assign colors
    bar_colors = []
    for name in model_names:
        for i, mtype in enumerate(model_types):
            if mtype in name:
                bar_colors.append(colors[i])
                break
    
    # Sample counts
    ax1.bar(x_pos, [s/1000 for s in n_samples], color=bar_colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Training Samples (thousands)', fontsize=11)
    ax1.set_title('Training Sample Counts', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=mtype)
                      for i, mtype in enumerate(model_types)]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Validation losses
    if not all(np.isnan(val_losses)):
        ax2.bar(x_pos, val_losses, color=bar_colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Validation Loss', fontsize=11)
        ax2.set_title('Final Validation Performance', fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Validation metrics not available',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training summary plot")
