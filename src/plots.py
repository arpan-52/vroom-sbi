"""
Visualization functions for RM synthesis results
"""

import numpy as np
import matplotlib.pyplot as plt
import corner

from .physics import load_frequencies, freq_to_lambda_sq, compute_rmsf


def plot_result(result, qu_obs, freq_file, phi_range=None, figsize=(12, 10), 
                output_path=None):
    """
    Create a comprehensive visualization of inference results.
    
    Creates a 2x2 plot with:
    1. Observed Q and U vs lambda^2
    2. Rotation Measure Spread Function (RMSF)
    3. Posterior distributions (histograms)
    4. Corner plot for parameters
    
    Parameters
    ----------
    result : InferenceResult
        Inference result from RMInference
    qu_obs : np.ndarray
        Observed Q and U values
    freq_file : str
        Path to frequency file
    phi_range : tuple, optional
        (min, max) range for Faraday depth in rad/m^2
    figsize : tuple
        Figure size
    output_path : str, optional
        Path to save the figure
    """
    # Load frequencies
    frequencies = load_frequencies(freq_file)
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
    ax1.plot(lambda_sq, Q_obs, 'ro-', label='Q', markersize=6)
    ax1.plot(lambda_sq, U_obs, 'bs-', label='U', markersize=6)
    ax1.set_xlabel('λ² (m²)', fontsize=12)
    ax1.set_ylabel('Stokes Q, U', fontsize=12)
    ax1.set_title('Observed Polarization', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSF
    ax2 = fig.add_subplot(gs[0, 1])
    
    if phi_range is None:
        # Use range based on components
        rm_values = [comp.rm_mean for comp in result.components]
        if rm_values:
            rm_range = max(abs(max(rm_values)), abs(min(rm_values)))
            phi_range = (-rm_range * 2, rm_range * 2)
        else:
            phi_range = (-1000, 1000)
    
    phi = np.linspace(phi_range[0], phi_range[1], 500)
    rmsf = compute_rmsf(lambda_sq, phi)
    
    ax2.plot(phi, np.abs(rmsf), 'k-', linewidth=2)
    ax2.set_xlabel('Faraday Depth (rad/m²)', fontsize=12)
    ax2.set_ylabel('|RMSF|', fontsize=12)
    ax2.set_title('Rotation Measure Spread Function', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark inferred RM components
    for i, comp in enumerate(result.components):
        ax2.axvline(comp.rm_mean, color=f'C{i}', linestyle='--', 
                   label=f'RM{i+1}={comp.rm_mean:.1f}±{comp.rm_std:.1f}')
    
    if result.components:
        ax2.legend(fontsize=9)
    
    # 3. Posterior distributions
    ax3 = fig.add_subplot(gs[1, 0])
    
    n_components = result.n_components
    colors = plt.cm.tab10(np.linspace(0, 1, n_components))
    
    for i, comp in enumerate(result.components):
        ax3.hist(comp.samples[:, 0], bins=50, alpha=0.5, 
                color=colors[i], label=f'Component {i+1}',
                density=True)
    
    ax3.set_xlabel('Rotation Measure (rad/m²)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title(f'Posterior: {n_components} Component(s)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Corner plot
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Select subset of samples for corner plot (for speed)
    n_corner_samples = min(5000, len(result.all_samples))
    corner_samples = result.all_samples[:n_corner_samples]
    
    # Create labels
    labels = []
    for i in range(n_components):
        labels.append(f'RM{i+1}')
    for i in range(n_components):
        labels.append(f'q{i+1}')
    for i in range(n_components):
        labels.append(f'u{i+1}')
    labels.append('σ')
    
    # For corner plot, only show RM and noise parameters to keep it readable
    if n_components <= 3:
        # Show all parameters
        corner_indices = list(range(len(labels)))
        corner_labels = labels
    else:
        # Only show RM values and noise
        corner_indices = list(range(n_components)) + [len(labels) - 1]
        corner_labels = labels[:n_components] + ['σ']
    
    corner_data = corner_samples[:, corner_indices]
    
    # Create corner plot in the subplot
    fig_corner = corner.corner(
        corner_data,
        labels=corner_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.2f',
        fig=fig
    )
    
    # Move corner plot axes to our subplot position
    # This is a bit hacky but works
    corner_axes = fig_corner.get_axes()
    bbox = ax4.get_position()
    ax4.remove()
    
    # Adjust corner plot axes positions
    n_corner = len(corner_labels)
    size = bbox.width / n_corner
    
    for i, ax in enumerate(corner_axes):
        row = i // n_corner
        col = i % n_corner
        if row >= col:  # Only lower triangle
            ax.set_position([
                bbox.x0 + col * size,
                bbox.y0 + (n_corner - row - 1) * size,
                size,
                size
            ])
        else:
            ax.set_visible(False)
    
    # Add main title
    fig.suptitle(
        f'VROOM-SBI Results: {n_components} Component(s), '
        f'Log Evidence = {result.log_evidence:.2f}',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    
    return fig
