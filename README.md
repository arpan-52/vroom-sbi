# VROOM-SBI: Simulation-Based Inference for RM Synthesis

VROOM-SBI (Variable Rotation Measure via Simulation-Based Inference) is a Python package for inferring Rotation Measure (RM) components from polarized radio observations using neural posterior estimation.

## Features

- **Two-Layer Neural Network Architecture**: Decision layer classifies 1-comp vs 2-comp, worker models recover parameters
- **Weighted Frequency Channels**: Handles varying noise levels and missing data
- **Missing Channel Interpolation**: Network learns to interpolate through flagged/RFI channels
- **Simulation-Based Inference**: Uses neural density estimation (SNPE) to learn posterior distributions
- **Intelligent Model Selection**: Decision layer classifier for automatic component selection
- **GPU Acceleration**: Full CUDA support for training and inference
- **Faraday Synthesis**: Built on standard RM synthesis techniques

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arpan-52/vroom-sbi.git
cd vroom-sbi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train Models

Train the two-layer neural network system (decision layer + worker models):

```bash
python train_all.py
```

This will create trained models in the `models/` directory:
- `posterior_n1.pkl` - Single component worker model
- `posterior_n2.pkl` - Two component worker model
- `decision_layer.pkl` - Decision layer classifier (1-comp vs 2-comp)

The training pipeline:
1. Trains worker models with weighted channel augmentation
2. Trains decision layer to classify between 1 and 2 components
3. Both use realistic missing channel patterns for robustness

### 2. Run Inference

Provide observed Stokes Q and U values to infer RM components:

```bash
python infer.py --q 0.5,0.3,-0.1,0.2,0.4,0.1,-0.2,0.3,0.1,0.2,0.3,0.1 \
                --u 0.1,0.2,0.3,0.1,-0.2,0.3,0.4,0.2,0.1,0.3,0.2,0.1 \
                --output results.png
```

Output:
- Best model and number of components (selected by decision layer)
- Posterior samples for each parameter
- Visualization with data, RMSF, posterior, and corner plot

### 3. Validate System

Test the two-layer system with various scenarios:

```bash
python validate_weighted.py
```

This runs 8 test cases covering:
- Clean data (all weights = 1.0)
- Scattered missing channels
- Contiguous gaps (RFI simulation)
- Varying noise levels
- Both 1-component and 2-component ground truth

Output:
- Decision layer accuracy metrics
- Parameter recovery quality
- Confusion matrix
- Summary visualizations

## How It Works

### Two-Layer Architecture

```
                    ┌─────────────────────────┐
   RM Spectra  ───► │   LAYER 1: DECISION     │ ───► "Use 1-component" or "Use 2-component"
                    │   (Model Selector)       │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   LAYER 2: WORKERS      │
                    │  ┌─────────┬──────────┐ │
                    │  │ 1-Comp  │  2-Comp  │ │ ───► Best Q, U model parameters
                    │  │ Model   │  Model   │ │
                    │  └─────────┴──────────┘ │
                    └─────────────────────────┘
```

**Layer 1 (Decision Layer):** Neural classifier that takes RM spectra with weights and decides between 1 or 2 components.

**Layer 2 (Worker Layer):** Separate trained models that recover RM parameters for the selected number of components.

### Weighted Channels

The system handles varying data quality through channel weights:

```
Weight = 1/σ (normalized so max = 1.0)

Weight = 1.0  →  Best quality, lowest noise
Weight = 0.5  →  Moderate noise (σ = 2 × σ_min)
Weight = 0.0  →  Missing/flagged channel (network interpolates)
```

The `freq.txt` file includes weights:
```
# freq (Hz)      weight
1.0e9            1.0      # Good channel
1.1e9            0.8      # Noisier channel
1.2e9            0.0      # Missing/flagged channel
```

### Forward Model

VROOM-SBI models polarized emission as a sum of Faraday-thin components:

```
P(λ²) = Σᵢ pᵢ exp(2i·RMᵢ·λ²)
```

where:
- `pᵢ = qᵢ + i·uᵢ` is the complex polarization amplitude
- `RMᵢ` is the Rotation Measure in rad/m²
- `λ²` is the squared wavelength

This is converted to Stokes Q and U with weighted noise:
```
Q(λ²) = Re[P(λ²)] + noise/weight
U(λ²) = Im[P(λ²)] + noise/weight
```

For missing channels (weight=0), Q and U are set to zero and the network learns to interpolate.

### Neural Posterior Estimation

1. **Training Phase**: 
   - Generate synthetic observations with augmented weights (missing channels, noise variation)
   - Train worker models (1-comp and 2-comp) to approximate p(θ|x, weights)
   - Train decision layer classifier on labeled data

2. **Inference Phase**:
   - Decision layer classifies the number of components
   - Selected worker model computes posterior samples
   - Return posterior distributions for parameters

### Weight Augmentation

During training, random augmentation strategies teach the network to handle real-world data:

1. **Scattered missing**: Random channels set to weight=0
2. **Contiguous gaps**: 2-8 channel blocks (RFI simulation)
3. **Large RFI blocks**: 10-30 channel gaps
4. **Noise variation**: Varying weights for different noise levels

This makes the network robust to missing data and varying quality.

## Configuration

Edit `config.yaml` to customize:
- Frequency file path
- Faraday depth sampling range
- Prior ranges for RM, amplitude, and noise
- Training parameters (simulations, batch size, device)

## File Structure

```
vroom-sbi/
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── freq.txt                 # Frequency channels with weights
├── README.md                # This file
├── train_all.py             # Training script
├── infer.py                 # Inference script
├── validate_weighted.py     # Validation script for two-layer system
├── test_recovery.py         # Recovery tests
└── src/
    ├── __init__.py          # Package initialization
    ├── physics.py           # RM synthesis physics
    ├── simulator.py         # Forward model with weighted channels
    ├── train.py             # Training functions (decision + workers)
    ├── inference.py         # Two-layer inference pipeline
    ├── decision.py          # Decision layer classifier
    ├── augmentation.py      # Weight augmentation strategies
    └── plots.py             # Visualization functions
```

## Testing

Run recovery tests to validate the implementation:

```bash
pytest test_recovery.py -v
```

## Requirements

- Python 3.7+
- PyTorch (GPU recommended)
- SBI library
- NumPy, SciPy, AstroPy
- Matplotlib, Corner (for visualization)

## Citation

If you use VROOM-SBI in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## References

- Burn, B. J. 1966, MNRAS, 133, 67 (RM Synthesis)
- Brentjens, M. A., & de Bruyn, A. G. 2005, A&A, 441, 1217 (RM Synthesis)
- Greenberg, D. et al. 2019, ICML (SBI/SNPE)
- Papamakarios, G. et al. 2019, NeurIPS (Neural Posterior Estimation)
