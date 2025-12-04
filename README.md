# VROOM-SBI: Simulation-Based Inference for RM Synthesis

VROOM-SBI (Variable Rotation Measure via Simulation-Based Inference) is a Python package for inferring Rotation Measure (RM) components from polarized radio observations using neural posterior estimation.

## Features

- **Simulation-Based Inference**: Uses neural density estimation (SNPE) to learn posterior distributions
- **Multi-Component Support**: Automatically infers the number of RM components (1-5)
- **Model Selection**: Uses log evidence for model comparison
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

Train neural posterior estimators for all component models (N=1 to 5):

```bash
python train_all.py
```

This will create trained models in the `models/` directory:
- `model_n1.pkl` - Single component
- `model_n2.pkl` - Two components
- `model_n3.pkl` - Three components
- `model_n4.pkl` - Four components
- `model_n5.pkl` - Five components

### 2. Run Inference

Provide observed Stokes Q and U values to infer RM components:

```bash
python infer.py --q 0.5,0.3,-0.1,0.2,0.4,0.1,-0.2,0.3,0.1,0.2,0.3,0.1 \
                --u 0.1,0.2,0.3,0.1,-0.2,0.3,0.4,0.2,0.1,0.3,0.2,0.1 \
                --output results.png
```

Output:
- Best model and number of components
- Posterior samples for each parameter
- Visualization with data, RMSF, posterior, and corner plot

## How It Works

### Forward Model

VROOM-SBI models polarized emission as a sum of Faraday-thin components:

```
P(λ²) = Σᵢ pᵢ exp(2i·RMᵢ·λ²)
```

where:
- `pᵢ = qᵢ + i·uᵢ` is the complex polarization amplitude
- `RMᵢ` is the Rotation Measure in rad/m²
- `λ²` is the squared wavelength

This is converted to Stokes Q and U:
```
Q(λ²) = Re[P(λ²)] + noise
U(λ²) = Im[P(λ²)] + noise
```

### Neural Posterior Estimation

1. **Training Phase**: 
   - Generate synthetic observations from the forward model
   - Train a neural density estimator to approximate p(θ|x)
   - Separate models for N=1,2,3,4,5 components

2. **Inference Phase**:
   - Load trained models
   - Compute posterior samples for each model
   - Select best model using log evidence
   - Return posterior distributions for parameters

### Model Selection

The best number of components is selected using the log marginal likelihood (evidence):

```
log p(x|N) ≈ log Σᵢ p(x|θᵢ,N)
```

This naturally penalizes overly complex models (Occam's razor).

## Configuration

Edit `config.yaml` to customize:
- Frequency file path
- Faraday depth sampling range
- Prior ranges for RM, amplitude, and noise
- Training parameters (simulations, batch size, device)

## File Structure

```
vroom-sbi/
├── requirements.txt       # Python dependencies
├── config.yaml           # Configuration file
├── freq.txt              # Frequency channels
├── README.md             # This file
├── train_all.py          # Training script
├── infer.py              # Inference script
├── test_recovery.py      # Recovery tests
└── src/
    ├── __init__.py       # Package initialization
    ├── physics.py        # RM synthesis physics
    ├── simulator.py      # Forward model simulator
    ├── train.py          # Training functions
    ├── inference.py      # Inference and model selection
    └── plots.py          # Visualization functions
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
