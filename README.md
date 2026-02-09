# VROOM-SBI

Simulation-Based Inference for RM (Rotation Measure) Synthesis.

## Features

- **Multiple physical models**: Faraday thin, Burn slab, External dispersion, Internal dispersion
- **Multi-component support**: 1-5 RM components with RM sorting to break label switching
- **CNN classifier**: Automatic model/component selection
- **Proper depolarization**: Corrected λ⁴ formulas for dispersion models
- **HuggingFace integration**: Push/download models to HF Hub

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd vroom-sbi

# Install in development mode
pip install -e .
```

## Usage

### Train Models

```bash
vroom-sbi train --config config.yaml
vroom-sbi train --config config.yaml --device cuda
vroom-sbi train --config config.yaml --classifier-only  # Retrain classifier only
```

### Validate Trained Models

```bash
vroom-sbi validate --config config.yaml --models-dir models/
vroom-sbi validate --config config.yaml --n-tests 50 --n-samples 10000
vroom-sbi validate --config config.yaml --output-dir validation_results/
```

### Run Inference

```bash
vroom-sbi infer --q "0.5,0.3,-0.1,0.2" --u "0.1,0.2,0.3,0.4"
vroom-sbi infer --q "..." --u "..." --models-dir models/ --n-samples 20000
vroom-sbi infer --q "..." --u "..." --output results.png
```

### Push to HuggingFace

```bash
export HF_TOKEN=your_token_here
vroom-sbi push --models-dir models/ --repo-id username/vroom-sbi-models
```

## Python API

```python
from src import Configuration, train_all_models, InferenceEngine

# Train
config = Configuration.from_yaml('config.yaml')
train_all_models(config)

# Inference
engine = InferenceEngine(config, model_dir='models')
engine.load_models()

import numpy as np
qu_obs = np.concatenate([Q_data, U_data])
result, all_results = engine.infer(qu_obs)

print(f"Best model: {result.model_type}, N={result.n_components}")
for comp in result.components:
    print(f"  RM = {comp.rm_mean:.2f} ± {comp.rm_std:.2f}")
```

## Physical Models

| Model | Parameters per component | Description |
|-------|-------------------------|-------------|
| `faraday_thin` | RM, amp, χ₀ | Simple Faraday rotation |
| `burn_slab` | φ_c, Δφ, amp, χ₀ | Slab with sinc depolarization |
| `external_dispersion` | φ, σ_φ, amp, χ₀ | External Gaussian depolarization (λ⁴) |
| `internal_dispersion` | φ, σ_φ, amp, χ₀ | Internal/Sokoloff dispersion (λ⁴) |

## RM Sorting (Label Switching)

For multi-component models (N ≥ 2), components are sorted by Faraday depth:
- RM₁ > RM₂ > RM₃ > ... (descending)

This ensures unique ordering and breaks the label switching symmetry where
(RM₁, RM₂) and (RM₂, RM₁) represent the same physical configuration.

Sorting is applied:
1. During prior sampling (training)
2. After posterior sampling (inference)

## Configuration

See `config.yaml` for all options:

```yaml
freq_file: "freq.txt"

physics:
  model_types:
    - faraday_thin
    - burn_slab
    - external_dispersion
    - internal_dispersion

priors:
  rm:
    min: -800.0
    max: 800.0
  amp:
    min: 0.001
    max: 1.0

model_selection:
  max_components: 5
  use_classifier: true

training:
  n_simulations: 20000
  device: "cuda"
  save_dir: "models"
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- sbi >= 0.18
- numpy, astropy, matplotlib, corner, tqdm, pyyaml
- huggingface_hub (optional, for HF integration)
