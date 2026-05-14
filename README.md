# VROOM-SBI

[![CI](https://github.com/arpan-52/vroom-sbi/actions/workflows/cube_integration.yml/badge.svg)](https://github.com/arpan-52/vroom-sbi/actions/workflows/cube_integration.yml)
[![Lint](https://github.com/arpan-52/vroom-sbi/actions/workflows/lint.yml/badge.svg)](https://github.com/arpan-52/vroom-sbi/actions/workflows/lint.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![sbi](https://img.shields.io/badge/sbi-0.18%2B-blueviolet)](https://sbi-dev.github.io/sbi/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-arpan--52%2Fvroom--sbi-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/arpan-52/vroom-sbi)

Simulation-Based Inference for Rotation Measure Synthesis. Given observed Q(λ²) and U(λ²) spectra, VROOM-SBI returns a full posterior probability distribution over Faraday depth structure using pre-trained neural posterior estimators.

Pre-trained models: [huggingface.co/arpan-52/vroom-sbi](https://huggingface.co/arpan-52/vroom-sbi)

---

## Installation

```bash
git clone https://github.com/arpan-52/vroom-sbi
cd vroom-sbi
pip install -e .
```

For FITS cube inference, add the IO extras:

```bash
pip install -e ".[io]"
```

---

## Frequency setup and observing constraints

VROOM-SBI is trained on a specific λ² sampling defined by `freq.txt`. The pre-trained models ship with the following L-band setup:

| Property | Value |
|----------|-------|
| Frequency range | 1.0 -- 2.0 GHz |
| Number of channels | 128 |
| Channel width | ~7.87 MHz |
| λ² range | 0.023 -- 0.090 m² |
| λ² span (Δλ²) | 0.067 m² |

These translate to the following RM synthesis figures of merit:

| Figure of merit | Value | Notes |
|----------------|-------|-------|
| RMSF FWHM (RM resolution) | ~51 rad/m² | Set by Δλ² = 0.067 m² |
| Max detectable RM scale | ~140 rad/m² | Set by λ²_min = 0.023 m² |
| RM prior range | -500 to +500 rad/m² | ~10 resolution elements |
| RM dispersion prior (σ_φ) | 0 -- 200 rad/m² | For external/internal dispersion models |

**Why the prior range matters.** The RMSF FWHM of ~51 rad/m² sets the minimum separation at which two RM components can be independently resolved. Components closer than ~25 rad/m² will produce degenerate posteriors with large uncertainties. The prior covers 10 resolution elements on each side of zero, which spans the RM range expected for most extragalactic sources observed at L-band.

**Channel flagging.** The weight augmentation during training simulates scattered flagging (30% probability), contiguous RFI gaps (30%), and large RFI blocks (10%), using continuous inverse-variance weights rather than binary masks. This means the trained posteriors degrade gracefully under realistic flagging -- no retraining required for typical RFI conditions.

**Using different frequency coverage.** To train on a different array or subband, replace `freq.txt` with your own frequency list (one frequency in Hz per line, optional second column for per-channel weights), update the prior ranges in `config.yaml` if your RM range differs, and retrain. The network architecture does not change -- only the simulation setup does.

---

## Quickstart: inference with pre-trained models

Models download automatically from HuggingFace the first time you run inference.

```python
import numpy as np
from src.inference import InferenceEngine

engine = InferenceEngine(model_dir="models")
engine.load_models()  # auto-downloads from HuggingFace if models/ is absent

# Q and U as 1D arrays over the 128 frequency channels
qu_obs = np.concatenate([Q, U])  # shape (256,)
result, all_results = engine.infer(qu_obs)
```

Or from the command line:

```bash
# Explicit download
vroom-sbi download

# Inline inference (auto-downloads if models/ is missing)
vroom-sbi infer --q "0.42,0.38,..." --u "0.11,0.19,..."
```

---

## Worked example on simulated data

Here is a minimal self-contained example that simulates a two-component polarized source, runs it through VROOM-SBI, and interprets the result.

```python
import numpy as np
from src.simulator import RMSimulator
from src.inference import InferenceEngine

# Simulate a 2-component Faraday-thin source
# Component 1: RM = +180 rad/m², fractional polarization = 0.55
# Component 2: RM =  -60 rad/m², fractional polarization = 0.30
sim = RMSimulator("freq.txt", n_components=2, model_type="faraday_thin")
theta_true = np.array([[180.0, 0.55, 0.4,    # [RM, amp, chi0]  component 1
                         -60.0, 0.30, 1.8]])  # [RM, amp, chi0]  component 2

qu_noiseless = sim.simulate_noiseless(theta_true)  # shape (256,): [Q_0..Q_127, U_0..U_127]

# Add noise at SNR ~ 20
rng = np.random.default_rng(42)
qu_obs = qu_noiseless + rng.normal(0, 0.02, qu_noiseless.shape)

# Run inference
engine = InferenceEngine(model_dir="models")
engine.load_models()
result, all_results = engine.infer(qu_obs, n_samples=10000)
```

**Example output:**

```
Best model: faraday_thin, 2 component(s)
Classifier probability: 0.94

Component 1:
  RM     =  181.3 ±  4.2 rad/m²   (true: 180.0)
  amp    =   0.54 ± 0.03
  chi0   =   0.41 ± 0.07 rad

Component 2:
  RM     =  -61.8 ±  6.1 rad/m²   (true: -60.0)
  amp    =   0.31 ± 0.04
  chi0   =   1.77 ± 0.12 rad
```

The two components are separated by 240 rad/m², well above the ~51 rad/m² resolution limit, so they are resolved cleanly. The posterior widths (~4--6 rad/m² on RM) reflect the SNR of each component individually, not the overall signal level.

---

## Interpreting results

### What the posterior gives you

Each `ComponentResult` carries full posterior marginals, not just point estimates:

```python
comp = result.components[0]

# Summary statistics
comp.rm_mean          # posterior mean RM in rad/m²
comp.rm_std           # posterior std (1-sigma uncertainty)

# Full samples for corner plots, credible intervals, derived quantities
comp.samples          # shape (n_samples, n_params)
                      # columns: [RM, amp, chi0] for faraday_thin
                      #          [phi, sigma_phi, amp, chi0] for dispersion models
                      #          [phi_c, delta_phi, amp, chi0] for burn_slab

# Model-specific dispersion parameters (None for faraday_thin)
comp.sigma_phi_mean   # RM dispersion for external/internal dispersion
comp.sigma_phi_std
comp.delta_phi_mean   # Slab half-width for burn_slab
comp.delta_phi_std
```

### Model selection

The classifier selects both the physical model and component count simultaneously. The `log_evidence` field on `InferenceResult` is the log marginal likelihood from the trained flow -- higher is better, and differences greater than ~2--3 nats are meaningful.

```python
# See all models evaluated
for key, res in all_results.items():
    print(f"{key:35s}  log_evidence={res.log_evidence:7.2f}  "
          f"p={res.classifier_probability:.3f}")
```

Typical output for a clean 2-component source:

```
faraday_thin_n1                    log_evidence= -12.4   p=0.02
faraday_thin_n2                    log_evidence=  -3.1   p=0.94
faraday_thin_n3                    log_evidence=  -5.8   p=0.03
external_dispersion_n1             log_evidence= -11.2   p=0.01
```

### Posterior corner plot

```python
import corner

comp = result.components[0]
param_labels = ["RM (rad/m²)", "amp", "χ₀ (rad)"]
corner.corner(comp.samples, labels=param_labels,
              truths=[180.0, 0.55, 0.4],
              quantiles=[0.16, 0.5, 0.84])
```

### Credible intervals

```python
# 68% and 95% credible intervals on RM for component 1
rm_samples = result.components[0].samples[:, 0]
p16, p50, p84 = np.percentile(rm_samples, [16, 50, 84])
p2, p98 = np.percentile(rm_samples, [2.5, 97.5])
print(f"RM = {p50:.1f}  [{p16:.1f}, {p84:.1f}]  (68%)")
print(f"RM = {p50:.1f}  [{p2:.1f},  {p98:.1f}]  (95%)")
```

### When to use each physical model

**`faraday_thin`** -- use when the source is compact along the line of sight and the polarized emission arises from a single well-defined Faraday depth. The simplest model; try this first.

**`external_dispersion`** -- use when a turbulent foreground screen causes depolarization that increases with λ⁴. The σ_φ parameter quantifies the width of the foreground RM distribution. At L-band with σ_φ > ~30 rad/m² you will see significant depolarization toward the long-λ end of the band.

**`internal_dispersion`** (Sokoloff model) -- use when Faraday rotation and synchrotron emission are co-spatial, e.g., a magnetized jet or lobe. The depolarization is more complex (the (1 - exp(-S))/S function) and the recovered φ is the mean depth of the emitting region, not a foreground screen.

**`burn_slab`** -- use when the source has a roughly uniform RM gradient along the line of sight, producing sinc-function depolarization. Δφ is the half-width of the slab in Faraday depth.

---

## FITS cube inference

Run inference over all spatial pixels of a polarization cube:

```bash
vroom-sbi cube-infer-pol \
    --cube-q Q.fits \
    --cube-u U.fits \
    --cube-i I.fits \
    --output-dir results/
```

Providing `--cube-i` divides Q and U by I per channel before inference, converting to fractional polarization. Without it, the input is assumed to already be in fractional units.

Output directory contains FITS and NPZ maps:

```
results/
  rm_mean_comp1.fits      # posterior mean RM, component 1
  rm_std_comp1.fits       # posterior std
  rm_p16_comp1.fits       # 16th percentile
  rm_p84_comp1.fits       # 84th percentile
  amp_mean_comp1.fits
  chi0_mean_comp1.fits
  sigma_phi_mean_comp1.fits   # if dispersion model selected
  ...
```

---

## Training your own models

Copy and edit `config.yaml` -- at minimum set `freq_file` to your own channel list. Then:

```bash
vroom-sbi train --config config.yaml --device cuda
```

Training budget scales with the number of components. With `n_simulations: 30000` and `scaling_power: 2.5`, the N=5 model trains on ~30000 × 5^2.5 ≈ 1.7M simulations. On a single GPU this takes a few hours per model type. The prior sampler uses Sobol quasi-random sequences, which gives better parameter-space coverage than uniform random at the same simulation budget.

```bash
# Retrain the classifier only (after posteriors already exist)
vroom-sbi train --config config.yaml --classifier-only

# Push your trained models to HuggingFace
export HF_TOKEN=your_token
vroom-sbi push --model-dir models/ --repo-id your-org/your-repo
```

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- sbi >= 0.18
- numpy, astropy, matplotlib, corner, tqdm, pyyaml, scipy
- huggingface-hub (auto-download and push)
- spectral-cube (optional, FITS cube inference only)
