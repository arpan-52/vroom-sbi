# Implementation Summary: Extended VROOM-SBI

## Overview
Successfully extended VROOM-SBI to support:
- **1-5 components** (up from 1-2)
- **4 physical models**: Faraday thin, Burn slab, External dispersion, Internal dispersion (Sokoloff)
- All existing features maintained: weight augmentation, noise augmentation, flags

## Changes Made

### 1. Configuration (`config.yaml`)

**Changed:**
```yaml
model_selection:
  max_components: 5  # Changed from 2 to 5
```

**Added:**
```yaml
physics:
  model_type: "faraday_thin"  # Options: faraday_thin, burn_slab, external_dispersion, internal_dispersion

  faraday_thick:
    enable_depolarization: true

  burn_slab:
    depolarization_sigma: 0.1

  sokoloff:
    enable_spatial_variation: true
    turbulence_scale: 100.0
```

### 2. Simulator (`src/simulator.py`)

**Physical Models Implemented:**

#### Faraday Thin (Original)
```
P(λ²) = Σⱼ pⱼ exp[2i(χ₀,ⱼ + φⱼλ²)]
Parameters: [RM, amp, chi0] per component → 3N params
```

#### Burn Slab (NEW)
```
P(λ²) = p₀ [sin(Δφλ²)/(Δφλ²)] exp[2i(χ₀ + φ_c λ²)]
Parameters: [phi_c, delta_phi, amp, chi0] per component → 4N params
```

#### External Dispersion (NEW)
```
P(λ²) = p₀ exp(-2σ_φ² λ⁴) exp[2i(χ₀ + φλ²)]
Parameters: [phi, sigma_phi, amp, chi0] per component → 4N params
```

#### Internal Dispersion - Sokoloff (NEW)
```
P(λ²) = p₀ [(1-exp(-S))/S] exp(2iχ₀)
where S = 2σ_φ² λ⁴ - 2iφλ²
Parameters: [phi, sigma_phi, amp, chi0] per component → 4N params
```

**Key Changes:**
- `RMSimulator.__init__()`: Added `model_type` and `model_params` parameters
- Added 4 model-specific polarization computation methods
- `__call__()`: Routes to appropriate physical model
- `build_prior()`: Handles 3N params (thin) and 4N params (others)
- `sample_prior()`: Supports different parameter counts per model
- `sort_components_by_rm()`: Generalized to handle variable `params_per_comp`

### 3. Training (`src/train.py`)

**Key Changes:**
- `train_model()`: Added `model_type` and `model_params` parameters
- `train_all_models()`:
  - Removed restriction: `max_comp = min(training_cfg["max_components"], 2)` → `max_comp = training_cfg["max_components"]`
  - Extracts `model_type` from config and passes to all training functions
  - Updated output messages to reflect 1-5 components

**Scaling Factors** (already implemented):
```python
scaling_factors = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8}
```
- 1-comp: 20,000 simulations (base)
- 2-comp: 40,000 simulations (2x)
- 3-comp: 80,000 simulations (4x)
- 4-comp: 120,000 simulations (6x)
- 5-comp: 160,000 simulations (8x)

### 4. Classifier (`src/classifier.py`)

**No changes needed!** The classifier was already extensible:
- `SpectralClassifier(n_classes=...)` supports arbitrary number of classes
- `ClassifierTrainer` automatically adjusts to `n_classes`
- Training code passes `n_classes=max_components`

### 5. Files Modified

| File | Status | Changes |
|------|--------|---------|
| `config.yaml` | ✅ Modified | max_components: 5, added physics section |
| `src/simulator.py` | ✅ Modified | 4 physical models, variable params_per_comp |
| `src/train.py` | ✅ Modified | Removed 2-comp limit, model_type support |
| `src/classifier.py` | ✅ No change | Already supports n_classes=5 |
| `src/inference.py` | ✅ No change | Dynamically loads models |
| `src/augmentation.py` | ✅ No change | Works with all models |
| `src/physics.py` | ✅ No change | Generic functions |

### 6. Test Script Created

**`test_physical_models.py`**:
- Tests all 4 physical models
- Tests 1, 2, 3, 5 components for each model
- Verifies:
  - Simulator instantiation
  - Prior sampling
  - Simulation output shapes
  - No NaN/Inf values
  - RM sorting (multi-component)

## Training Workflow

### For Faraday Thin model (1-5 components):
```bash
# config.yaml: physics.model_type: "faraday_thin"
python train_all.py
```

This will train:
1. **Worker models**: posterior_n1.pkl through posterior_n5.pkl
2. **Classifier**: classifier.pkl (5-class CNN)

### For Burn Slab model (1-5 components):
```yaml
# config.yaml
physics:
  model_type: "burn_slab"
```
```bash
python train_all.py
```

### For External Dispersion (1-5 components):
```yaml
# config.yaml
physics:
  model_type: "external_dispersion"
```
```bash
python train_all.py
```

### For Internal Dispersion / Sokoloff (1-5 components):
```yaml
# config.yaml
physics:
  model_type: "internal_dispersion"
```
```bash
python train_all.py
```

## Parameter Layouts

### Faraday Thin
```
[RM_1, amp_1, chi0_1, RM_2, amp_2, chi0_2, ...] → 3N params
```

### Burn Slab / External / Internal
```
[phi_1, sigma/delta_1, amp_1, chi0_1, phi_2, sigma/delta_2, amp_2, chi0_2, ...] → 4N params
```

## Key Features Maintained

✅ **Weight Augmentation**:
- Scattered missing channels
- Contiguous gaps (RFI)
- Large RFI blocks
- Noise variation

✅ **Noise Augmentation**:
- Random variation: 0.5x to 2.0x base_level

✅ **RM Sorting**:
- Enforces RM_1 > RM_2 > RM_3 > ... (descending)
- Breaks label switching symmetry

✅ **Classifier**:
- 1D CNN with spectral convolutions
- Input: [Q, U, weights] as 3 channels
- Output: n_classes probabilities

## What's Next

1. **Run tests**: `python test_physical_models.py`
2. **Train initial model**: Start with `faraday_thin` and 1-5 components
3. **Validate**: Check classifier accuracy across all component counts
4. **Train other models**: Burn slab, external dispersion, internal dispersion
5. **Compare**: Analyze which models best fit real data

## Notes

- Each model type should be trained separately (not mixed in single training run)
- Linear combinations of different model types can be explored in future work (equation 6)
- Current implementation trains one model_type at a time for clarity
- Classifier learns to distinguish 1-5 components within a single model type

## Equations Reference

All implemented equations match the user-provided formulas:
1. ✅ Faraday thin: P = p₀ exp[2i(χ₀ + φλ²)]
2. ✅ N components: P = Σⱼ pⱼ exp[2i(χ₀,ⱼ + φⱼλ²)]
3. ✅ Burn slab: P = p₀ sinc(Δφλ²) exp[2i(χ₀ + φ_c λ²)]
4. ✅ External dispersion: P = p₀ exp(-2σ_φ² λ⁴) exp[2i(χ₀ + φλ²)]
5. ✅ Internal dispersion: P = p₀ [(1-exp(-S))/S] exp(2iχ₀), S = 2σ_φ² λ⁴ - 2iφλ²
6. ⚡ Linear superposition: P = Σ_k P_k(λ²) [future work]
7. ✅ Stokes: Q = Re[P], U = Im[P]
