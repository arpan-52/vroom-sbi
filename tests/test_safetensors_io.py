"""
Tests for src/utils/safetensors_io.py — pickle-free .pt <-> .safetensors export.

Covers:
- Round-trip: tensors and nested state-dict groups survive bit-for-bit, and
  JSON-native metadata (scalars, lists, nested dicts) is preserved.
- The exported file carries no pickle: it opens with safetensors' own reader.
- Non-tensor "oddball" values (e.g. torch's TorchVersion) don't break export.
- Guard rails: non-dict checkpoints and tensor-free dicts raise.
"""

from collections import OrderedDict

import pytest
import torch

from src.utils.safetensors_io import (
    is_safetensors,
    load_safetensors_as_dict,
    pt_to_safetensors,
)


def _make_checkpoint():
    """A checkpoint shaped like a saved posterior (state-dict format)."""
    return {
        "density_estimator_state": OrderedDict(
            {"net.0.weight": torch.randn(4, 3), "net.0.bias": torch.randn(4)}
        ),
        "embedding_net_state": OrderedDict({"conv.weight": torch.randn(2, 1, 5)}),
        "lambda_sq": torch.linspace(0.01, 0.1, 8),
        "architecture": {
            "input_dim": 24,
            "embedding_dim": 4,
            "sbi_model": "nsf",
            "hidden_features": 16,
            "num_transforms": 3,
            "num_bins": 8,
        },
        "model_type": "faraday_thin",
        "n_components": 3,
        "n_params": 9,
        "n_freq": 8,
        "param_names": ["RM_1", "amp_1", "chi0_1"],
        "training_history": {"val_loss": [5.0, 1.0, 0.4]},
    }


def test_roundtrip_tensors_and_metadata(tmp_path):
    ckpt = _make_checkpoint()
    pt_path = tmp_path / "posterior_faraday_thin_n3.pt"
    torch.save(ckpt, pt_path)

    st_path = pt_to_safetensors(pt_path, tmp_path / "posterior.safetensors")
    assert st_path.exists()

    out = load_safetensors_as_dict(st_path)

    # Same top-level keys (groups regrouped, scalars restored).
    assert set(out) == set(ckpt)

    # State-dict groups survive bit-for-bit.
    for group in ("density_estimator_state", "embedding_net_state"):
        assert set(out[group]) == set(ckpt[group])
        for k, v in ckpt[group].items():
            assert torch.equal(out[group][k], v)
    assert torch.equal(out["lambda_sq"], ckpt["lambda_sq"])

    # JSON-native metadata preserved exactly.
    assert out["architecture"] == ckpt["architecture"]
    assert out["model_type"] == "faraday_thin"
    assert out["n_components"] == 3
    assert out["param_names"] == ckpt["param_names"]
    assert out["training_history"] == ckpt["training_history"]


def test_export_is_pickle_free(tmp_path):
    """The exported file must be readable by safetensors' own loader."""
    from safetensors import safe_open

    pt_path = tmp_path / "posterior.pt"
    torch.save(_make_checkpoint(), pt_path)
    st_path = pt_to_safetensors(pt_path, tmp_path / "out.safetensors")

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    # Flattened group keys present, no pickle needed to read them.
    assert any(k.startswith("density_estimator_state::") for k in keys)


def test_oddball_metadata_does_not_break_export(tmp_path):
    ckpt = _make_checkpoint()
    ckpt["torch_version"] = torch.__version__  # TorchVersion, not JSON-native
    pt_path = tmp_path / "posterior.pt"
    torch.save(ckpt, pt_path)

    st_path = pt_to_safetensors(pt_path, tmp_path / "out.safetensors")
    out = load_safetensors_as_dict(st_path)
    # Coerced to string via default=str; conversion must not raise.
    assert str(out["torch_version"]) == str(torch.__version__)


def test_non_dict_checkpoint_raises(tmp_path):
    pt_path = tmp_path / "bad.pt"
    torch.save(torch.randn(3), pt_path)
    with pytest.raises(ValueError):
        pt_to_safetensors(pt_path, tmp_path / "out.safetensors")


def test_tensorless_checkpoint_raises(tmp_path):
    pt_path = tmp_path / "empty.pt"
    torch.save({"model_type": "faraday_thin", "n_components": 1}, pt_path)
    with pytest.raises(ValueError):
        pt_to_safetensors(pt_path, tmp_path / "out.safetensors")


def test_is_safetensors():
    assert is_safetensors("a/b/model.safetensors")
    assert not is_safetensors("a/b/model.pt")
