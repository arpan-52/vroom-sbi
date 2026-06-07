"""
Pickle-free conversion between the working ``.pt`` checkpoints and the
``.safetensors`` distribution format.

Why: ``torch.save`` writes a pickle stream, so Hugging Face's ``picklescan``
flags uploaded ``.pt`` files as potentially dangerous. ``safetensors`` stores
only raw tensors plus a string→string metadata header — no executable code —
so it passes the scanner cleanly. We keep ``.pt`` as the local working format
and convert to ``.safetensors`` only at push time (see ``utils.huggingface``).

Layout of a converted file:
  - tensor leaves of the checkpoint dict go in the safetensors body. Nested
    state-dict groups (``density_estimator_state`` etc.) are flattened with a
    ``"<group>::<param>"`` key so they regroup losslessly on load.
  - every non-tensor value (scalars, lists, nested config dicts) is collected
    into one JSON blob stored under the ``__json__`` metadata key. ``default=str``
    handles oddballs like ``torch.version.TorchVersion``.

``load_safetensors_as_dict`` is the exact inverse: it returns a dict equivalent
to what ``torch.load`` would have produced, so existing rebuild paths
(``engine._rebuild_posterior_from_statedict``, ``ClassifierTrainer.load``) work
unchanged.
"""

import json
from pathlib import Path
from typing import Any

import torch

# Separator between a state-dict group name and its parameter key. Chosen to
# avoid collision with the dotted module paths inside a state-dict.
_GROUP_SEP = "::"
_JSON_KEY = "__json__"
_FORMAT_KEY = "__vroom_format__"
_FORMAT_VAL = "vroom-safetensors-v1"


def _split_tensors_and_meta(
    data: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Partition a checkpoint dict into flat tensors and JSON-able metadata."""
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value.detach().contiguous().cpu()
        elif (
            isinstance(value, dict)
            and value
            and all(isinstance(v, torch.Tensor) for v in value.values())
        ):
            # A state-dict group: flatten with a prefixed key.
            for sub_key, sub_val in value.items():
                tensors[f"{key}{_GROUP_SEP}{sub_key}"] = (
                    sub_val.detach().contiguous().cpu()
                )
            meta.setdefault("__tensor_groups__", []).append(key)
        else:
            meta[key] = value
    return tensors, meta


def pt_to_safetensors(pt_path: str | Path, out_path: str | Path) -> Path:
    """Convert a ``.pt`` checkpoint to ``.safetensors``. Returns the output path."""
    from safetensors.torch import save_file

    pt_path = Path(pt_path)
    out_path = Path(out_path)

    # weights_only=False: the source is the user's own trusted local file, and
    # it carries non-tensor config (TorchVersion, dicts) we need to read.
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(
            f"{pt_path} does not hold a checkpoint dict (got {type(data).__name__}); "
            "cannot convert to safetensors."
        )

    tensors, meta = _split_tensors_and_meta(data)
    if not tensors:
        raise ValueError(f"No tensors found in {pt_path}; nothing to save.")

    header = {
        _FORMAT_KEY: _FORMAT_VAL,
        _JSON_KEY: json.dumps(meta, default=str),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=header)
    return out_path


def load_safetensors_as_dict(
    path: str | Path, device: str = "cpu"
) -> dict[str, Any]:
    """Inverse of ``pt_to_safetensors``: reconstruct the original checkpoint dict."""
    from safetensors import safe_open

    path = Path(path)
    data: dict[str, Any] = {}
    groups: dict[str, dict[str, torch.Tensor]] = {}

    with safe_open(str(path), framework="pt", device=device) as f:
        header = f.metadata() or {}
        meta = json.loads(header.get(_JSON_KEY, "{}"))
        group_names = set(meta.pop("__tensor_groups__", []))

        for key in f.keys():
            tensor = f.get_tensor(key)
            if _GROUP_SEP in key:
                group, sub_key = key.split(_GROUP_SEP, 1)
                groups.setdefault(group, {})[sub_key] = tensor
            else:
                data[key] = tensor

    # Reattach flattened groups (e.g. density_estimator_state) and scalar meta.
    for group, state in groups.items():
        data[group] = state
    # Sanity: declared groups should have materialized.
    missing = group_names - set(groups)
    if missing:
        raise ValueError(f"{path}: missing tensor groups {sorted(missing)}")
    data.update(meta)
    return data


def is_safetensors(path: str | Path) -> bool:
    return Path(path).suffix == ".safetensors"
