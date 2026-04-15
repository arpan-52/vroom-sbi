"""
HuggingFace Hub integration for VROOM-SBI.

Allows pushing and pulling trained models to/from HuggingFace Hub.
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def push_to_hub(
    model_dir: Path,
    repo_id: str,
    token: str,
    private: bool = False,
    commit_message: str | None = None,
    model_types: list[str] | None = None,
    max_components: int | None = None,
):
    """
    Push trained models to HuggingFace Hub.

    Parameters
    ----------
    model_dir : Path
        Directory containing trained models
    repo_id : str
        HuggingFace repo ID (e.g., "username/vroom-sbi-models")
    token : str
        HuggingFace API token
    private : bool
        Whether to make the repo private
    commit_message : str, optional
        Commit message
    model_types : List[str], optional
        Model types to upload (None = all)
    max_components : int, optional
        Max components to upload (None = all)
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface_hub package required. Install with: pip install huggingface_hub"
        )

    model_dir = Path(model_dir)
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, token=token, exist_ok=True)
        logger.info(f"Repository ready: {repo_id}")
    except Exception as e:
        logger.warning(f"Could not create repo (may already exist): {e}")

    # Find all model files
    files_to_upload = []

    # Posterior models
    for f in model_dir.glob("posterior_*.pt"):
        if model_types is not None:
            # Check if model type matches
            model_type = f.stem.split("_")[1]
            if model_type not in model_types:
                continue
        if max_components is not None:
            # Check component count
            n_comp = int(f.stem.split("_n")[-1])
            if n_comp > max_components:
                continue
        files_to_upload.append(f)

    # Legacy .pkl files
    for f in model_dir.glob("posterior_*.pkl"):
        files_to_upload.append(f)

    # Classifier
    for f in model_dir.glob("classifier.*"):
        files_to_upload.append(f)

    # Simulations (optional, can be large)
    # for f in model_dir.glob("simulations_*.pt"):
    #     files_to_upload.append(f)

    # Training plots
    for f in model_dir.glob("*.png"):
        files_to_upload.append(f)

    # Summary files
    for f in model_dir.glob("*.txt"):
        files_to_upload.append(f)

    if not files_to_upload:
        logger.warning(f"No files found in {model_dir}")
        return

    logger.info(f"Uploading {len(files_to_upload)} files to {repo_id}")

    # Create model card
    model_card = _create_model_card(model_dir, files_to_upload)
    model_card_path = model_dir / "README.md"
    with open(model_card_path, "w") as f:
        f.write(model_card)
    files_to_upload.append(model_card_path)

    # Upload files
    for f in files_to_upload:
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message or f"Upload {f.name}",
            )
            logger.info(f"  Uploaded: {f.name}")
        except Exception as e:
            logger.error(f"  Failed to upload {f.name}: {e}")

    logger.info(f"\nModels uploaded to: https://huggingface.co/{repo_id}")


def pull_from_hub(
    repo_id: str,
    output_dir: Path,
    token: str | None = None,
    revision: str = "main",
):
    """
    Pull trained models from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo ID
    output_dir : Path
        Directory to save models
    token : str, optional
        HuggingFace API token (for private repos)
    revision : str
        Git revision to pull
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub package required. Install with: pip install huggingface_hub"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading models from {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        token=token,
        revision=revision,
    )

    logger.info(f"Models downloaded to {output_dir}")


def _create_model_card(model_dir: Path, files: list[Path]) -> str:
    """Create a model card for HuggingFace."""

    # Count models
    posteriors = [f for f in files if f.name.startswith("posterior_")]
    has_classifier = any(f.name.startswith("classifier") for f in files)

    # Extract model types and components
    model_types = set()
    max_components = 0
    for f in posteriors:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            model_types.add(parts[1])
            n = int(parts[-1].replace("n", ""))
            max_components = max(max_components, n)

    card = f"""---
tags:
- radio-astronomy
- polarization
- rm-synthesis
- simulation-based-inference
- pytorch
license: mit
---

# VROOM-SBI Trained Models

Trained neural posterior estimators for Rotation Measure (RM) synthesis.

## Model Information

- **Model Types**: {", ".join(sorted(model_types)) if model_types else "N/A"}
- **Max Components**: {max_components}
- **Classifier**: {"Yes" if has_classifier else "No"}
- **Upload Date**: {datetime.now().strftime("%Y-%m-%d")}

## Files

| File | Description |
|------|-------------|
"""

    for f in sorted(files, key=lambda x: x.name):
        if f.name.startswith("posterior_"):
            desc = f"Posterior model ({f.suffix})"
        elif f.name.startswith("classifier"):
            desc = "Model selection classifier"
        elif f.name.startswith("simulations_"):
            desc = "Training simulations"
        elif f.suffix == ".png":
            desc = "Training plot"
        elif f.suffix == ".txt":
            desc = "Summary file"
        else:
            desc = ""
        card += f"| {f.name} | {desc} |\n"

    card += """

## Usage

```python
from vroom_sbi.inference import InferenceEngine

# Load models
engine = InferenceEngine(model_dir="path/to/downloaded/models")
engine.load_models()

# Run inference
result, all_results = engine.infer(qu_obs)
print(f"Best model: {result.n_components} components")
```

## Citation

If you use these models, please cite:

```
[Citation information to be added]
```
"""

    return card
