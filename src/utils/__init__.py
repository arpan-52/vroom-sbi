"""
Utility functions for VROOM-SBI.
"""

import logging
import os
from pathlib import Path

from .validation import validate_all_models

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "skunkworks-ra/vroomsbi"

__all__ = [
    "DEFAULT_HF_REPO",
    "get_huggingface_token",
    "push_to_huggingface",
    "download_from_huggingface",
    "setup_logging",
    "validate_all_models",
]


def get_huggingface_token() -> str | None:
    """Get HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def push_to_huggingface(
    model_dir: str,
    repo_id: str,
    token: str | None = None,
    private: bool = True,
):
    """
    Push trained models to HuggingFace Hub.

    Parameters
    ----------
    model_dir : str
        Directory containing trained models
    repo_id : str
        HuggingFace repository ID (e.g., "username/vroom-sbi-models")
    token : str, optional
        HuggingFace token. If None, uses HF_TOKEN environment variable.
    private : bool
        Whether to create a private repository
    """
    from .huggingface import push_to_hub

    if token is None:
        token = get_huggingface_token()

    if token is None:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN environment variable or pass token parameter."
        )

    push_to_hub(model_dir=Path(model_dir), repo_id=repo_id, token=token, private=private)


def download_from_huggingface(
    repo_id: str,
    model_dir: str = "models",
    token: str | None = None,
):
    """
    Download models from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID
    model_dir : str
        Local directory to save models
    token : str, optional
        HuggingFace token for private repositories
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    if token is None:
        token = get_huggingface_token()

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        token=token,
    )

    logger.info(f"Models downloaded to: {model_dir}")


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Configure logging for VROOM-SBI."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
