"""
Tests for src/core/checkpoint.py.
"""

from datetime import datetime
from unittest.mock import patch

from src.core.checkpoint import CheckpointManager, ModelCheckpoint


def _make_checkpoint(**overrides):
    defaults = dict(
        model_type="faraday_thin",
        n_components=1,
        n_params=3,
        n_freq=64,
        epoch=10,
        state_dict={},
        optimizer_state=None,
        scheduler_state=None,
        train_loss=0.5,
        val_loss=0.4,
        best_val_loss=0.4,
        loss_history={},
        timestamp=datetime.now(),
        config_snapshot={},
        prior_bounds={},
        embedding_net_state=None,
    )
    defaults.update(overrides)
    return ModelCheckpoint(**defaults)


class TestCheckpointManager:
    def test_torch_save_called_exactly_once_when_is_best(self, tmp_path):
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        with patch("src.core.checkpoint.torch.save") as mock_save:
            manager.save_checkpoint(_make_checkpoint(), is_best=True)
        assert mock_save.call_count == 1, (
            f"torch.save called {mock_save.call_count} times; expected 1"
        )

    def test_torch_save_called_once_for_regular_checkpoint(self, tmp_path):
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        with patch("src.core.checkpoint.torch.save") as mock_save:
            manager.save_checkpoint(_make_checkpoint(), is_best=False)
        assert mock_save.call_count == 1
