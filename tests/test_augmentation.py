"""
Tests for src/simulator/augmentation.py.
"""

from unittest.mock import patch

import numpy as np

import src.simulator.augmentation as aug_mod


class TestGenerateAugmentedWeightsBatch:
    BASE = np.ones(32)

    def test_continuous_mode_calls_continuous_fn(self):
        with (
            patch.object(aug_mod, "augment_weights_continuous", wraps=aug_mod.augment_weights_continuous) as mock_cont,
            patch.object(aug_mod, "augment_weights_combined", wraps=aug_mod.augment_weights_combined) as mock_comb,
        ):
            aug_mod.generate_augmented_weights_batch(self.BASE, batch_size=4, continuous_weights=True)

        assert mock_cont.call_count == 4
        assert mock_comb.call_count == 0

    def test_legacy_mode_calls_combined_fn(self):
        with (
            patch.object(aug_mod, "augment_weights_continuous", wraps=aug_mod.augment_weights_continuous) as mock_cont,
            patch.object(aug_mod, "augment_weights_combined", wraps=aug_mod.augment_weights_combined) as mock_comb,
        ):
            aug_mod.generate_augmented_weights_batch(self.BASE, batch_size=3, continuous_weights=False)

        assert mock_comb.call_count == 3
        assert mock_cont.call_count == 0

    def test_output_shape(self):
        out = aug_mod.generate_augmented_weights_batch(self.BASE, batch_size=5)
        assert out.shape == (5, len(self.BASE))

    def test_default_is_continuous(self):
        with patch.object(aug_mod, "augment_weights_continuous", wraps=aug_mod.augment_weights_continuous) as mock_cont:
            aug_mod.generate_augmented_weights_batch(self.BASE, batch_size=2)
        assert mock_cont.call_count == 2
