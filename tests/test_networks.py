"""
Tests for src/training/networks.py.
"""

import src.training.networks as nets


class TestNetworksDeadCodeRemoved:
    def test_deep_spectral_classifier_removed(self):
        assert not hasattr(nets, "DeepSpectralClassifier")

    def test_residual_block_removed(self):
        assert not hasattr(nets, "ResidualBlock")

    def test_spectral_classifier_still_present(self):
        assert hasattr(nets, "SpectralClassifier")

    def test_spectral_embedding_still_present(self):
        assert hasattr(nets, "SpectralEmbedding")
