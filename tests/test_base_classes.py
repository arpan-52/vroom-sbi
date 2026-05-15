"""
Tests for src/core/base_classes.py.
"""

import src.core.base_classes as bc
from src.core.base_classes import InferenceEngineInterface


class TestBaseClasses:
    def test_posterior_interface_removed(self):
        assert not hasattr(bc, "PosteriorInterface")

    def test_inference_engine_interface_abstract_methods_match_engine(self):
        abstract = InferenceEngineInterface.__abstractmethods__
        assert "run_inference" not in abstract
        assert "get_model_for_n" not in abstract

    def test_inference_engine_interface_declares_infer(self):
        assert "infer" in InferenceEngineInterface.__abstractmethods__

    def test_inference_engine_interface_declares_load_models(self):
        assert "load_models" in InferenceEngineInterface.__abstractmethods__

    def test_base_simulator_still_present(self):
        assert hasattr(bc, "BaseSimulator")
