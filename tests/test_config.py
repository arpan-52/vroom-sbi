"""
Tests for src/config/validators.py and src/config/configuration.py.
"""

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

FREQ_FILE = str(Path(__file__).parent.parent / "freq.txt")


def _make_config(freq_file: str = FREQ_FILE):
    from src.config.configuration import Configuration

    return Configuration(freq_file=freq_file)


# ---------------------------------------------------------------------------
# validate_config / print_config_summary
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_returns_warnings_list_on_valid_config(self):
        from src.config.validators import validate_config

        result = validate_config(_make_config())
        assert isinstance(result, list)

    def test_summary_does_not_raise(self):
        from src.config.validators import print_config_summary

        buf = io.StringIO()
        sys.stdout = buf
        try:
            print_config_summary(_make_config())
        finally:
            sys.stdout = sys.__stdout__

    def test_summary_shows_sigma_range_not_base_level(self):
        from src.config.validators import print_config_summary

        buf = io.StringIO()
        sys.stdout = buf
        print_config_summary(_make_config())
        sys.stdout = sys.__stdout__
        output = buf.getvalue()
        assert "sigma" in output.lower()
        assert "base_level" not in output


# ---------------------------------------------------------------------------
# Configuration.to_dict
# ---------------------------------------------------------------------------


class TestToDictSpectralShape:
    def test_to_dict_returns_dict(self):
        assert isinstance(_make_config().to_dict(), dict)

    def test_spectral_shape_block_has_no_log_F0_keys(self):
        ss = _make_config().to_dict().get("spectral_shape", {})
        assert "log_F0_min" not in ss
        assert "log_F0_max" not in ss


# ---------------------------------------------------------------------------
# TrainingConfig.get_scaled_simulations
# ---------------------------------------------------------------------------


class TestScalingModes:
    def _cfg(self, mode):
        from src.config.configuration import TrainingConfig

        return TrainingConfig(
            simulation_scaling=True,
            simulation_scaling_mode=mode,
            n_simulations=1000,
        )

    def test_quadratic_mode_not_active(self):
        n = 3
        result = self._cfg("quadratic").get_scaled_simulations(n)
        assert result != 1000 * n**2

    def test_subquadratic_mode_not_active(self):
        n = 3
        result = self._cfg("subquadratic").get_scaled_simulations(n)
        assert result != int(1000 * n**1.5)

    def test_power_mode_still_works(self):
        from src.config.configuration import TrainingConfig

        cfg = TrainingConfig(
            simulation_scaling=True,
            simulation_scaling_mode="power",
            scaling_power=2.0,
            n_simulations=1000,
        )
        assert cfg.get_scaled_simulations(3) == 1000 * 3**2
