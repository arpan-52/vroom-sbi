"""
Inference engine for VROOM-SBI.

Handles model loading and posterior inference with memory management.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import Configuration, MemoryConfig
from ..core.base_classes import InferenceEngineInterface
from ..core.result import ClassifierResult, ComponentResult, InferenceResult
from ..simulator.prior import get_params_per_component, sort_posterior_samples

logger = logging.getLogger(__name__)


def load_posterior(model_path: Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """
    Load a trained posterior from disk and move to device.

    CRITICAL: For SBI posteriors with rejection sampling, we need to move:
    1. The posterior's neural network
    2. The prior's bounds (used for rejection sampling support check)
    """
    model_path = Path(model_path)

    # Always load to CPU first, then move
    if model_path.suffix == ".pt":
        data = torch.load(model_path, map_location="cpu", weights_only=False)
        posterior = data.get("posterior") or data.get("posterior_object")
        if posterior is None:
            raise ValueError(f"No posterior object found in {model_path}")
    else:
        import pickle

        with open(model_path, "rb") as f:
            data = pickle.load(f)
        posterior = data["posterior"]

    # Move everything to device
    if device != "cpu":
        # 1. Move neural network
        if hasattr(posterior, "posterior_estimator"):
            posterior.posterior_estimator = posterior.posterior_estimator.to(device)
        if hasattr(posterior, "_neural_net"):
            posterior._neural_net = posterior._neural_net.to(device)

        # 2. Move prior bounds (CRITICAL for rejection sampling)
        def move_prior_to_device(prior, dev):
            if prior is None:
                return
            if hasattr(prior, "base_dist"):
                bd = prior.base_dist
                if hasattr(bd, "low") and hasattr(bd, "high"):
                    bd.low = bd.low.to(dev)
                    bd.high = bd.high.to(dev)
            elif hasattr(prior, "low") and hasattr(prior, "high"):
                prior.low = prior.low.to(dev)
                prior.high = prior.high.to(dev)

        if hasattr(posterior, "_prior"):
            move_prior_to_device(posterior._prior, device)
        if hasattr(posterior, "prior"):
            move_prior_to_device(posterior.prior, device)

        # 3. Set device attribute
        if hasattr(posterior, "_device"):
            posterior._device = device

        # 4. Try generic .to() - BUT DON'T reassign if it returns None!
        if hasattr(posterior, "to"):
            try:
                result = posterior.to(device)
                if result is not None:
                    posterior = result
            except Exception:
                pass

    return posterior, data


def load_classifier(model_path: Path, device: str = "cpu") -> Any:
    """Load a trained classifier."""
    from ..training.classifier_trainer import ClassifierTrainer

    trainer = ClassifierTrainer(n_freq=1, n_classes=1, device=device)
    trainer.load(str(model_path))
    return trainer


class InferenceEngine(InferenceEngineInterface):
    """Main inference engine for VROOM-SBI."""

    def __init__(
        self,
        config: Configuration | None = None,
        model_dir: str = "models",
        device: str = "cuda",
    ):
        self.config = config
        self.model_dir = Path(model_dir)
        self.device = device

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            self.device = "cpu"

        self.posteriors: dict[str, Any] = {}
        self.posterior_metadata: dict[str, dict] = {}
        self.model_lambda_sq: dict[str, np.ndarray] = {}
        self.classifier = None
        self.memory_config = config.memory if config else MemoryConfig()
        self._models_on_device: dict[str, str] = {}

    def load_models(
        self, max_components: int = 5, model_types: list[str] | None = None
    ):
        """Load trained posterior models."""
        if model_types is None:
            model_types = (
                self.config.physics.model_types if self.config else ["faraday_thin"]
            )

        logger.info(f"Loading models from {self.model_dir}")

        for model_type in model_types:
            for n in range(1, max_components + 1):
                model_path = self.model_dir / f"posterior_{model_type}_n{n}.pt"
                if not model_path.exists():
                    model_path = self.model_dir / f"posterior_{model_type}_n{n}.pkl"

                if model_path.exists():
                    try:
                        posterior, metadata = load_posterior(model_path, self.device)
                        key = f"{model_type}_n{n}"
                        self.posteriors[key] = posterior
                        self.posterior_metadata[key] = metadata
                        self._models_on_device[key] = self.device
                        if "lambda_sq" in metadata:
                            self.model_lambda_sq[key] = np.asarray(metadata["lambda_sq"])
                        logger.info(f"  Loaded {key}")
                    except Exception as e:
                        logger.warning(f"  Failed to load {model_path}: {e}")

        # Load classifier
        for ext in [".pt", ".pkl"]:
            classifier_path = self.model_dir / f"classifier{ext}"
            if classifier_path.exists():
                try:
                    self.classifier = load_classifier(classifier_path, self.device)
                    logger.info("  Loaded classifier")
                    break
                except Exception as e:
                    logger.warning(f"  Failed to load classifier: {e}")

        logger.info(f"Loaded {len(self.posteriors)} posterior models")

    def get_model_for_n(
        self, n_components: int, model_type: str = "faraday_thin"
    ) -> Any | None:
        """Get posterior for given configuration."""
        return self.posteriors.get(f"{model_type}_n{n_components}")

    def _get_input_channels(self, key: str) -> int:
        """Return expected input channels (2 or 3) for a model key."""
        meta = self.posterior_metadata.get(key, {})
        return int(meta.get("input_channels", 2))

    def _build_observation(
        self, qu_obs: np.ndarray, weights: np.ndarray | None, key: str
    ) -> np.ndarray:
        """
        Build the observation vector for the posterior.

        For new 3-channel models ([Q, U, w]): append per-channel weights.
        For legacy 2-channel models ([Q, U]):  return qu_obs unchanged.
        Weights are clipped to [0,1] and zero where flagged.
        """
        n_chan = self._get_input_channels(key)
        if n_chan == 3:
            n_freq = len(qu_obs) // 2
            if weights is not None and len(weights) >= n_freq:
                w = np.clip(weights[:n_freq], 0.0, 1.0).astype(np.float32)
            else:
                # No weights provided — assume all good channels equally weighted
                w = (np.abs(qu_obs[:n_freq]) > 0).astype(np.float32)
            return np.concatenate([qu_obs, w])
        return qu_obs

    def run_inference(
        self,
        qu_obs: np.ndarray,
        weights: np.ndarray | None = None,
        n_samples: int = 10000,
        use_classifier: bool = True,
        model_type: str | None = None,
    ) -> tuple[dict[str, InferenceResult], str]:
        """Run inference on observed spectrum."""
        start_time = datetime.now()
        # qu_obs_t built per-key below (may append weights for 3-ch models)

        if use_classifier and self.classifier is not None:
            classifier_result = self._run_classifier(qu_obs, weights)
            best_n = classifier_result.predicted_n_components
            best_model_type = (
                classifier_result.predicted_model_type or model_type or "faraday_thin"
            )
            selected_keys = [f"{best_model_type}_n{best_n}"]
            logger.info(f"Classifier selected: {best_model_type} N={best_n}")
        else:
            selected_keys = list(self.posteriors.keys())
            if model_type:
                selected_keys = [k for k in selected_keys if k.startswith(model_type)]

        results = {}
        for key in selected_keys:
            if key not in self.posteriors:
                continue

            posterior = self.posteriors[key]
            model_type_key = key.rsplit("_n", 1)[0]
            n_components = int(key.rsplit("_n", 1)[1])

            logger.debug(f"Running inference for {key}...")

            # Build per-key observation (appends weights for 3-channel models)
            obs = self._build_observation(qu_obs, weights, key)
            qu_obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

            samples = posterior.sample((n_samples,), x=qu_obs_t)
            samples_np = samples.cpu().numpy()

            params_per_comp = get_params_per_component(model_type_key)
            samples_np = sort_posterior_samples(
                samples_np, n_components, params_per_comp
            )

            log_probs = posterior.log_prob(samples, x=qu_obs_t)
            log_evidence = (
                (torch.logsumexp(log_probs, dim=0) - np.log(n_samples)).cpu().item()
            )

            components = self._parse_components(
                samples_np, n_components, model_type_key
            )

            results[key] = InferenceResult(
                n_components=n_components,
                model_type=model_type_key,
                log_evidence=log_evidence,
                components=components,
                all_samples=samples_np,
                n_posterior_samples=n_samples,
                inference_time_seconds=(datetime.now() - start_time).total_seconds(),
            )

        best_key = (
            max(results.keys(), key=lambda k: results[k].log_evidence)
            if results
            else None
        )
        return results, best_key

    def _run_classifier(
        self, qu_obs: np.ndarray, weights: np.ndarray | None = None
    ) -> ClassifierResult:
        """Run classifier on spectrum."""
        n_freq = len(qu_obs) // 2
        if weights is None:
            weights = np.ones(n_freq)

        classifier_input = np.concatenate([qu_obs, weights])
        classifier_input_t = torch.tensor(
            classifier_input, dtype=torch.float32, device=self.device
        )
        n_comp, prob_dict = self.classifier.predict(classifier_input_t)

        return ClassifierResult(
            predicted_n_components=n_comp,
            predicted_model_type=None,
            probabilities={str(k): v for k, v in prob_dict.items()},
            confidence=max(prob_dict.values()),
        )

    def _parse_components(
        self, samples: np.ndarray, n_components: int, model_type: str
    ) -> list[ComponentResult]:
        """Parse posterior samples into component results."""
        params_per_comp = get_params_per_component(model_type)
        components = []

        for i in range(n_components):
            base_idx = i * params_per_comp
            rm_samples = samples[:, base_idx]

            if model_type == "faraday_thin":
                amp_samples = samples[:, base_idx + 1]
                chi0_samples = samples[:, base_idx + 2]
                q_samples = amp_samples * np.cos(2 * chi0_samples)
                u_samples = amp_samples * np.sin(2 * chi0_samples)

                component = ComponentResult(
                    rm_mean=np.mean(rm_samples),
                    rm_std=np.std(rm_samples),
                    q_mean=np.mean(q_samples),
                    q_std=np.std(q_samples),
                    u_mean=np.mean(u_samples),
                    u_std=np.std(u_samples),
                    samples=np.column_stack([rm_samples, amp_samples, chi0_samples]),
                    chi0_mean=np.mean(chi0_samples),
                    chi0_std=np.std(chi0_samples),
                )
            else:
                second_param = samples[:, base_idx + 1]
                amp_samples = samples[:, base_idx + 2]
                chi0_samples = samples[:, base_idx + 3]
                q_samples = amp_samples * np.cos(2 * chi0_samples)
                u_samples = amp_samples * np.sin(2 * chi0_samples)

                component = ComponentResult(
                    rm_mean=np.mean(rm_samples),
                    rm_std=np.std(rm_samples),
                    q_mean=np.mean(q_samples),
                    q_std=np.std(q_samples),
                    u_mean=np.mean(u_samples),
                    u_std=np.std(u_samples),
                    samples=np.column_stack(
                        [rm_samples, second_param, amp_samples, chi0_samples]
                    ),
                    chi0_mean=np.mean(chi0_samples),
                    chi0_std=np.std(chi0_samples),
                )

                if model_type == "burn_slab":
                    component.delta_phi_mean = np.mean(second_param)
                    component.delta_phi_std = np.std(second_param)
                else:
                    component.sigma_phi_mean = np.mean(second_param)
                    component.sigma_phi_std = np.std(second_param)

            components.append(component)

        return components

    def infer(
        self,
        qu_obs: np.ndarray,
        weights: np.ndarray | None = None,
        n_samples: int = 10000,
    ):
        """Convenience method - run inference and return best result."""
        results, best_key = self.run_inference(
            qu_obs, weights=weights, n_samples=n_samples
        )
        if best_key is None:
            raise ValueError("No models available for inference")
        return results[best_key], results

    def _check_frequency_compatibility(self, frequencies_hz: np.ndarray) -> None:
        """Warn if cube frequencies differ from the model's training frequency grid."""
        from ..simulator.physics import freq_to_lambda_sq

        cube_lsq = freq_to_lambda_sq(frequencies_hz)
        for key, model_lsq in self.model_lambda_sq.items():
            if len(cube_lsq) != len(model_lsq):
                logger.warning(
                    f"Frequency mismatch for model '{key}': "
                    f"expected {len(model_lsq)} channels, got {len(cube_lsq)}. "
                    f"Cube range: {frequencies_hz[0]/1e6:.1f}–{frequencies_hz[-1]/1e6:.1f} MHz."
                )
            elif not np.allclose(cube_lsq, model_lsq, rtol=1e-3):
                logger.warning(
                    f"Frequency values for model '{key}' differ from cube "
                    f"(rtol=1e-3). Results may be unreliable. "
                    f"Cube range: {frequencies_hz[0]/1e6:.1f}–{frequencies_hz[-1]/1e6:.1f} MHz."
                )

    def _infer_batch_key(
        self,
        x_batch: np.ndarray,
        key: str,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run batched inference for a specific model key.

        Parameters
        ----------
        x_batch : np.ndarray, shape (B, 2*n_freq)
        key : str
        n_samples : int

        Returns
        -------
        samples_np : np.ndarray, shape (B, n_samples, n_params)
        log_evidence : np.ndarray, shape (B,)
        """
        B = x_batch.shape[0]
        posterior = self.posteriors[key]
        model_type_key = key.rsplit("_n", 1)[0]
        n_components = int(key.rsplit("_n", 1)[1])
        params_per_comp = get_params_per_component(model_type_key)

        x_t = torch.tensor(x_batch, dtype=torch.float32, device=self.device)  # (B, n_obs)

        # sbi 0.25: use sample_batched for multiple observations
        # returns (B, n_samples, n_params)
        samples = posterior.sample_batched(
            (n_samples,), x=x_t, show_progress_bars=False
        )
        samples_np = samples.cpu().numpy()  # (B, n_samples, n_params)

        # Sort components per pixel (no-op for n_components=1)
        if n_components > 1:
            for i in range(B):
                samples_np[i] = sort_posterior_samples(
                    samples_np[i], n_components, params_per_comp
                )

        # Log evidence per pixel via log_prob_batched
        # samples: (B, n_samples, n_params) -> log_probs: (B, n_samples)
        log_probs = posterior.log_prob_batched(samples, x=x_t)  # (B, n_samples)
        log_evidence = (
            torch.logsumexp(log_probs, dim=1) - np.log(n_samples)
        ).cpu().numpy()  # (B,)

        return samples_np, log_evidence

    def run_inference_cube(
        self,
        q_data: np.ndarray,
        u_data: np.ndarray,
        weights: np.ndarray | None = None,
        n_samples: int = 1000,
        batch_size: int = 1,
        mask: np.ndarray | None = None,
        snr_threshold: float | None = None,
        frequencies_hz: np.ndarray | None = None,
        **infer_kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Run inference over all valid spatial pixels of a spectral cube.

        Parameters
        ----------
        q_data, u_data : np.ndarray, shape (n_freq, n_dec, n_ra)
            Stokes Q and U cubes (or Q/I, U/I fractional polarisation).
        weights : np.ndarray, optional
            Inverse-variance channel weights.
            - ``(n_freq,)``            → same weights for every pixel.
            - ``(n_freq, n_dec, n_ra)`` → per-pixel weights.
            If *None*, uniform weights are used (existing engine default).
        n_samples : int
            Number of posterior samples per pixel.
        batch_size : int
            Pixels per progress-bar chunk. Default 1 (serial).  Higher
            values only affect display granularity; each pixel still calls
            ``self.infer()`` once.
        mask : np.ndarray, optional
            Boolean 2D array shape ``(n_dec, n_ra)``.  Only pixels where
            ``mask == True`` are processed.
        snr_threshold : float, optional
            If given, only pixels whose mean polarised SNR exceeds this
            value are processed.  SNR = mean(|P|) / std(Q) per pixel.
        **infer_kwargs
            Forwarded to :meth:`run_inference` (e.g. ``use_classifier``,
            ``model_type``).

        Returns
        -------
        results : dict
            ``{name: 2D np.ndarray}`` of shape ``(n_dec, n_ra)``.
            Unprocessed pixels are NaN.  Keys follow the convention
            ``rm_mean_comp1``, ``rm_std_comp1``, ``rm_p16_comp1``, …
        """
        from tqdm import tqdm

        if q_data.shape != u_data.shape:
            raise ValueError(
                f"q_data and u_data must have the same shape; "
                f"got {q_data.shape} and {u_data.shape}"
            )

        if frequencies_hz is not None and self.model_lambda_sq:
            self._check_frequency_compatibility(frequencies_hz)

        n_freq, n_dec, n_ra = q_data.shape

        # ------------------------------------------------------------------
        # Determine maximum number of components across loaded models
        # ------------------------------------------------------------------
        if self.posteriors:
            max_n = max(int(k.rsplit("_n", 1)[1]) for k in self.posteriors.keys())
        else:
            raise ValueError("No posterior models loaded. Call load_models() first.")

        # ------------------------------------------------------------------
        # Pre-allocate output arrays (NaN = not processed / masked)
        # ------------------------------------------------------------------
        results: dict[str, np.ndarray] = {}
        shape2d = (n_dec, n_ra)

        def _nan():
            return np.full(shape2d, np.nan, dtype=np.float32)

        for comp in range(1, max_n + 1):
            tag = f"comp{comp}"
            for stat in ("mean", "std", "p16", "p84"):
                results[f"rm_{stat}_{tag}"] = _nan()
                results[f"amp_{stat}_{tag}"] = _nan()
                results[f"chi0_{stat}_{tag}"] = _nan()
                # Extended-model params (NaN for faraday_thin)
                results[f"sigma_phi_{stat}_{tag}"] = _nan()
                results[f"delta_phi_{stat}_{tag}"] = _nan()

        results["log_evidence"] = _nan()
        results["n_components"] = _nan()

        # ------------------------------------------------------------------
        # Build combined pixel mask
        # ------------------------------------------------------------------
        # Layer 1: auto NaN mask — only check channels with non-zero weight
        if weights is not None:
            w = np.asarray(weights)
            good_chans = (w > 0) if w.ndim == 1 else np.any(w > 0, axis=(1, 2))
            q_check = q_data[good_chans]
            u_check = u_data[good_chans]
        else:
            q_check = q_data
            u_check = u_data
        valid = ~np.any(np.isnan(q_check) | np.isnan(u_check), axis=0)

        # Layer 2: user spatial mask
        if mask is not None:
            if mask.shape != shape2d:
                raise ValueError(
                    f"mask shape {mask.shape} does not match spatial dimensions {shape2d}"
                )
            valid = valid & mask.astype(bool)

        # Layer 3: P-map collapse — only process pixels with significant
        # polarised emission.  Use good channels only (weight > 0).
        if weights is not None:
            w1d = weights if weights.ndim == 1 else np.any(weights > 0, axis=(1, 2))
            good_chans = w1d > 0
        else:
            good_chans = slice(None)
        p_map = np.nanmean(
            np.sqrt(q_data[good_chans] ** 2 + u_data[good_chans] ** 2), axis=0
        )  # (n_dec, n_ra)
        logger.info(
            f"P map: min={np.nanmin(p_map):.4f}, max={np.nanmax(p_map):.4f}, "
            f"mean={np.nanmean(p_map):.4f}"
        )

        # Estimate noise in the collapsed P map via MAD (robust to bright sources)
        p_median = np.nanmedian(p_map)
        sigma_p = 1.4826 * np.nanmedian(np.abs(p_map - p_median))
        logger.info(f"P map noise (MAD): {sigma_p:.4f}")

        if snr_threshold is not None:
            valid = valid & (p_map >= snr_threshold * sigma_p)
        else:
            valid = valid & (p_map > sigma_p)

        pixel_list = np.argwhere(valid)  # shape (N_valid, 2)
        n_valid = len(pixel_list)
        n_total = n_dec * n_ra
        logger.info(
            f"Cube inference: fitting {n_valid}/{n_total} pixels "
            f"({100 * n_valid / n_total:.1f}%)"
        )

        # ------------------------------------------------------------------
        # Per-pixel inference loop
        # ------------------------------------------------------------------
        for dec_idx, ra_idx in tqdm(pixel_list, desc="Cube inference", unit="px"):
            Q_pix = q_data[:, dec_idx, ra_idx].copy()
            U_pix = u_data[:, dec_idx, ra_idx].copy()

            if weights is not None and weights.ndim == 3:
                pix_weights = weights[:, dec_idx, ra_idx]
            else:
                pix_weights = weights

            # Zero out flagged channels (model trained with 0.0, not NaN)
            if pix_weights is not None:
                Q_pix[pix_weights == 0] = 0.0
                U_pix[pix_weights == 0] = 0.0
            else:
                Q_pix[~np.isfinite(Q_pix)] = 0.0
                U_pix[~np.isfinite(U_pix)] = 0.0

            qu_obs = np.concatenate([Q_pix, U_pix])

            try:
                best_result, _ = self.infer(
                    qu_obs,
                    weights=pix_weights,
                    n_samples=n_samples,
                    **infer_kwargs,
                )
            except Exception as exc:
                logger.warning(f"Pixel ({dec_idx}, {ra_idx}) failed: {exc}")
                continue

            results["log_evidence"][dec_idx, ra_idx] = best_result.log_evidence
            results["n_components"][dec_idx, ra_idx] = best_result.n_components

            for comp_i, comp in enumerate(best_result.components):
                tag = f"comp{comp_i + 1}"

                rm_samp = comp.samples[:, 0]
                results[f"rm_mean_{tag}"][dec_idx, ra_idx] = np.mean(rm_samp)
                results[f"rm_std_{tag}"][dec_idx, ra_idx] = np.std(rm_samp)
                results[f"rm_p16_{tag}"][dec_idx, ra_idx] = np.percentile(rm_samp, 16)
                results[f"rm_p84_{tag}"][dec_idx, ra_idx] = np.percentile(rm_samp, 84)

                amp_col = comp.samples.shape[1] - 2
                amp_samp = comp.samples[:, amp_col]
                results[f"amp_mean_{tag}"][dec_idx, ra_idx] = np.mean(amp_samp)
                results[f"amp_std_{tag}"][dec_idx, ra_idx] = np.std(amp_samp)
                results[f"amp_p16_{tag}"][dec_idx, ra_idx] = np.percentile(amp_samp, 16)
                results[f"amp_p84_{tag}"][dec_idx, ra_idx] = np.percentile(amp_samp, 84)

                chi0_samp = comp.samples[:, -1]
                results[f"chi0_mean_{tag}"][dec_idx, ra_idx] = np.mean(chi0_samp)
                results[f"chi0_std_{tag}"][dec_idx, ra_idx] = np.std(chi0_samp)
                results[f"chi0_p16_{tag}"][dec_idx, ra_idx] = np.percentile(chi0_samp, 16)
                results[f"chi0_p84_{tag}"][dec_idx, ra_idx] = np.percentile(chi0_samp, 84)

                if comp.samples.shape[1] >= 4:
                    sec_samp = comp.samples[:, 1]
                    if comp.sigma_phi_mean is not None:
                        prefix = "sigma_phi"
                    elif comp.delta_phi_mean is not None:
                        prefix = "delta_phi"
                    else:
                        prefix = "sigma_phi"
                    results[f"{prefix}_mean_{tag}"][dec_idx, ra_idx] = np.mean(sec_samp)
                    results[f"{prefix}_std_{tag}"][dec_idx, ra_idx] = np.std(sec_samp)
                    results[f"{prefix}_p16_{tag}"][dec_idx, ra_idx] = np.percentile(sec_samp, 16)
                    results[f"{prefix}_p84_{tag}"][dec_idx, ra_idx] = np.percentile(sec_samp, 84)

        logger.info("Cube inference complete.")
        return results

    def run_inference_cube_chunked(
        self,
        q_cube,
        u_cube,
        shape: tuple[int, int, int],
        frequencies_hz: np.ndarray,
        snr_threshold: float = 5.0,
        n_samples: int = 1000,
        mem_fraction: float = 0.5,
        i_cube=None,
    ) -> dict[str, np.ndarray]:
        """
        Chunked cube inference — never loads the full cube into RAM.

        Determines chunk size automatically from available RAM.

        Two passes:
          Pass 1 — build collapsed P map + per-channel noise weights chunk by chunk
          Pass 2 — run per-pixel inference on active pixels only

        Parameters
        ----------
        q_cube, u_cube : SpectralCube (lazy)
        shape : (n_freq, n_dec, n_ra)
        frequencies_hz : np.ndarray
        snr_threshold : float
            Keep pixels where mean_P >= snr_threshold * sigma_P (MAD-based)
        n_samples : int
        mem_fraction : float
            Fraction of available RAM to use per chunk (default 0.5)
        """
        import psutil
        from tqdm import tqdm
        from ..io import load_spatial_chunk

        n_freq, n_dec, n_ra = shape

        # ------------------------------------------------------------------
        # Frequency channel reordering: cube channels must match training order
        # SpectralCube often returns frequencies in descending order (CDELT<0),
        # while the training freq.txt is typically ascending.  Feeding reversed
        # channels to the network inverts the sign of inferred RM.
        # ------------------------------------------------------------------
        from ..simulator.physics import freq_to_lambda_sq as _f2lsq

        freq_sort_idx = None
        if self.model_lambda_sq:
            ref_lsq = next(iter(self.model_lambda_sq.values()))
            cube_lsq = _f2lsq(frequencies_hz)
            if len(ref_lsq) == len(cube_lsq):
                if np.allclose(cube_lsq[::-1], ref_lsq, rtol=1e-3):
                    freq_sort_idx = np.arange(n_freq)[::-1]
                    logger.info(
                        "Cube frequency channels are in reverse order relative to "
                        "training — reordering channels to match training grid."
                    )
                elif not np.allclose(cube_lsq, ref_lsq, rtol=1e-3):
                    logger.warning(
                        "Cube frequency grid does not match training grid in forward "
                        "or reverse order — inferred RM values may be unreliable."
                    )
        else:
            # No stored training grid — fall back to sorting cube frequencies
            # to ascending order, which matches the standard freq.txt convention.
            if frequencies_hz[0] > frequencies_hz[-1]:
                freq_sort_idx = np.arange(n_freq)[::-1]
                logger.info(
                    "No training frequency grid stored in checkpoint; cube frequencies "
                    "are descending — reordering to ascending to match training convention."
                )

        # ------------------------------------------------------------------
        # Auto chunk size from available RAM
        # Use 25% of available RAM — leaves room for intermediate arrays,
        # PyTorch model tensors, and OS overhead during inference.
        # Hard cap at 1024 to avoid excessive per-chunk I/O time.
        # ------------------------------------------------------------------
        available = psutil.virtual_memory().available
        n_cubes = 3 if i_cube is not None else 2  # Q + U [+ I]
        bytes_per_pixel = n_freq * 4 * n_cubes  # float32 per cube
        max_pixels = int(available * 0.25 / bytes_per_pixel)
        chunk_side = max(64, int(np.sqrt(max_pixels)))
        chunk_side = min(chunk_side, n_dec, n_ra)
        logger.info(
            f"Available RAM: {available / 2**30:.1f} GB  →  "
            f"chunk size: {chunk_side}×{chunk_side} "
            f"({chunk_side**2 * bytes_per_pixel / 2**20:.0f} MB per chunk)"
        )

        # ------------------------------------------------------------------
        # Initialise output maps (full spatial size, NaN)
        # ------------------------------------------------------------------
        shape2d = (n_dec, n_ra)

        def _nan2d():
            return np.full(shape2d, np.nan, dtype=np.float32)

        results: dict[str, np.ndarray] = {}
        model_keys = list(self.posteriors.keys())
        for key in model_keys:
            mt = key.rsplit("_n", 1)[0]
            nc = int(key.rsplit("_n", 1)[1])
            ppc = get_params_per_component(mt)
            for ci in range(1, nc + 1):
                tag = f"comp{ci}"
                for stat in ("mean", "std", "p16", "p84"):
                    results[f"rm_{stat}_{tag}"] = _nan2d()
                    results[f"amp_{stat}_{tag}"] = _nan2d()
                    results[f"chi0_{stat}_{tag}"] = _nan2d()
                    if ppc >= 4:
                        prefix = "delta_phi" if mt == "burn_slab" else "sigma_phi"
                        results[f"{prefix}_{stat}_{tag}"] = _nan2d()
        results["log_evidence"] = _nan2d()
        results["n_components"] = _nan2d()

        # ------------------------------------------------------------------
        # Pass 1a: per-channel noise → weights → good_chans
        # ------------------------------------------------------------------
        logger.info("Pass 1a: estimating per-channel noise...")
        noise_sum = np.zeros(n_freq, dtype=np.float64)
        noise_sum2 = np.zeros(n_freq, dtype=np.float64)
        n_pix_total = 0

        for y0 in tqdm(range(0, n_dec, chunk_side), desc="Pass 1a (rows)", unit="row"):
            y1 = min(y0 + chunk_side, n_dec)
            for x0 in range(0, n_ra, chunk_side):
                x1 = min(x0 + chunk_side, n_ra)
                q_chunk, _ = load_spatial_chunk(q_cube, u_cube, y0, y1, x0, x1)
                if freq_sort_idx is not None:
                    q_chunk = q_chunk[freq_sort_idx]
                finite_q = np.where(np.isfinite(q_chunk), q_chunk, 0.0)
                noise_sum += finite_q.sum(axis=(1, 2))
                noise_sum2 += (finite_q ** 2).sum(axis=(1, 2))
                n_pix_total += (y1 - y0) * (x1 - x0)

        mean_q = noise_sum / n_pix_total
        var_q = noise_sum2 / n_pix_total - mean_q ** 2
        noise_per_chan = np.sqrt(np.maximum(var_q, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(noise_per_chan > 0, 1.0 / noise_per_chan ** 2, 0.0)
        w_max = weights.max()
        if w_max > 0:
            weights /= w_max
        n_flagged = int((weights == 0).sum())
        good_chans = weights > 0
        logger.info(f"Weights: {n_flagged}/{n_freq} channels flagged (weight=0)")

        # ------------------------------------------------------------------
        # Pass 1b: collapsed P map using good channels only
        # ------------------------------------------------------------------
        logger.info("Pass 1b: building P map over good channels...")
        p_map = np.zeros(shape2d, dtype=np.float64)

        for y0 in tqdm(range(0, n_dec, chunk_side), desc="Pass 1b (rows)", unit="row"):
            y1 = min(y0 + chunk_side, n_dec)
            for x0 in range(0, n_ra, chunk_side):
                x1 = min(x0 + chunk_side, n_ra)
                q_chunk, u_chunk = load_spatial_chunk(q_cube, u_cube, y0, y1, x0, x1)
                if freq_sort_idx is not None:
                    q_chunk = q_chunk[freq_sort_idx]
                    u_chunk = u_chunk[freq_sort_idx]
                p_chunk = np.sqrt(q_chunk[good_chans] ** 2 + u_chunk[good_chans] ** 2)
                p_map[y0:y1, x0:x1] = np.nanmean(p_chunk, axis=0)

        # Noise in the 2D P map via MAD (robust against bright sources)
        p_median = np.nanmedian(p_map)
        sigma_p = 1.4826 * np.nanmedian(np.abs(p_map - p_median))
        p_threshold = snr_threshold * p_median
        logger.info(
            f"P map: median={p_median:.6f}, threshold ({snr_threshold}x median)={p_threshold:.6f}"
        )

        # ------------------------------------------------------------------
        # Dump P map and noise map for inspection
        # ------------------------------------------------------------------
        from astropy.io import fits as _fits
        _fits.writeto("p_map.fits", p_map.astype(np.float32), overwrite=True)
        _fits.writeto("noise_per_chan.fits", noise_per_chan.astype(np.float32), overwrite=True)
        _fits.writeto("weights.fits", weights.astype(np.float32), overwrite=True)
        logger.info("Dumped p_map.fits, noise_per_chan.fits, weights.fits")

        # ------------------------------------------------------------------
        # Pass 2: inference on active pixels
        # Re-read available RAM now that results arrays and P map are allocated
        # ------------------------------------------------------------------
        n_active = int((p_map >= p_threshold).sum())
        logger.info(f"Pass 2: inference on {n_active}/{n_dec*n_ra} active pixels...")

        available_p2 = psutil.virtual_memory().available
        max_pixels_p2 = int(available_p2 * 0.25 / bytes_per_pixel)
        chunk_side = max(64, int(np.sqrt(max_pixels_p2)))
        chunk_side = min(chunk_side, n_dec, n_ra)
        logger.info(
            f"Pass 2 available RAM: {available_p2 / 2**30:.1f} GB  →  "
            f"chunk size: {chunk_side}×{chunk_side} "
            f"({chunk_side**2 * bytes_per_pixel / 2**20:.0f} MB per chunk)"
        )

        self._check_frequency_compatibility(frequencies_hz)

        for y0 in tqdm(range(0, n_dec, chunk_side), desc="Pass 2 (rows)", unit="row"):
            y1 = min(y0 + chunk_side, n_dec)
            for x0 in range(0, n_ra, chunk_side):
                x1 = min(x0 + chunk_side, n_ra)

                # Skip chunks with no active pixels
                chunk_p = p_map[y0:y1, x0:x1]
                if not np.any(chunk_p >= p_threshold):
                    continue

                q_chunk, u_chunk = load_spatial_chunk(q_cube, u_cube, y0, y1, x0, x1)
                if freq_sort_idx is not None:
                    q_chunk = q_chunk[freq_sort_idx]
                    u_chunk = u_chunk[freq_sort_idx]

                if i_cube is not None:
                    from ..io import load_i_chunk, normalize_qu_by_i
                    i_chunk = load_i_chunk(i_cube, y0, y1, x0, x1)
                    if freq_sort_idx is not None:
                        i_chunk = i_chunk[freq_sort_idx]
                    q_chunk, u_chunk = normalize_qu_by_i(q_chunk, u_chunk, i_chunk)

                active_local = np.argwhere(chunk_p >= p_threshold)
                for dec_local, ra_local in active_local:
                    dec_idx = y0 + dec_local
                    ra_idx = x0 + ra_local

                    Q_pix = q_chunk[:, dec_local, ra_local].copy()
                    U_pix = u_chunk[:, dec_local, ra_local].copy()

                    bad = ~good_chans | ~np.isfinite(Q_pix) | ~np.isfinite(U_pix)
                    Q_pix[bad] = 0.0
                    U_pix[bad] = 0.0

                    qu_obs = np.concatenate([Q_pix, U_pix])

                    try:
                        best_result, _ = self.infer(
                            qu_obs, weights=weights, n_samples=n_samples
                        )
                    except Exception as exc:
                        logger.warning(f"Pixel ({dec_idx}, {ra_idx}) failed: {exc}")
                        continue

                    results["log_evidence"][dec_idx, ra_idx] = best_result.log_evidence
                    results["n_components"][dec_idx, ra_idx] = best_result.n_components

                    for comp_i, comp in enumerate(best_result.components):
                        tag = f"comp{comp_i + 1}"

                        rm_samp = comp.samples[:, 0]
                        results[f"rm_mean_{tag}"][dec_idx, ra_idx] = np.mean(rm_samp)
                        results[f"rm_std_{tag}"][dec_idx, ra_idx] = np.std(rm_samp)
                        results[f"rm_p16_{tag}"][dec_idx, ra_idx] = np.percentile(rm_samp, 16)
                        results[f"rm_p84_{tag}"][dec_idx, ra_idx] = np.percentile(rm_samp, 84)

                        amp_col = comp.samples.shape[1] - 2
                        amp_samp = comp.samples[:, amp_col]
                        results[f"amp_mean_{tag}"][dec_idx, ra_idx] = np.mean(amp_samp)
                        results[f"amp_std_{tag}"][dec_idx, ra_idx] = np.std(amp_samp)
                        results[f"amp_p16_{tag}"][dec_idx, ra_idx] = np.percentile(amp_samp, 16)
                        results[f"amp_p84_{tag}"][dec_idx, ra_idx] = np.percentile(amp_samp, 84)

                        chi0_samp = comp.samples[:, -1]
                        results[f"chi0_mean_{tag}"][dec_idx, ra_idx] = np.mean(chi0_samp)
                        results[f"chi0_std_{tag}"][dec_idx, ra_idx] = np.std(chi0_samp)
                        results[f"chi0_p16_{tag}"][dec_idx, ra_idx] = np.percentile(chi0_samp, 16)
                        results[f"chi0_p84_{tag}"][dec_idx, ra_idx] = np.percentile(chi0_samp, 84)

                        if comp.samples.shape[1] >= 4:
                            sec_samp = comp.samples[:, 1]
                            prefix = (
                                "delta_phi"
                                if best_result.model_type == "burn_slab"
                                else "sigma_phi"
                            )
                            results[f"{prefix}_mean_{tag}"][dec_idx, ra_idx] = np.mean(sec_samp)
                            results[f"{prefix}_std_{tag}"][dec_idx, ra_idx] = np.std(sec_samp)
                            results[f"{prefix}_p16_{tag}"][dec_idx, ra_idx] = np.percentile(sec_samp, 16)
                            results[f"{prefix}_p84_{tag}"][dec_idx, ra_idx] = np.percentile(sec_samp, 84)

        logger.info("Chunked cube inference complete.")
        return results

    # ------------------------------------------------------------------
    # Spectral shape inference
    # ------------------------------------------------------------------

    def load_spectral_shape_model(self, model_path: str | Path) -> None:
        """
        Load a trained spectral shape posterior.

        The posterior is stored under the key ``"spectral_shape"`` in
        ``self.posteriors``.

        Parameters
        ----------
        model_path : str or Path
            Path to the ``.pt`` file produced by :class:`SpectralShapeTrainer`.
        """
        posterior, metadata = load_posterior(Path(model_path), self.device)
        self.posteriors["spectral_shape"] = posterior
        self.posterior_metadata["spectral_shape"] = metadata
        logger.info(f"Loaded spectral shape model from {model_path}")

    def infer_spectra(
        self,
        i_obs: np.ndarray,
        weights: np.ndarray | None = None,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """
        Run spectral shape inference on a single total-intensity spectrum.

        Parameters
        ----------
        i_obs : np.ndarray, shape (n_freq,)
            Observed total intensity spectrum.
        weights : np.ndarray, optional
            Channel weights of shape (n_freq,).  Zero-weight channels are
            zeroed out before feeding to the network (matching training).
        n_samples : int
            Number of posterior samples to draw.

        Returns
        -------
        samples : np.ndarray, shape (n_samples, 4)
            Posterior samples: [log_F0, alpha, beta, gamma].
        """
        if "spectral_shape" not in self.posteriors:
            raise ValueError(
                "No spectral shape model loaded. Call load_spectral_shape_model() first."
            )

        obs = i_obs.copy().astype(np.float32)
        if weights is not None:
            obs[weights == 0] = 0.0
        else:
            obs[~np.isfinite(obs)] = 0.0

        # Mean-normalize matching training: divide by mean over good channels
        good = obs != 0.0
        mean_I = float(obs[good].mean()) if good.any() else 1.0
        obs = obs / mean_I

        x_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        posterior = self.posteriors["spectral_shape"]
        samples = posterior.sample((n_samples,), x=x_t)
        # Return mean_I as second element so caller can reconstruct absolute I_model:
        #   I_model(ν) = exp(log_F0 + α·x + β·x² + γ·x³) × mean_I
        return samples.detach().cpu().numpy(), mean_I

    def run_spectral_shape_cube(
        self,
        i_data: np.ndarray,
        weights: np.ndarray | None = None,
        n_samples: int = 1000,
        mask: np.ndarray | None = None,
        snr_threshold: float | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Run spectral shape inference over all valid spatial pixels of a
        total-intensity cube.

        Parameters
        ----------
        i_data : np.ndarray, shape (n_freq, n_dec, n_ra)
            Total intensity cube.
        weights : np.ndarray, optional
            Inverse-variance channel weights.
            - ``(n_freq,)``             → same weights for every pixel.
            - ``(n_freq, n_dec, n_ra)`` → per-pixel weights.
        n_samples : int
            Number of posterior samples per pixel.
        mask : np.ndarray, optional
            Boolean 2D array shape ``(n_dec, n_ra)``.  Only pixels where
            ``mask == True`` are processed.
        snr_threshold : float, optional
            If given, only process pixels whose mean intensity exceeds
            ``snr_threshold * sigma_I`` where ``sigma_I`` is estimated
            via MAD over the collapsed intensity map.

        Returns
        -------
        results : dict
            ``{name: 2D np.ndarray}`` of shape ``(n_dec, n_ra)``.
            Unprocessed pixels are NaN.  Keys:
            ``log_F0_mean``, ``log_F0_std``, ``log_F0_p16``, ``log_F0_p84``,
            ``alpha_*``, ``beta_*``, ``gamma_*``.
        """
        from tqdm import tqdm

        if "spectral_shape" not in self.posteriors:
            raise ValueError(
                "No spectral shape model loaded. Call load_spectral_shape_model() first."
            )

        n_freq, n_dec, n_ra = i_data.shape
        shape2d = (n_dec, n_ra)

        # Pre-allocate output arrays
        param_names = ["log_F0", "alpha", "beta", "gamma"]
        results: dict[str, np.ndarray] = {}
        for name in param_names:
            for stat in ("mean", "std", "p16", "p84"):
                results[f"{name}_{stat}"] = np.full(shape2d, np.nan, dtype=np.float32)

        # Build pixel mask
        if weights is not None:
            w = np.asarray(weights)
            good_chans = (w > 0) if w.ndim == 1 else np.any(w > 0, axis=(1, 2))
            i_check = i_data[good_chans]
        else:
            i_check = i_data
        valid = ~np.any(np.isnan(i_check), axis=0)

        if mask is not None:
            if mask.shape != shape2d:
                raise ValueError(
                    f"mask shape {mask.shape} does not match spatial dimensions {shape2d}"
                )
            valid = valid & mask.astype(bool)

        # SNR threshold based on collapsed intensity map
        good_chans_idx = (
            (weights > 0) if (weights is not None and weights.ndim == 1) else slice(None)
        )
        i_map = np.nanmean(i_data[good_chans_idx], axis=0)
        i_median = np.nanmedian(i_map)
        sigma_i = 1.4826 * np.nanmedian(np.abs(i_map - i_median))
        logger.info(
            f"I map: min={np.nanmin(i_map):.4f}, max={np.nanmax(i_map):.4f}, "
            f"noise (MAD)={sigma_i:.4f}"
        )

        if snr_threshold is not None:
            valid = valid & (i_map >= snr_threshold * sigma_i)
        else:
            valid = valid & (i_map > sigma_i)

        pixel_list = np.argwhere(valid)
        n_valid = len(pixel_list)
        n_total = n_dec * n_ra
        logger.info(
            f"Spectral cube inference: fitting {n_valid}/{n_total} pixels "
            f"({100 * n_valid / n_total:.1f}%)"
        )

        for dec_idx, ra_idx in tqdm(pixel_list, desc="Spectra cube infer", unit="px"):
            i_pix = i_data[:, dec_idx, ra_idx].copy()

            if weights is not None and np.asarray(weights).ndim == 3:
                pix_weights = weights[:, dec_idx, ra_idx]
            else:
                pix_weights = weights

            if pix_weights is not None:
                i_pix[pix_weights == 0] = 0.0
            else:
                i_pix[~np.isfinite(i_pix)] = 0.0

            try:
                samples = self.infer_spectra(i_pix, weights=pix_weights, n_samples=n_samples)
            except Exception as exc:
                logger.warning(f"Pixel ({dec_idx}, {ra_idx}) failed: {exc}")
                continue

            # samples: (n_samples, 4) — [log_F0, alpha, beta, gamma]
            for pi, name in enumerate(param_names):
                col = samples[:, pi]
                results[f"{name}_mean"][dec_idx, ra_idx] = np.mean(col)
                results[f"{name}_std"][dec_idx, ra_idx] = np.std(col)
                results[f"{name}_p16"][dec_idx, ra_idx] = np.percentile(col, 16)
                results[f"{name}_p84"][dec_idx, ra_idx] = np.percentile(col, 84)

        logger.info("Spectral cube inference complete.")
        return results

    def run_spectral_shape_cube_chunked(
        self,
        i_cube,
        shape: tuple[int, int, int],
        frequencies_hz: np.ndarray,
        snr_threshold: float = 5.0,
        n_samples: int = 1000,
    ) -> dict[str, np.ndarray]:
        """
        Chunked spectral shape cube inference — never loads the full cube into RAM.

        Parameters
        ----------
        i_cube : SpectralCube (lazy)
        shape : (n_freq, n_dec, n_ra)
        frequencies_hz : np.ndarray
        snr_threshold : float
        n_samples : int
        """
        import psutil
        from tqdm import tqdm

        from ..io import load_i_chunk

        if "spectral_shape" not in self.posteriors:
            raise ValueError(
                "No spectral shape model loaded. Call load_spectral_shape_model() first."
            )

        n_freq, n_dec, n_ra = shape
        shape2d = (n_dec, n_ra)

        # Auto chunk size from available RAM
        available = psutil.virtual_memory().available
        bytes_per_pixel = n_freq * 4  # float32 for I cube
        max_pixels = int(available * 0.25 / bytes_per_pixel)
        chunk_side = max(64, int(np.sqrt(max_pixels)))
        chunk_side = min(chunk_side, n_dec, n_ra)
        logger.info(
            f"Available RAM: {available / 2**30:.1f} GB  →  "
            f"chunk size: {chunk_side}×{chunk_side}"
        )

        # Pre-allocate output arrays
        param_names = ["log_F0", "alpha", "beta", "gamma"]
        results: dict[str, np.ndarray] = {}
        for name in param_names:
            for stat in ("mean", "std", "p16", "p84"):
                results[f"{name}_{stat}"] = np.full(shape2d, np.nan, dtype=np.float32)

        # Pass 1: per-channel noise → weights → good_chans
        logger.info("Pass 1: estimating per-channel noise from I cube...")
        noise_sum = np.zeros(n_freq, dtype=np.float64)
        noise_sum2 = np.zeros(n_freq, dtype=np.float64)
        n_pix_total = 0

        for y0 in tqdm(range(0, n_dec, chunk_side), desc="Pass 1 (rows)", unit="row"):
            y1 = min(y0 + chunk_side, n_dec)
            for x0 in range(0, n_ra, chunk_side):
                x1 = min(x0 + chunk_side, n_ra)
                i_chunk = load_i_chunk(i_cube, y0, y1, x0, x1)
                finite_i = np.where(np.isfinite(i_chunk), i_chunk, 0.0)
                noise_sum += finite_i.sum(axis=(1, 2))
                noise_sum2 += (finite_i**2).sum(axis=(1, 2))
                n_pix_total += (y1 - y0) * (x1 - x0)

        mean_i = noise_sum / n_pix_total
        var_i = noise_sum2 / n_pix_total - mean_i**2
        noise_per_chan = np.sqrt(np.maximum(var_i, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(noise_per_chan > 0, 1.0 / noise_per_chan**2, 0.0)
        w_max = weights.max()
        if w_max > 0:
            weights /= w_max
        good_chans = weights > 0
        logger.info(f"Weights: {int((~good_chans).sum())}/{n_freq} channels flagged")

        # Pass 2: collapsed I map for masking
        logger.info("Pass 2: building I map for SNR masking...")
        i_map = np.zeros(shape2d, dtype=np.float64)
        i_count = np.zeros(shape2d, dtype=np.int32)

        for y0 in tqdm(range(0, n_dec, chunk_side), desc="Pass 2 (rows)", unit="row"):
            y1 = min(y0 + chunk_side, n_dec)
            for x0 in range(0, n_ra, chunk_side):
                x1 = min(x0 + chunk_side, n_ra)
                i_chunk = load_i_chunk(i_cube, y0, y1, x0, x1)
                finite = np.isfinite(i_chunk[good_chans])
                i_map[y0:y1, x0:x1] += np.nanmean(i_chunk[good_chans], axis=0)
                i_count[y0:y1, x0:x1] += 1

        with np.errstate(invalid="ignore"):
            i_map = np.where(i_count > 0, i_map / i_count, np.nan)

        i_median = np.nanmedian(i_map)
        sigma_i = 1.4826 * np.nanmedian(np.abs(i_map - i_median))
        logger.info(f"I map noise (MAD): {sigma_i:.4f}")

        valid = np.isfinite(i_map) & (i_map >= snr_threshold * sigma_i)
        pixel_list = np.argwhere(valid)
        n_valid = len(pixel_list)
        logger.info(
            f"Spectral cube inference: fitting {n_valid}/{n_dec * n_ra} pixels "
            f"({100 * n_valid / (n_dec * n_ra):.1f}%)"
        )

        # Pass 3: per-pixel inference
        for y0 in tqdm(range(0, n_dec, chunk_side), desc="Pass 3 (inference)", unit="row"):
            y1 = min(y0 + chunk_side, n_dec)
            for x0 in range(0, n_ra, chunk_side):
                x1 = min(x0 + chunk_side, n_ra)
                i_chunk = load_i_chunk(i_cube, y0, y1, x0, x1)

                for dy in range(y1 - y0):
                    for dx in range(x1 - x0):
                        dec_idx = y0 + dy
                        ra_idx = x0 + dx
                        if not valid[dec_idx, ra_idx]:
                            continue

                        i_pix = i_chunk[:, dy, dx].copy().astype(np.float32)
                        i_pix[~good_chans] = 0.0
                        i_pix[~np.isfinite(i_pix)] = 0.0

                        try:
                            samples = self.infer_spectra(
                                i_pix, weights=weights, n_samples=n_samples
                            )
                        except Exception as exc:
                            logger.warning(f"Pixel ({dec_idx}, {ra_idx}) failed: {exc}")
                            continue

                        for pi, name in enumerate(param_names):
                            col = samples[:, pi]
                            results[f"{name}_mean"][dec_idx, ra_idx] = np.mean(col)
                            results[f"{name}_std"][dec_idx, ra_idx] = np.std(col)
                            results[f"{name}_p16"][dec_idx, ra_idx] = np.percentile(col, 16)
                            results[f"{name}_p84"][dec_idx, ra_idx] = np.percentile(col, 84)

        logger.info("Chunked spectral cube inference complete.")
        return results
