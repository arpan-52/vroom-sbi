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
        qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=self.device)

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

            logger.info(f"Running inference for {key}...")

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

    def run_inference_cube(
        self,
        q_data: np.ndarray,
        u_data: np.ndarray,
        weights: np.ndarray | None = None,
        n_samples: int = 1000,
        batch_size: int = 1,
        mask: np.ndarray | None = None,
        snr_threshold: float | None = None,
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

        _, n_dec, n_ra = q_data.shape

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
        # Layer 1: auto NaN mask
        valid = ~np.any(np.isnan(q_data) | np.isnan(u_data), axis=0)

        # Layer 2: user spatial mask
        if mask is not None:
            if mask.shape != shape2d:
                raise ValueError(
                    f"mask shape {mask.shape} does not match spatial dimensions {shape2d}"
                )
            valid = valid & mask.astype(bool)

        # Layer 3: SNR threshold
        if snr_threshold is not None:
            p_mean = np.nanmean(np.sqrt(q_data**2 + u_data**2), axis=0)
            noise_est = np.nanstd(q_data, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                snr_map = np.where(noise_est > 0, p_mean / noise_est, 0.0)
            valid = valid & (snr_map >= snr_threshold)

        pixel_list = np.argwhere(valid)  # shape (N_valid, 2)
        n_valid = len(pixel_list)
        n_total = n_dec * n_ra
        logger.info(
            f"Cube inference: fitting {n_valid}/{n_total} pixels "
            f"({100 * n_valid / n_total:.1f}%)"
        )

        # ------------------------------------------------------------------
        # Pixel loop
        # ------------------------------------------------------------------
        for chunk_start in range(0, n_valid, max(1, batch_size)):
            chunk = pixel_list[chunk_start : chunk_start + max(1, batch_size)]
            for dec_idx, ra_idx in tqdm(
                chunk,
                desc=f"Pixels {chunk_start}–{chunk_start + len(chunk) - 1}",
                leave=False,
            ):
                Q_pix = q_data[:, dec_idx, ra_idx]
                U_pix = u_data[:, dec_idx, ra_idx]
                qu_obs = np.concatenate([Q_pix, U_pix])

                # Per-pixel weights
                if weights is not None and weights.ndim == 3:
                    pix_weights = weights[:, dec_idx, ra_idx]
                else:
                    pix_weights = weights

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

                    # RM
                    rm_samp = comp.samples[:, 0]
                    results[f"rm_mean_{tag}"][dec_idx, ra_idx] = np.mean(rm_samp)
                    results[f"rm_std_{tag}"][dec_idx, ra_idx] = np.std(rm_samp)
                    results[f"rm_p16_{tag}"][dec_idx, ra_idx] = np.percentile(
                        rm_samp, 16
                    )
                    results[f"rm_p84_{tag}"][dec_idx, ra_idx] = np.percentile(
                        rm_samp, 84
                    )

                    # Amplitude (second-to-last column in samples)
                    amp_col = comp.samples.shape[1] - 2
                    amp_samp = comp.samples[:, amp_col]
                    results[f"amp_mean_{tag}"][dec_idx, ra_idx] = np.mean(amp_samp)
                    results[f"amp_std_{tag}"][dec_idx, ra_idx] = np.std(amp_samp)
                    results[f"amp_p16_{tag}"][dec_idx, ra_idx] = np.percentile(
                        amp_samp, 16
                    )
                    results[f"amp_p84_{tag}"][dec_idx, ra_idx] = np.percentile(
                        amp_samp, 84
                    )

                    # chi0 (last column)
                    chi0_samp = comp.samples[:, -1]
                    results[f"chi0_mean_{tag}"][dec_idx, ra_idx] = np.mean(chi0_samp)
                    results[f"chi0_std_{tag}"][dec_idx, ra_idx] = np.std(chi0_samp)
                    results[f"chi0_p16_{tag}"][dec_idx, ra_idx] = np.percentile(
                        chi0_samp, 16
                    )
                    results[f"chi0_p84_{tag}"][dec_idx, ra_idx] = np.percentile(
                        chi0_samp, 84
                    )

                    # Extended-model second parameter (column 1 when ndim >= 4)
                    if comp.samples.shape[1] >= 4:
                        sec_samp = comp.samples[:, 1]
                        if comp.sigma_phi_mean is not None:
                            prefix = "sigma_phi"
                        elif comp.delta_phi_mean is not None:
                            prefix = "delta_phi"
                        else:
                            prefix = "sigma_phi"  # safe fallback
                        results[f"{prefix}_mean_{tag}"][dec_idx, ra_idx] = np.mean(
                            sec_samp
                        )
                        results[f"{prefix}_std_{tag}"][dec_idx, ra_idx] = np.std(
                            sec_samp
                        )
                        results[f"{prefix}_p16_{tag}"][dec_idx, ra_idx] = np.percentile(
                            sec_samp, 16
                        )
                        results[f"{prefix}_p84_{tag}"][dec_idx, ra_idx] = np.percentile(
                            sec_samp, 84
                        )

        logger.info("Cube inference complete.")
        return results
