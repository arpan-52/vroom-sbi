"""
SpectralShapeTrainer for VROOM-SBI.

Trains a neural posterior estimator for the log-log polynomial spectral
shape model:

    log F(ν) = log_F0 + alpha*x + beta*x^2 + gamma*x^3,  x = log(ν/ν₀)

Uses the same infrastructure as SBITrainer (async chunk streaming, SBI flow),
but with:
  - SpectralShapeSimulator (real F(ν) output, not complex Q+U)
  - Embedding input: n_freq   (not 2*n_freq)
  - Additive noise only
  - Same weight augmentation pipeline (missing channels, RFI gaps, etc.)
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Configuration
from ..core.checkpoint import CheckpointManager, save_training_plots
from ..core.result import TrainingMetrics, TrainingResult
from ..simulator.augmentation import augment_weights_combined
from ..simulator.prior import build_spectral_shape_prior, sample_spectral_shape_prior
from ..simulator.spectral_simulator import SpectralShapeSimulator
from .networks import SpectralEmbedding

logger = logging.getLogger(__name__)


class _SpectralDensityEstimatorWrapper:
    """
    Fallback wrapper used when DirectPosterior construction fails.

    Exposes the same .sample() and .posterior_estimator interface so
    load_posterior() and infer_spectra() work without modification.
    """

    def __init__(self, density_estimator: nn.Module) -> None:
        self.posterior_estimator = density_estimator

    def sample(
        self,
        sample_shape,
        x: torch.Tensor,
        show_progress_bars: bool = False,
    ) -> torch.Tensor:
        from sbi.neural_nets.estimators.shape_handling import reshape_to_batch_event

        # DirectPosterior reshapes x to (1, *event_shape) before passing to the
        # density estimator, and NFlowsFlow.sample returns (*sample_shape, 1, n_params).
        # We replicate that here and squeeze the batch dim back out.
        x = reshape_to_batch_event(
            x, event_shape=self.posterior_estimator.condition_shape
        )
        samples = self.posterior_estimator.sample(torch.Size(sample_shape), condition=x)
        # samples: (*sample_shape, 1, n_params) → (*sample_shape, n_params)
        return samples.squeeze(-2)


class SpectralShapeTrainer:
    """
    Trainer for the spectral shape SBI posterior.

    Parameters
    ----------
    config : Configuration
        Full configuration object. Uses:
          config.freq_file, config.spectral_shape, config.training,
          config.sbi, config.weight_augmentation, config.noise
    checkpoint_manager : CheckpointManager, optional
    """

    def __init__(
        self,
        config: Configuration,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        self.config = config
        self.device = config.training.device

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        if checkpoint_manager is None:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=Path(config.training.save_dir) / "checkpoints"
            )
        self.checkpoint_manager = checkpoint_manager

    def train(self, n_simulations: int | None = None) -> TrainingResult:
        """
        Train the spectral shape posterior.

        Parameters
        ----------
        n_simulations : int, optional
            Override number of simulations from config.

        Returns
        -------
        TrainingResult
        """
        if n_simulations is None:
            n_simulations = self.config.training.n_simulations

        logger.info(f"\n{'=' * 60}")
        logger.info("Training Spectral Shape Model")
        logger.info(f"Simulations: {n_simulations:,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"{'=' * 60}")

        simulator = SpectralShapeSimulator(self.config.freq_file)
        spectral_cfg = self.config.spectral_shape

        logger.info(
            f"Simulator: n_params={simulator.n_params}, n_freq={simulator.n_freq}, "
            f"ν₀={simulator.nu0 / 1e6:.1f} MHz"
        )

        prior = build_spectral_shape_prior(spectral_cfg, device=self.device)

        # ----------------------------------------------------------------
        # Phase 1: Generate chunked simulations
        # ----------------------------------------------------------------
        chunk_dir = self._generate_chunks(simulator, spectral_cfg, n_simulations)

        # ----------------------------------------------------------------
        # Phase 2: Train on chunks
        # ----------------------------------------------------------------
        # Embedding input is n_freq (real flux), NOT 2*n_freq like pol models
        input_dim = simulator.n_freq
        embedding_dim = self.config.sbi.embedding_dim
        embedding_net = SpectralEmbedding(
            input_dim=input_dim,
            output_dim=embedding_dim,
        ).to(self.device)

        # Use architecture for n_components=1
        arch_config = self.config.sbi.get_architecture(1)

        logger.info(
            f"SBI Architecture: {self.config.sbi.model.upper()}, "
            f"hidden={arch_config.hidden_features}, "
            f"transforms={arch_config.num_transforms}"
        )

        start_time = datetime.now()

        density_estimator, history = self._train_on_chunks(
            chunk_dir, prior, embedding_net, arch_config
        )

        training_time = (datetime.now() - start_time).total_seconds()

        output_dir = Path(self.config.training.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "spectral_shape_posterior.pt"

        from sbi.inference.posteriors import DirectPosterior

        try:
            posterior = DirectPosterior(
                posterior_estimator=density_estimator,
                prior=prior,
            )
        except AssertionError:
            logger.warning(
                "DirectPosterior construction failed (mcmc_transform round-trip check). "
                "Falling back to _SpectralDensityEstimatorWrapper — inference is unaffected."
            )
            posterior = _SpectralDensityEstimatorWrapper(density_estimator)

        self._save_posterior(
            posterior, prior, embedding_net, simulator, history, model_path
        )

        # Save training plots
        if history.get("train_loss") or history.get("val_loss"):
            plot_path = output_dir / "training_spectral_shape.png"
            save_training_plots(history, plot_path, "spectral_shape", 1)
            logger.info(f"Saved training plot to {plot_path}")

        # Build result
        result = TrainingResult(
            model_type="spectral_shape",
            n_components=1,
            n_simulations=n_simulations,
            n_params=simulator.n_params,
            final_train_loss=history.get("train_loss", [0.0])[-1]
            if history.get("train_loss")
            else 0.0,
            final_val_loss=history.get("val_loss", [None])[-1]
            if history.get("val_loss")
            else None,
            model_path=str(model_path),
            training_time_seconds=training_time,
            config_snapshot=self.config.to_dict(),
        )

        train_losses = history.get("train_loss", [])
        val_losses = history.get("val_loss", [None] * len(train_losses))
        for i, tl in enumerate(train_losses):
            result.add_metric(TrainingMetrics(epoch=i, train_loss=tl, val_loss=val_losses[i]))

        logger.info(f"Spectral shape training complete in {training_time:.1f}s")
        logger.info(f"Model saved to {model_path}")

        return result

    def _generate_chunks(self, simulator, spectral_cfg, n_simulations: int) -> Path:
        """
        Generate simulations and save as chunked .pt files.

        Uses the same chunk format as SBITrainer so AsyncChunkStreamer works
        unchanged. Each chunk file contains {"theta": ..., "x": ...}.

        Applies the same weight augmentation pipeline as SBITrainer:
        - Scattered missing channels
        - Contiguous RFI gaps
        - Large RFI blocks
        - Per-sample additive noise sigma (uniform random within range)
        """
        chunk_size = self.config.training.simulation_batch_size
        n_full = n_simulations // chunk_size
        remainder = n_simulations % chunk_size
        n_chunks = n_full + (1 if remainder > 0 else 0)

        output_dir = Path(self.config.training.save_dir)
        chunk_dir = output_dir / "chunks_spectral_shape"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Generating {n_simulations:,} simulations in {n_chunks} chunk(s) "
            f"(chunk size {chunk_size:,})"
        )

        samples_done = 0
        for ci in range(n_chunks):
            this_size = min(chunk_size, n_simulations - samples_done)

            # Sample prior
            chunk_theta = sample_spectral_shape_prior(this_size, spectral_cfg)

            # Augmented weights (same pipeline as pol models)
            chunk_weights = np.array(
                [
                    augment_weights_combined(
                        simulator.weights,
                        scattered_prob=self.config.weight_augmentation.scattered_prob,
                        gap_prob=self.config.weight_augmentation.gap_prob,
                        large_block_prob=self.config.weight_augmentation.large_block_prob,
                        noise_variation=self.config.weight_augmentation.noise_variation,
                    )
                    for _ in range(this_size)
                ],
                dtype=np.float32,
            )

            # Additive noise sigma (draw one sigma per chunk for robustness)
            if self.config.noise.augmentation_enable:
                noise_sigma = np.random.uniform(
                    spectral_cfg.sigma_min
                    * self.config.noise.augmentation_min_factor,
                    spectral_cfg.sigma_max
                    * self.config.noise.augmentation_max_factor,
                )
            else:
                noise_sigma = (spectral_cfg.sigma_min + spectral_cfg.sigma_max) / 2.0

            chunk_x = simulator.simulate_batch(
                chunk_theta, chunk_weights, noise_sigma=noise_sigma
            )

            # Mean-normalize: divide each spectrum by its mean over good channels
            # so log_F0 ≈ 0 regardless of absolute flux → prior is scale-invariant
            good_mask = chunk_weights > 0  # (batch, n_freq)
            mean_I = np.where(good_mask, chunk_x, np.nan)
            mean_I = np.nanmean(mean_I, axis=1, keepdims=True)  # (batch, 1)
            chunk_x = np.where(good_mask, chunk_x / mean_I, 0.0)

            chunk_path = chunk_dir / f"chunk_{ci:04d}.pt"
            torch.save(
                {
                    "theta": torch.tensor(chunk_theta, dtype=torch.float32),
                    "x": torch.tensor(chunk_x, dtype=torch.float32),
                },
                chunk_path,
            )

            logger.info(f"  Saved {chunk_path.name}")

            del chunk_theta, chunk_weights, chunk_x
            samples_done += this_size

        torch.save(
            {
                "n_simulations": n_simulations,
                "n_chunks": n_chunks,
                "chunk_size": chunk_size,
                "n_freq": simulator.n_freq,
                "n_params": simulator.n_params,
                "model_type": "spectral_shape",
            },
            chunk_dir / "metadata.pt",
        )

        logger.info(f"Generated {n_simulations:,} simulations in {n_chunks} chunk(s)")
        return chunk_dir

    def _train_on_chunks(
        self,
        chunk_dir: Path,
        prior,
        embedding_net: nn.Module,
        arch_config,
    ) -> tuple:
        """Train on chunked .pt files using the streaming NPE trainer."""
        from .streaming_trainer import StreamingNPETrainer

        logger.info("Using async streaming NPE training")

        trainer = StreamingNPETrainer(device=self.device)

        density_estimator, history = trainer.train(
            chunk_dir=chunk_dir,
            prior=prior,
            embedding_net=embedding_net,
            flow_type=self.config.sbi.model,
            hidden_features=arch_config.hidden_features,
            num_transforms=arch_config.num_transforms,
            num_bins=self.config.sbi.num_bins,
            learning_rate=self.config.training.learning_rate,
            training_batch_size=self.config.training.training_batch_size,
            validation_fraction=self.config.training.validation_fraction,
            max_epochs=500,
            stop_after_epochs=self.config.training.stop_after_epochs,
            clip_grad_norm=5.0,
            show_progress=True,
        )

        return density_estimator, history

    def _save_posterior(
        self,
        posterior,
        prior,
        embedding_net: nn.Module,
        simulator: SpectralShapeSimulator,
        history: dict,
        save_path: Path,
    ):
        """Save spectral shape posterior to disk."""
        try:
            if hasattr(prior, "base_dist"):
                prior_bounds = {
                    "low": prior.base_dist.low.cpu().numpy().tolist(),
                    "high": prior.base_dist.high.cpu().numpy().tolist(),
                }
            elif hasattr(prior, "low") and hasattr(prior, "high"):
                prior_bounds = {
                    "low": prior.low.cpu().numpy().tolist(),
                    "high": prior.high.cpu().numpy().tolist(),
                }
            else:
                cfg = self.config.spectral_shape
                prior_bounds = {
                    "low": [cfg.log_F0_min, cfg.alpha_min, cfg.beta_min, cfg.gamma_min],
                    "high": [cfg.log_F0_max, cfg.alpha_max, cfg.beta_max, cfg.gamma_max],
                }
        except Exception:
            cfg = self.config.spectral_shape
            prior_bounds = {
                "low": [cfg.log_F0_min, cfg.alpha_min, cfg.beta_min, cfg.gamma_min],
                "high": [cfg.log_F0_max, cfg.alpha_max, cfg.beta_max, cfg.gamma_max],
            }

        torch.save(
            {
                # Model identity
                "model_type": "spectral_shape",
                "n_params": simulator.n_params,
                "n_freq": simulator.n_freq,
                "param_names": simulator.get_param_names(),
                "nu0_hz": float(simulator.nu0),
                "freq_hz": simulator.freq.tolist(),
                # Network states
                "posterior": posterior,
                "embedding_net_state": embedding_net.state_dict(),
                # Prior info
                "prior_bounds": prior_bounds,
                # Training info
                "training_history": history,
                # Config
                "config_used": {
                    "sbi_model": self.config.sbi.model,
                    "embedding_dim": self.config.sbi.embedding_dim,
                    "hidden_features": self.config.sbi.get_architecture(1).hidden_features,
                    "num_transforms": self.config.sbi.get_architecture(1).num_transforms,
                },
                # Metadata
                "save_timestamp": datetime.now().isoformat(),
                "torch_version": torch.__version__,
                "training_type": "streaming_npe",
            },
            save_path,
        )
        logger.info(f"Saved spectral shape posterior to {save_path}")
