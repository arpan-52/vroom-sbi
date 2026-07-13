"""
On-the-fly NPE trainer: generates each mini-batch on the GPU and trains
immediately, keeping the GPU hot with no disk round-trip.

Mirrors the interface and training mechanics of ``StreamingNPETrainer``
(density-estimator build, Adam + ReduceLROnPlateau, gradient clipping,
early stopping, best-state restore) but sources data from
``simulator.gpu_simulator.GPUSimulator`` instead of chunk files on disk.

"Epoch" is defined as ``steps_per_epoch`` freshly-generated batches. The
validation set is generated once with a fixed seed so the validation loss is
comparable across epochs. SBC remains the arbiter of posterior calibration.
"""

import logging
from copy import deepcopy

import torch
import torch.nn as nn
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.net_builders import build_maf, build_nsf
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OnlineNPETrainer:
    """Train NPE with GPU on-the-fly simulation (no disk)."""

    def __init__(self, simulator, device: str = "cuda"):
        self.simulator = simulator
        self.device = device if torch.cuda.is_available() else "cpu"
        self.density_estimator = None
        self.prior = None
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_median": [],
            "val_p99": [],
            "val_max": [],
            "learning_rates": [],
        }

    def build_density_estimator(
        self,
        theta_sample: torch.Tensor,
        x_sample: torch.Tensor,
        embedding_net: nn.Module | None = None,
        flow_type: str = "nsf",
        hidden_features: int = 256,
        num_transforms: int = 15,
        num_bins: int = 16,
    ) -> nn.Module:
        """Build the normalizing flow. SBI standardizes from the sample data."""
        build_kwargs = {
            "hidden_features": hidden_features,
            "num_transforms": num_transforms,
        }
        if embedding_net is not None:
            build_kwargs["embedding_net"] = embedding_net

        if flow_type.lower() == "nsf":
            build_kwargs["num_bins"] = num_bins
            self.density_estimator = build_nsf(theta_sample, x_sample, **build_kwargs)
        elif flow_type.lower() == "maf":
            self.density_estimator = build_maf(theta_sample, x_sample, **build_kwargs)
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

        self.density_estimator = self.density_estimator.to(self.device)
        n_params = sum(
            p.numel() for p in self.density_estimator.parameters() if p.requires_grad
        )
        logger.info(f"  Density estimator parameters: {n_params:,}")
        return self.density_estimator

    def _generate_val_set(self, val_size: int, batch_size: int):
        """Generate a fixed validation set once, with a deterministic seed."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(12345)
        thetas, xs = [], []
        remaining = val_size
        while remaining > 0:
            b = min(batch_size, remaining)
            theta, x = self.simulator.generate_batch(b, generator=gen)
            thetas.append(theta)
            xs.append(x)
            remaining -= b
        return torch.cat(thetas), torch.cat(xs)

    def _validation_loss(self, val_theta, val_x, batch_size: int) -> dict[str, float]:
        """Per-sample validation loss statistics.

        The mean alone is misleading when posteriors are sharp: a handful of
        val points in near-zero-density regions can move the 50k-sample mean
        by whole nats. The median tracks typical performance; p99/max expose
        whether divergence is a tail phenomenon rather than a global one.
        """
        self.density_estimator.eval()
        all_losses = []
        with torch.no_grad():
            for i in range(0, len(val_theta), batch_size):
                tb = val_theta[i : i + batch_size]
                xb = val_x[i : i + batch_size]
                all_losses.append(self.density_estimator.loss(tb, condition=xb))
        if not all_losses:
            return {"mean": float("inf"), "median": float("inf"),
                    "p99": float("inf"), "max": float("inf")}
        losses = torch.cat(all_losses)
        return {
            "mean": losses.mean().item(),
            "median": losses.median().item(),
            "p99": torch.quantile(losses, 0.99).item(),
            "max": losses.max().item(),
        }

    def train(
        self,
        prior,
        embedding_net: nn.Module | None = None,
        flow_type: str = "nsf",
        hidden_features: int = 256,
        num_transforms: int = 15,
        num_bins: int = 16,
        learning_rate: float = 5e-4,
        training_batch_size: int = 1024,
        steps_per_epoch: int = 200,
        val_size: int = 10000,
        max_epochs: int = 500,
        stop_after_epochs: int = 20,
        clip_grad_norm: float | None = 5.0,
        tf32: bool = True,
        amp: str = "none",
        show_progress: bool = True,
    ) -> tuple[nn.Module, dict[str, list[float]]]:
        self.prior = prior

        # --- GPU performance levers -------------------------------------
        cuda = self.device == "cuda" or str(self.device).startswith("cuda")
        if tf32 and cuda:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        amp = (amp or "none").lower()
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
        use_amp = amp_dtype is not None and cuda
        # fp16 needs loss scaling; bf16 does not.
        scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and cuda))
        if amp != "none" and not cuda:
            logger.warning("amp=%s requested but device is not CUDA; ignoring", amp)
        logger.info(f"  TF32: {tf32 and cuda}, AMP: {amp if use_amp else 'none'}")

        # Build density estimator from one freshly generated batch
        theta0, x0 = self.simulator.generate_batch(
            min(5000, steps_per_epoch * training_batch_size)
        )
        self.build_density_estimator(
            theta_sample=theta0,
            x_sample=x0,
            embedding_net=embedding_net,
            flow_type=flow_type,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            num_bins=num_bins,
        )
        del theta0, x0

        # Fixed validation set
        val_theta, val_x = self._generate_val_set(val_size, training_batch_size)

        optimizer = Adam(self.density_estimator.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        logger.info("Starting on-the-fly GPU NPE training")
        logger.info(
            f"  steps/epoch: {steps_per_epoch}, batch: {training_batch_size}, "
            f"val_size: {val_size}, device: {self.device}"
        )

        for epoch in range(max_epochs):
            self.density_estimator.train()
            train_loss_sum, n_train = 0.0, 0

            pbar = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch + 1}/{max_epochs}",
                disable=not show_progress,
                leave=False,
            )
            for _ in pbar:
                theta_batch, x_batch = self.simulator.generate_batch(
                    training_batch_size
                )
                optimizer.zero_grad()
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype or torch.float16, enabled=use_amp
                ):
                    losses = self.density_estimator.loss(
                        theta_batch, condition=x_batch
                    )
                    loss = losses.mean()
                scaler.scale(loss).backward()
                if clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.density_estimator.parameters(), clip_grad_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                train_loss_sum += losses.sum().item()
                n_train += len(losses)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss = train_loss_sum / n_train
            val_stats = self._validation_loss(val_theta, val_x, training_batch_size)
            val_loss = val_stats["mean"]

            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_median"].append(val_stats["median"])
            self.training_history["val_p99"].append(val_stats["p99"])
            self.training_history["val_max"].append(val_stats["max"])
            self.training_history["learning_rates"].append(
                optimizer.param_groups[0]["lr"]
            )
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(self.density_estimator.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            logger.info(
                f"Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}, "
                f"best={best_val_loss:.4f}, "
                f"patience={epochs_without_improvement}/{stop_after_epochs}"
            )
            logger.info(
                f"  val per-sample: median={val_stats['median']:.4f}, "
                f"p99={val_stats['p99']:.2f}, max={val_stats['max']:.2f}"
            )

            if epochs_without_improvement >= stop_after_epochs:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.density_estimator.load_state_dict(best_state)
            logger.info(f"Restored best model (val_loss={best_val_loss:.4f})")

        return self.density_estimator, self.training_history

    def build_posterior(self, prior=None) -> DirectPosterior:
        if self.density_estimator is None:
            raise RuntimeError("No density estimator trained yet")
        if prior is None:
            prior = self.prior
        return DirectPosterior(
            posterior_estimator=self.density_estimator, prior=prior
        )
