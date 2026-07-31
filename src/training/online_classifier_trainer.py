"""
On-the-fly GPU training for the model-selection classifier.

The classifier consumes ``[Q, U, weights]`` (3 * n_freq) — exactly the data
contract ``GPUSimulator.generate_batch`` produces — and predicts the class
index for ``(model_type, n_components)``. This module builds one ``GPUSimulator``
per class and yields balanced, freshly generated, labelled batches on-device,
then reuses the existing ``ClassifierTrainer`` for the training/eval/save loop.

Class ordering mirrors ``data_loader.prepare_classifier_data`` exactly so the
saved classifier stays compatible with ``inference/engine.py``:
  - cross-model: nested ``for model_type: for n_comp`` -> 0,1,2,...
  - single-model: ``class = n_comp - min_components``
"""

import logging
from pathlib import Path
from typing import Any

import torch

from ..config import Configuration
from ..simulator.gpu_simulator import GPUSimulator
from .classifier_trainer import ClassifierTrainer, _save_classifier_training_plot

logger = logging.getLogger(__name__)


class OnlineClassifierData:
    """Iterable of ``{"x", "label"}`` batches generated on-the-fly.

    Each batch is balanced across classes (split as evenly as possible). With
    ``fixed=True`` the batches are generated once and cached — used for a stable
    validation set.
    """

    def __init__(
        self,
        simulators: list[GPUSimulator],
        labels: list[int],
        batch_size: int,
        steps: int,
        device: str,
        seed: int | None = None,
        fixed: bool = False,
    ):
        self.simulators = simulators
        self.labels = labels
        self.batch_size = batch_size
        self.steps = steps
        self.device = device
        self.fixed = fixed
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(seed)
        self._cache = [self._make_batch() for _ in range(steps)] if fixed else None

    def _make_batch(self) -> dict[str, torch.Tensor]:
        C = len(self.simulators)
        base = self.batch_size // C
        rem = self.batch_size % C
        xs, ys = [], []
        for i, (sim, label) in enumerate(zip(self.simulators, self.labels)):
            n = base + (1 if i < rem else 0)
            if n == 0:
                continue
            _, x = sim.generate_batch(n, generator=self.generator)
            xs.append(x)
            ys.append(
                torch.full((n,), label, dtype=torch.long, device=self.device)
            )
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        # Shuffle within the batch so classes aren't contiguous
        perm = torch.randperm(len(y), device=self.device, generator=self.generator)
        return {"x": x[perm], "label": y[perm]}

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        if self.fixed:
            yield from self._cache
        else:
            for _ in range(self.steps):
                yield self._make_batch()


def _build_class_mapping(
    min_components, max_components, model_types, cross_model_training
):
    """Replicate prepare_classifier_data class ordering."""
    class_to_label = {}
    specs = []  # (model_type, n_comp, class_idx)
    if cross_model_training:
        idx = 0
        for mt in model_types:
            for n in range(min_components, max_components + 1):
                class_to_label[idx] = (mt, n)
                specs.append((mt, n, idx))
                idx += 1
    else:
        mt = model_types[0]
        for n in range(min_components, max_components + 1):
            idx = n - min_components
            class_to_label[idx] = (mt, n)
            specs.append((mt, n, idx))
    return class_to_label, specs


def train_classifier_online(
    config: Configuration,
    output_dir: Path,
    min_components: int = 1,
    max_components: int = 5,
    model_types: list[str] | None = None,
    cross_model_training: bool = False,
) -> dict[str, Any]:
    """Train the model-selection classifier with GPU on-the-fly data."""
    if model_types is None:
        model_types = config.physics.model_types

    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    sampling_method = getattr(config.training, "sampling_method", "uniform")
    class_to_label, specs = _build_class_mapping(
        min_components, max_components, model_types, cross_model_training
    )
    n_classes = len(class_to_label)
    logger.info(
        f"Online classifier: {n_classes} classes "
        f"(cross_model={cross_model_training})"
    )

    simulators = [
        GPUSimulator(config, mt, n, device=device, sampling_method=sampling_method)
        for (mt, n, _idx) in specs
    ]
    labels = [idx for (_mt, _n, idx) in specs]
    n_freq = simulators[0].n_freq

    batch_size = config.classifier.batch_size
    steps_per_epoch = config.training.steps_per_epoch
    val_steps = max(1, config.training.val_size // batch_size)

    train_loader = OnlineClassifierData(
        simulators, labels, batch_size, steps_per_epoch, device
    )
    val_loader = OnlineClassifierData(
        simulators, labels, batch_size, val_steps, device, seed=777, fixed=True
    )

    trainer = ClassifierTrainer(
        n_freq=n_freq,
        n_classes=n_classes,
        config=config.classifier,
        device=device,
    )

    history = trainer.train(train_loader, val_loader)
    eval_results = trainer.evaluate(val_loader)

    trainer.class_to_label = class_to_label
    save_path = output_dir / "classifier.pt"
    trainer.save(str(save_path))
    _save_classifier_training_plot(history, output_dir)

    result = {
        "model_path": str(save_path),
        "n_freq": n_freq,
        "n_classes": n_classes,
        "max_components": max_components,
        "model_types": model_types,
        "cross_model_training": cross_model_training,
        "class_to_label": class_to_label,
        "history": history,
        "final_val_accuracy": eval_results["accuracy"],
    }
    for key, value in eval_results.items():
        if key != "accuracy":
            result[key] = value
    return result
