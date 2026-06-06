"""
Profiling harness for the GPU on-the-fly training path.

Uses ``torch.profiler`` to measure where time goes in a short online-training
run: batch generation vs. the NPE training step (forward/backward/optimizer).
Reports per-step throughput and, on CUDA, whether the GPU stays busy (low
self-CPU time between kernels indicates the GPU is staying hot).

Intended use: a short sanity/throughput run on local hardware (e.g. a 1080 Ti)
to confirm the path scales before committing to rented A100 time.
"""

import logging
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from ..config import Configuration
from ..simulator.gpu_simulator import GPUSimulator
from ..training.networks import SpectralEmbedding
from ..training.online_trainer import OnlineNPETrainer

logger = logging.getLogger(__name__)


def profile_online_training(
    config: Configuration,
    model_type: str = "faraday_thin",
    n_components: int = 1,
    n_steps: int = 50,
    warmup: int = 5,
    trace_path: Path | None = None,
) -> dict:
    """Profile a short online-training run.

    Parameters
    ----------
    config : Configuration
        Full config; ``config.freq_file`` selects the band grid,
        ``config.training.training_batch_size`` the batch size.
    model_type, n_components : str, int
        Which posterior model to profile.
    n_steps : int
        Number of profiled training steps.
    warmup : int
        Unprofiled warmup steps (CUDA kernel autotuning, allocator warmup).
    trace_path : Path, optional
        If given, export a Chrome trace (view in chrome://tracing).

    Returns
    -------
    dict
        Timing summary (device, batch size, n_freq, gen/step/total ms/step,
        samples/sec).
    """
    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — profiling on CPU")
        device = "cpu"
    cuda = device == "cuda"

    # GPU performance levers (match the online trainer)
    tf32 = getattr(config.training, "tf32", True)
    amp = (getattr(config.training, "amp", "none") or "none").lower()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    use_amp = amp_dtype is not None and cuda
    if tf32 and cuda:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    batch_size = config.training.training_batch_size
    arch = config.sbi.get_architecture(n_components)

    sim = GPUSimulator(
        config,
        model_type,
        n_components,
        device=device,
        sampling_method=getattr(config.training, "sampling_method", "uniform"),
    )
    emb = SpectralEmbedding(
        input_dim=3 * sim.n_freq, output_dim=config.sbi.embedding_dim
    ).to(device)

    trainer = OnlineNPETrainer(sim, device=device)
    theta0, x0 = sim.generate_batch(batch_size)
    trainer.build_density_estimator(
        theta0, x0, embedding_net=emb, flow_type=config.sbi.model,
        hidden_features=arch.hidden_features, num_transforms=arch.num_transforms,
        num_bins=config.sbi.num_bins,
    )
    de = trainer.density_estimator
    optimizer = torch.optim.Adam(de.parameters(), lr=config.training.learning_rate)
    de.train()

    def _sync():
        if cuda:
            torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        theta, x = sim.generate_batch(batch_size)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=amp_dtype or torch.float16, enabled=use_amp):
            loss = de.loss(theta, condition=x).mean()
        loss.backward()
        optimizer.step()
    _sync()

    activities = [ProfilerActivity.CPU]
    if cuda:
        activities.append(ProfilerActivity.CUDA)

    gen_ms, step_ms = 0.0, 0.0
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(n_steps):
            _sync()
            tg = time.perf_counter()
            with record_function("generate_batch"):
                theta, x = sim.generate_batch(batch_size)
            _sync()
            gen_ms += (time.perf_counter() - tg) * 1e3

            ts = time.perf_counter()
            with record_function("train_step"):
                optimizer.zero_grad()
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype or torch.float16, enabled=use_amp
                ):
                    loss = de.loss(theta, condition=x).mean()
                loss.backward()
                optimizer.step()
            _sync()
            step_ms += (time.perf_counter() - ts) * 1e3
    total_s = time.perf_counter() - t0

    if trace_path is not None:
        prof.export_chrome_trace(str(trace_path))

    sort_key = "cuda_time_total" if cuda else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=15)

    summary = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "model": f"{model_type}_n{n_components}",
        "n_freq": sim.n_freq,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "gen_ms_per_step": gen_ms / n_steps,
        "step_ms_per_step": step_ms / n_steps,
        "total_ms_per_step": total_s / n_steps * 1e3,
        "samples_per_sec": batch_size * n_steps / total_s,
        "gen_fraction": gen_ms / (gen_ms + step_ms) if (gen_ms + step_ms) else 0.0,
    }
    if cuda:
        summary["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1024**2

    return {"summary": summary, "table": table}


def format_summary(result: dict) -> str:
    """Human-readable one-block summary."""
    s = result["summary"]
    lines = [
        f"Device:          {s['device']}" + (f" ({s['gpu_name']})" if s.get("gpu_name") else ""),
        f"Model / n_freq:  {s['model']} / {s['n_freq']} channels",
        f"Batch size:      {s['batch_size']}",
        f"Generation:      {s['gen_ms_per_step']:.2f} ms/step ({s['gen_fraction'] * 100:.1f}% of compute)",
        f"Train step:      {s['step_ms_per_step']:.2f} ms/step",
        f"Total:           {s['total_ms_per_step']:.2f} ms/step",
        f"Throughput:      {s['samples_per_sec']:,.0f} samples/sec",
    ]
    if "peak_mem_mb" in s:
        lines.append(f"Peak GPU mem:    {s['peak_mem_mb']:,.0f} MB")
    lines.append("")
    lines.append(result["table"])
    return "\n".join(lines)


def _main() -> None:
    import argparse

    from ..config import Configuration

    p = argparse.ArgumentParser(description="Profile the GPU on-the-fly path")
    p.add_argument("--config", default="config_a100.yaml")
    p.add_argument("--model", default="faraday_thin")
    p.add_argument("--n-components", type=int, default=1)
    p.add_argument("--freq-file", default=None, help="override config freq_file")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trace", default=None, help="export chrome trace to this path")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    cfg = Configuration.from_yaml(args.config)
    if args.freq_file:
        cfg.freq_file = args.freq_file
    if args.batch_size:
        cfg.training.training_batch_size = args.batch_size

    result = profile_online_training(
        cfg,
        model_type=args.model,
        n_components=args.n_components,
        n_steps=args.steps,
        warmup=args.warmup,
        trace_path=Path(args.trace) if args.trace else None,
    )
    print(format_summary(result))


if __name__ == "__main__":
    _main()
