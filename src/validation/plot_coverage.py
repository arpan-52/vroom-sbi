"""
Plot per-parameter coverage from one or more ``online_probe`` runs.

``online_probe`` prints a fixed table (param / bias / rmse / cov68 / cov90) per
posterior but saves no figure. This parses that exact stdout (as captured in a
log) and draws:

  - cov68 and cov90 per parameter, with the 0.68 / 0.90 ideal lines. Points
    above the line = over-covering (conservative); below = over-confident.
  - bias and rmse per parameter (prior-width units).

Reading the saved log avoids re-running the GPU probe. The parser keys off the
``Model : <type>  N=<k>`` header and the dashed table, so it is robust to the
nflows deprecation warnings interleaved in the log.

Run:
    pixi run -e gpu python -m src.validation.plot_coverage \
        validation_results/coverage_probe.log \
        --output validation_results/coverage_summary.png
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HEADER = re.compile(r"Model\s*:\s*(\S+)\s+N=(\d+)")
_ROW = re.compile(
    r"^([A-Za-z]\w*?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$"
)


def parse_coverage_log(path: str | Path) -> list[dict]:
    """Return one block dict per posterior: label + per-param arrays."""
    blocks: list[dict] = []
    cur: dict | None = None
    for line in Path(path).read_text().splitlines():
        h = _HEADER.search(line)
        if h:
            cur = {
                "label": f"{h.group(1)} N={h.group(2)}",
                "params": [],
                "bias": [],
                "rmse": [],
                "cov68": [],
                "cov90": [],
            }
            blocks.append(cur)
            continue
        m = _ROW.match(line.strip())
        if m and cur is not None:
            cur["params"].append(m.group(1))
            cur["bias"].append(float(m.group(2)))
            cur["rmse"].append(float(m.group(3)))
            cur["cov68"].append(float(m.group(4)))
            cur["cov90"].append(float(m.group(5)))
    return [b for b in blocks if b["params"]]


def plot_coverage(logs: list[str | Path], output: str | Path) -> Path:
    blocks: list[dict] = []
    for lg in logs:
        blocks.extend(parse_coverage_log(lg))
    if not blocks:
        raise ValueError(f"No coverage tables parsed from {logs}")

    n = len(blocks)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.6 * n), squeeze=False)
    for r, b in enumerate(blocks):
        x = np.arange(len(b["params"]))
        ax_c, ax_e = axes[r, 0], axes[r, 1]

        ax_c.axhline(0.68, color="tab:blue", ls="--", lw=1, alpha=0.7)
        ax_c.axhline(0.90, color="tab:orange", ls="--", lw=1, alpha=0.7)
        ax_c.plot(x, b["cov68"], "o-", color="tab:blue", ms=4, label="cov68")
        ax_c.plot(x, b["cov90"], "s-", color="tab:orange", ms=4, label="cov90")
        ax_c.set_ylim(0.3, 1.02)
        ax_c.set_ylabel(f"{b['label']}\ncoverage", fontsize=9)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(b["params"], rotation=45, fontsize=7, ha="right")
        ax_c.grid(alpha=0.3)

        ax_e.axhline(0.0, color="0.6", lw=0.8)
        ax_e.bar(x - 0.2, b["bias"], width=0.4, color="tab:purple", label="bias")
        ax_e.bar(x + 0.2, b["rmse"], width=0.4, color="tab:green", label="rmse")
        ax_e.set_ylabel("prior-width units", fontsize=8)
        ax_e.set_xticks(x)
        ax_e.set_xticklabels(b["params"], rotation=45, fontsize=7, ha="right")
        ax_e.grid(alpha=0.3)
        if r == 0:
            ax_c.legend(fontsize=7, loc="lower left")
            ax_e.legend(fontsize=7, loc="upper right")
    axes[0, 0].set_title("Coverage (dashed = ideal)")
    axes[0, 1].set_title("Bias / RMSE")
    fig.suptitle("Coverage probe summary", y=1.0)
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", help="coverage probe log file(s)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = plot_coverage(args.logs, args.output)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
