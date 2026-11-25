#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "matplotlib",
# ]
# ///
"""Grouped bar plot comparing single-configuration vs fastest-configuration
timings for CPU and CUDA.

Four bars per system, in this order:

  * CPU single:  1 rank / 1 thread runs (all mod*_1rank_1thread.csv files)
  * CPU best:    the fastest of every mod*_Rrank_Tthread.csv configuration
  * CUDA single: 1 rank / 1 thread runs (GPU_1rank_1thread.csv)
  * CUDA best:   the fastest of every CSV in benchmark_results/CUDA/timings

CPU bars are blue and CUDA bars green, matching the other figures; the
'single' bars are tinted toward white (lower saturation) so single and best
are told apart at the same hue -- also in grayscale prints.

Each CUDA bar is annotated with its speedup against the CPU bar of the same
variant (single vs single, best vs best).

The loaders and plotting parameters are shared with scripts/plot_cpu_vs_gpu.py
and scripts/plot_best_cpu_vs_cuda.py (imported; the script directory is on
sys.path when run as a script), so all figures share the same style.

Run with:
    uv run scripts/plot_single_vs_best.py
    uv run scripts/plot_single_vs_best.py --mode scf_forces --yscale linear
    uv run scripts/plot_single_vs_best.py --no-speedups   (hide annotations)

Output: plots/single_vs_best_<mode>.pdf
"""

from __future__ import annotations

import argparse
import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from plot_best_cpu_vs_cuda import load_best_timings, mode_totals
from plot_cpu_vs_gpu import (
    ARCH_COLORS,
    MODE,
    MODE_METRICS,
    MODE_YLABELS,
    OUTPUT_DIR,
    RESULTS_DIR,
    SYSTEM_LABELS,
    Y_SCALE,
    add_legend,
    load_avg_timings,
)

# --- Configuration -----------------------------------------------------------

OUTPUT_NAME = "single_vs_best"       # saved as plots/<OUTPUT_NAME>_<mode>.pdf

# The four bar variants, in plot order: the two CPU columns first, then the
# two CUDA columns; single before best within each architecture.
VARIANTS = ["CPU single", "CPU best", "CUDA single", "CUDA best"]

LEGEND_LABELS = {
    "CPU single": "CPU (single core)",
    "CPU best": "CPU (fastest config)",
    "CUDA single": "CUDA (single GPU)",
    "CUDA best": "CUDA (fastest config)",
}

# Fraction of the base colour kept for the 'single' bars; the rest is blended
# toward white (lower saturation, higher lightness than the 'best' bars).
SINGLE_TINT = 0.55

# Font size of the speedup annotations. Smaller than LABEL_FONTSIZE because
# four bars per group leave less horizontal room for the rotated labels.
ANNOTATION_FONTSIZE = 10

# Timing sources for the 'single' variants (the 'best' variants scan every
# CSV in the directories below, see load_best_timings).
CPU_SINGLE_GLOB = RESULTS_DIR / "CPU" / "timings" / "mod*_1rank_1thread.csv"
CUDA_SINGLE_GLOB = RESULTS_DIR / "CUDA" / "timings" / "GPU_1rank_1thread.csv"
CPU_TIMINGS_DIR = RESULTS_DIR / "CPU" / "timings"
CUDA_TIMINGS_DIR = RESULTS_DIR / "CUDA" / "timings"


def tint(color: str, keep: float = SINGLE_TINT) -> str:
    """Blend `color` toward white, keeping a fraction `keep` of it.

    This lowers the saturation (and raises the lightness) of the base colour,
    which distinguishes the 'single' bars from the fully saturated 'best'
    bars at the same hue -- also in grayscale prints.
    """
    r, g, b = mcolors.to_rgb(color)
    toward_white = lambda c: keep * c + (1 - keep)
    return mcolors.to_hex((toward_white(r), toward_white(g), toward_white(b)))


# --- Data loading ------------------------------------------------------------


def collect_data(mode: str) -> pd.DataFrame:
    """Return (system x [variant, metric]) data with MultiIndex columns.

    The 'arch' level holds the four bar variants (see VARIANTS), so the
    shared mode_totals() works unchanged. Only systems present for every
    variant are kept (with a warning). Metrics without any data at all
    (e.g. t_stresses before the stress benchmarks are run) are reported
    and do not cause systems to be dropped.
    """
    metrics = list(MODE_METRICS[mode])
    print("Loading timings:")
    cpu_single = load_avg_timings("CPU single", CPU_SINGLE_GLOB, metrics)
    cpu_best, _ = load_best_timings("CPU best", CPU_TIMINGS_DIR, metrics)
    cuda_single = load_avg_timings("CUDA single", CUDA_SINGLE_GLOB, metrics)
    cuda_best, _ = load_best_timings("CUDA best", CUDA_TIMINGS_DIR, metrics)

    frames = []
    for name, df in zip(VARIANTS, (cpu_single, cpu_best, cuda_single, cuda_best)):
        df.columns = pd.MultiIndex.from_product([[name], df.columns],
                                                names=("arch", "metric"))
        frames.append(df)
    data = pd.concat(frames, axis=1).sort_index()

    # Metrics without any data (all NaN) would drop every system below;
    # report them and leave them out of the completeness check instead.
    empty_metrics = sorted({m for m in metrics
                            if not data.xs(m, axis=1, level="metric")
                            .notna().any().any()})
    for m in empty_metrics:
        print(f"WARNING: no data at all for {m!r} (all values are NaN); "
              f"it is skipped in sums and completeness checks")
    check_cols = [c for c in data.columns if c[1] not in empty_metrics]

    missing = data[check_cols].isna().any(axis=1)
    if missing.any():
        print(f"WARNING: dropping systems missing in some variant: "
              f"{list(data.index[missing])}")
        data = data[~missing]
    return data


# --- Plotting ----------------------------------------------------------------


def annotate_speedup(ax, positions, values, labels) -> None:
    """Annotate bars with their speedup (one decimal, 'x' suffix).

    Same style as the shared annotate_speedup, but with a smaller font:
    four bars per group leave less room for the rotated labels.
    """
    for xpos, value, label in zip(positions, values, labels):
        if np.isnan(value) or np.isnan(label):
            continue
        ax.annotate(
            f"{label:.1f}x",
            (xpos, value),
            xytext=(1.5, 1), textcoords="offset points",
            ha="center", va="bottom", rotation=90,
            fontsize=ANNOTATION_FONTSIZE,
        )


def plot_comparison(data: pd.DataFrame, y_scale: str, mode: str,
                    annotate: bool = True) -> plt.Figure:
    """Grouped bar plot: benchmark systems on x, four bars per system.

    `data` has MultiIndex columns (variant, metric); the bar heights are the
    sum of the metrics selected by `mode` (see MODE_METRICS). The CUDA speedup
    annotations are skipped when `annotate` is False.
    """
    n_systems = data.shape[0]
    n_var = len(VARIANTS)
    x = np.arange(n_systems)
    width = 0.8 / n_var

    ylabel = f"{MODE_YLABELS[mode]} (s)"

    values = mode_totals(data, mode).reindex(columns=VARIANTS)

    # Fully saturated bars for 'best', tinted ones for 'single', same hue
    # per architecture (blue for CPU, green for CUDA).
    bar_colors = {}
    for variant in VARIANTS:
        base = ARCH_COLORS["CPU"] if variant.startswith("CPU") \
            else ARCH_COLORS["CUDA"]
        bar_colors[variant] = base if variant.endswith("best") else tint(base)

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * n_systems), 6.0))

    for i, variant in enumerate(VARIANTS):
        offset = (i - (n_var - 1) / 2) * width
        ax.bar(
            x + offset, values[variant], width,
            label=LEGEND_LABELS[variant], color=bar_colors[variant],
            edgecolor="black", linewidth=0.3,
        )

    # Speedup of each CUDA bar against the CPU bar of the same variant
    # (single vs single, best vs best); CPU is the baseline architecture.
    # As in speedup_vs_baseline: speedup = CPU time / CUDA time, so values
    # below 1 mean the GPU is slower.
    if annotate:
        for cuda_variant in ("CUDA single", "CUDA best"):
            cpu_variant = cuda_variant.replace("CUDA", "CPU")
            i = VARIANTS.index(cuda_variant)
            offset = (i - (n_var - 1) / 2) * width
            speedups = values[cpu_variant] / values[cuda_variant]
            ok = (values[cuda_variant].notna() & speedups.notna()).to_numpy()
            annotate_speedup(ax, (x + offset)[ok], values[cuda_variant][ok],
                             speedups[ok])

    ax.set_yscale(y_scale)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [SYSTEM_LABELS.get(system, system) for system in data.index])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # Shift x tick labels a tiny bit to the right (screen-space points).
    shift = mtransforms.ScaledTranslation(3 / 72, 0, fig.dpi_scale_trans)
    for tick_label in ax.get_xticklabels():
        tick_label.set_transform(tick_label.get_transform() + shift)
    ax.yaxis.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    # Legend last, on the final layout, so the headroom computation
    # measures the finished figure.
    add_legend(ax, n_var)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=tuple(MODE_METRICS), default=MODE,
                        help="timings to plot: SCF, forces, or their sum "
                             f"(default: {MODE})")
    parser.add_argument("--yscale", choices=("log", "linear"), default=Y_SCALE,
                        help=f"y-axis scaling (default: {Y_SCALE})")
    parser.add_argument("--no-speedups", action="store_true",
                        help="hide the speedup annotations on the CUDA bars")
    args = parser.parse_args()

    data = collect_data(args.mode)
    print(f"\nPlotting {args.mode} for {len(data)} systems x "
          f"{len(VARIANTS)} variants ({args.yscale} y-axis):\n"
          f"{mode_totals(data, args.mode).reindex(columns=VARIANTS).to_string()}\n")

    fig = plot_comparison(data, args.yscale, args.mode,
                          annotate=not args.no_speedups)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTPUT_DIR / f"{OUTPUT_NAME}_{args.mode}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
