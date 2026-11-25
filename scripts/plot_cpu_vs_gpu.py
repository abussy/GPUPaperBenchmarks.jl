#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "matplotlib",
# ]
# ///
"""Grouped bar plot comparing timings across CPU, CUDA and AMDGPU.

Reads the timing CSVs produced by scripts/run_benchmarks.jl:

  * CPU timings are scattered over benchmark_results/CPU/timings/mod*_1rank_1thread.csv
    (each modN file covers a subset of the benchmark systems)
  * CUDA / AMDGPU timings live in a single GPU_1rank_1thread.csv each

Only the precomputed `repeat == "avg"` rows are used. Systems that are not
present in every architecture are dropped (with a warning) so the bars align.

Run with:
    uv run scripts/plot_cpu_vs_gpu.py
(any Python >= 3.10 with pandas + matplotlib installed works as well)

Modes (--mode) select which timings are shown on the y axis:
  * scf:          SCF times (default)
  * forces:       forces times
  * stresses:     stresses times (no data yet)
  * scf_forces:   SCF + forces times summed into one bar
  * scf_stresses: SCF + stresses times summed into one bar

Also supports --yscale linear to switch the y-axis scaling (default: log):
    uv run scripts/plot_cpu_vs_gpu.py --mode scf_forces --yscale linear

Bar annotations always show the speedup vs the CPU baseline (GPU bars only).

Output: plots/<OUTPUT_NAME>_<mode>.pdf
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

# --- Configuration -----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark_results"
OUTPUT_DIR = REPO_ROOT / "plots"

OUTPUT_NAME = "cpu_cuda_amdgpu"      # saved as plots/<OUTPUT_NAME>_<mode>.pdf
Y_SCALE = "log"                      # y-axis scaling: "log" or "linear"
MODE = "scf"                         # default mode: a key of MODE_METRICS
BASELINE = "CPU"                     # reference architecture for speedup mode
LABEL_FONTSIZE = 12                   # font size of the speedup bar annotations

# Font sizes. The figures are ~13.5 in wide but are scaled to the paper's
# full text width (~7 in) via \includegraphics, i.e. shrunk to roughly half;
# these sizes are chosen so the printed text stays legible (~7-8 pt).
plt.rcParams.update({
    "font.size": 15,        # base size (e.g. log-axis offset text)
    "axes.labelsize": 16,   # y-axis label
    "xtick.labelsize": 13,  # rotated system names
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
})

# Modes (--mode): mode name -> timing columns summed into the bar values.
# Columns without any data (all NaN) are reported and skipped in the sum.
MODE_METRICS = {
    "scf": ("t_scf",),
    "forces": ("t_forces",),
    "stresses": ("t_stresses",),
    "scf_forces": ("t_scf", "t_forces"),
    "scf_stresses": ("t_scf", "t_stresses"),
}

# Y-axis label per mode, without units (the plotting code appends " (s)");
# combined modes read e.g. "10 SCF steps + forces timing".
MODE_YLABELS = {
    "scf": "SCF time",
    "forces": "Forces time",
    "stresses": "Stresses time",
    "scf_forces": "10 SCF steps + forces timing",
    "scf_stresses": "SCF + stresses time",
}

# One entry per architecture, in bar order. The glob may contain wildcards;
# all matching files are concatenated.
SOURCES = [
    ("CPU", RESULTS_DIR / "CPU" / "timings" / "mod*_1rank_1thread.csv"),
    ("CUDA", RESULTS_DIR / "CUDA" / "timings" / "GPU_1rank_1thread.csv"),
    ("AMDGPU", RESULTS_DIR / "AMDGPU" / "timings" / "GPU_1rank_1thread.csv"),
]

ARCH_COLORS = {
    "CPU": "#4C72B0",    # muted blue
    "CUDA": "#76B900",   # NVIDIA green
    "AMDGPU": "#ED1C24", # AMD red
}

# Pretty x-axis labels: benchmark system name -> display label. Systems not
# listed here fall back to their internal name. Subscripts use mathtext
# (e.g. "MoS$_2$" renders as MoS with 2 as subscript).
SYSTEM_LABELS = {
    "aluminium_primitive": "Al_primitive",
    "aluminium_supercell": "Al_supercell",
    "copt_primitive": "CoPt_primitive",
    "copt_supercell": "CoPt_supercell",
    "gaas_110_slab": "GaAs_110_slab",
    "gold_111_slab": "Au_111_slab",
    "hbn_primitive": "hBN_primitive",
    "hbn_supercell": "hBN_supercell",
    "iron_primitive": "Fe_primitive",
    "iron_supercell": "Fe_supercell",
    "mgo_primitive": "MgO_primitive",
    "mgo_supercell": "MgO_supercell",
    "mos2_primitive": "MoS$_2$_primitive",
    "mos2_supercell": "MoS$_2$_supercell",
    "silicon_primitive": "Si_primitive",
    "silicon_supercell": "Si_supercell",
    "zro2_primitive": "ZrO$_2$_primitive",
    "zro2_supercell": "ZrO$_2$_supercell",
}

# --- Data loading ------------------------------------------------------------


def load_avg_timings(label: str, pattern: Path, metrics: list[str]) -> pd.DataFrame:
    """Load all CSVs matching `pattern`; one avg row per system, `metrics` columns."""
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"No timing files found for {label}: {pattern}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"{missing} not in columns of {pattern}: {list(df.columns)}")

    avg = df[df["repeat"].astype(str) == "avg"]

    # Some files contain several benchmark runs of the same system/config
    # (e.g. re-runs). Average those duplicate rows into one value per system.
    if avg["system"].duplicated().any():
        config_cols = [c for c in ("natoms", "nelectrons", "Ecut", "kgrid")
                       if c in avg.columns]
        inconsistent = avg.groupby("system")[config_cols].nunique().max(axis=1) > 1
        if inconsistent.any():
            print(f"WARNING: {label}: differing configurations for systems "
                  f"{inconsistent[inconsistent].index.tolist()}; they are "
                  f"averaged together anyway")
        result = avg.groupby("system")[metrics].mean()
    else:
        result = avg.set_index("system")[metrics]

    print(f"  {label}: {len(result)} systems from {len(files)} file(s)")
    return result


def collect_data(metrics: list[str]) -> pd.DataFrame:
    """Return (system x [architecture, metric]) data with MultiIndex columns,
    keeping only systems present for every architecture.

    Metrics without any data at all (e.g. t_stresses before the stress
    benchmarks are run) are reported and do not cause systems to be dropped.
    """
    print("Loading timings:")
    frames = []
    for label, pattern in SOURCES:
        df = load_avg_timings(label, pattern, metrics)
        df.columns = pd.MultiIndex.from_product([[label], df.columns],
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
        print(f"WARNING: dropping systems missing in some architecture: "
              f"{list(data.index[missing])}")
        data = data[~missing]
    return data


# --- Plotting ----------------------------------------------------------------


def speedup_vs_baseline(data: pd.DataFrame) -> pd.DataFrame:
    """Convert times (system x architecture) to speedup against BASELINE."""
    if BASELINE not in data.columns:
        raise ValueError(f"speedup annotations need a {BASELINE!r} entry in SOURCES")
    return data.rdiv(data[BASELINE], axis=0)


def mode_values(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Bar values for `mode`: sum of its metrics (system x architecture).

    Metrics without any data (all NaN) are skipped in the sum, so e.g.
    scf_stresses equals the SCF times until stresses data is available.
    """
    metrics = MODE_METRICS[mode]
    metric_level = data.columns.get_level_values("metric")
    components = data[data.columns[metric_level.isin(metrics)]]
    archs = data.columns.get_level_values("arch").unique()
    return pd.DataFrame({arch: components[arch].sum(axis=1) for arch in archs})


def annotate_speedup(ax, positions, values, labels) -> None:
    """Annotate bars with their speedup (one decimal, 'x' suffix)."""
    for xpos, value, label in zip(positions, values, labels):
        if np.isnan(value):
            continue
        ax.annotate(
            f"{label:.1f}x",
            (xpos, value),
            xytext=(1.5, 1), textcoords="offset points",
            ha="center", va="bottom", rotation=90,
            fontsize=LABEL_FONTSIZE,
        )


def add_legend(ax, n_entries: int) -> None:
    """Legend as a box in the upper right, with the y-axis top raised so the
    box never covers the bars.

    Must be called on the final layout (after tight_layout): the legend's
    on-screen extent and the tallest bar/annotation underneath it are
    measured on a first draw, and the top of the y range is extended until
    they clear the legend's bottom edge. The extension is computed exactly
    from the geometry (multiplicative in log10 on log scales, additive on
    linear ones), with small corrective steps for rounding.
    """
    leg = ax.legend(
        loc="upper right",
        ncol=n_entries,
        handlelength=1.2, columnspacing=1.2,
    )

    margin_px = 8.0   # clearance between the data and the legend box
    max_steps = 5     # corrective expansions after the exact computation
    fig = ax.figure
    is_log = ax.get_yscale() == "log"

    def tops_under_legend() -> list[float]:
        """Display-coord top edges of bars/annotations under the legend."""
        leg_bb = leg.get_window_extent()
        tops = []
        for artist in (*ax.patches, *ax.texts):
            bb = artist.get_window_extent()
            if not (np.isfinite(bb.x0) and np.isfinite(bb.y1)):
                continue                  # e.g. bars with NaN (missing) data
            if bb.x1 > leg_bb.x0 and bb.x0 < leg_bb.x1:
                tops.append(bb.y1)
        return tops

    def raise_top(growth: float) -> None:
        """Extend the y range upward by `growth` of its current span."""
        bottom, top = ax.get_ylim()
        if is_log:
            ax.set_ylim(top=bottom * (top / bottom) ** (1 + growth))
        else:
            ax.set_ylim(top=top + growth * (top - bottom))

    fig.canvas.draw()
    targets = tops_under_legend()
    if not targets:
        return                            # nothing under the legend at all
    ax_bb = ax.get_window_extent()
    clearance = leg.get_window_extent().ymin - margin_px
    highest = max(targets)
    if highest <= clearance:
        return                            # already clear

    # Exact limit placing the highest bar/annotation at the legend's bottom
    # (minus the margin): the display fraction is affine in the data on
    # linear scales and in log10(data) on log ones.
    f_new = (clearance - ax_bb.y0) / ax_bb.height
    f_cur = (highest - ax_bb.y0) / ax_bb.height
    if not 0 < f_new < 1:
        return                            # degenerate layout; keep autoscale
    bottom, top = ax.get_ylim()
    if is_log:
        new_top = bottom * (top / bottom) ** (f_cur / f_new)
    else:
        new_top = bottom + (top - bottom) * (f_cur / f_new)
    ax.set_ylim(top=new_top)

    # Corrective steps in case rounding left the data touching the legend.
    for _ in range(max_steps):
        fig.canvas.draw()
        targets = tops_under_legend()
        if not targets or max(targets) <= clearance:
            break
        raise_top(0.05)


def draw_grouped_bars(ax, values, archs, x, width, label_values) -> None:
    """One bar per (system, architecture); GPU bars get speedup annotations."""
    n_arch = len(archs)
    for i, arch in enumerate(archs):
        offset = (i - (n_arch - 1) / 2) * width
        ax.bar(
            x + offset, values[arch], width,
            label=arch, color=ARCH_COLORS.get(arch), edgecolor="black",
            linewidth=0.3,
        )
        if arch == BASELINE:
            continue  # the baseline bars need no speedup annotation
        annotate_speedup(ax, x + offset, values[arch], label_values[arch])


def plot_comparison(data: pd.DataFrame, y_scale: str, mode: str) -> plt.Figure:
    """Grouped bar plot: benchmark systems on x, one bar per architecture.

    `data` has MultiIndex columns (architecture, metric); the bar heights are
    the sum of the metrics selected by `mode` (see MODE_METRICS).
    """
    n_systems = data.shape[0]
    archs = list(data.columns.get_level_values("arch").unique())
    n_arch = len(archs)
    x = np.arange(n_systems)
    width = 0.8 / n_arch

    ylabel = f"{MODE_YLABELS[mode]} (s)"

    values = mode_values(data, mode)
    label_values = speedup_vs_baseline(values)

    fig, ax = plt.subplots(figsize=(max(10, 0.75 * n_systems), 6.0))

    draw_grouped_bars(ax, values, archs, x, width, label_values)

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
    add_legend(ax, n_arch)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=tuple(MODE_METRICS), default=MODE,
                        help="timings to plot: SCF, forces, or their sum "
                             f"(default: {MODE})")
    parser.add_argument("--yscale", choices=("log", "linear"), default=Y_SCALE,
                        help=f"y-axis scaling (default: {Y_SCALE})")
    args = parser.parse_args()

    data = collect_data(list(MODE_METRICS[args.mode]))
    n_arch = len(data.columns.get_level_values("arch").unique())
    print(f"\nPlotting {args.mode} for {len(data)} systems x {n_arch} "
          f"architectures ({args.yscale} y-axis):\n{data.to_string()}\n")

    fig = plot_comparison(data, args.yscale, args.mode)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{OUTPUT_NAME}_{args.mode}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
