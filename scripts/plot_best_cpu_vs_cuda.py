#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "matplotlib",
# ]
# ///
"""Grouped bar plot comparing the fastest CUDA timings against the fastest CPU timings.

Unlike scripts/plot_cpu_vs_gpu.py, which compares fixed 1 rank / 1 thread
runs, this script scans every timing CSV of each architecture and plots the
fastest average timing per system:

  * CPU:  the fastest of all mod*_Rrank_Tthread.csv configuration files
    (every ranks/threads combination) in benchmark_results/CPU/timings
  * CUDA: the fastest of every CSV in benchmark_results/CUDA/timings
    (a single file at the moment; more may appear later)

"Fastest" means: within each file only the precomputed `repeat == "avg"`
rows are used (re-runs of the same system are averaged), and the file whose
sum of the mode's metrics is smallest wins. Its timings are shown, so in
summed modes (e.g. scf_forces) the bar always corresponds to one real
configuration.

Systems that are not present for both architectures are dropped (with a
warning). The plotting code is imported from scripts/plot_cpu_vs_gpu.py (the
script directory is on sys.path when run as a script), so both figures share
the same parameters and style.

Run with:
    uv run scripts/plot_best_cpu_vs_cuda.py
    uv run scripts/plot_best_cpu_vs_cuda.py --mode scf_forces --yscale linear

Output: plots/best_cpu_vs_cuda_<mode>.pdf and plots/best_cpu_vs_cuda_<mode>.dat
(the .dat lists, per architecture and system, the fastest configuration)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from plot_cpu_vs_gpu import (
    METRIC_LABELS,
    MODE,
    MODE_METRICS,
    OUTPUT_DIR,
    RESULTS_DIR,
    SYSTEM_LABELS,
    Y_SCALE,
    draw_grouped_bars,
    speedup_vs_baseline,
)

# --- Configuration -----------------------------------------------------------

OUTPUT_NAME = "best_cpu_vs_cuda"     # saved as plots/<OUTPUT_NAME>_<mode>.pdf

# One entry per architecture, in bar order. Every *.csv in the directory is
# scanned and the fastest avg timing per system wins.
TIMINGS_DIRS = [
    ("CPU", RESULTS_DIR / "CPU" / "timings"),
    ("CUDA", RESULTS_DIR / "CUDA" / "timings"),
]

# --- Data loading ------------------------------------------------------------


def load_best_timings(label: str, timings_dir: Path,
                      metrics: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Fastest avg timing per system across every CSV in `timings_dir`.

    Within a file, duplicate `repeat == "avg"` rows of a system (re-runs) are
    averaged. Across files (rank/thread configurations), the file with the
    smallest sum of `metrics` wins per system and its timings are returned.

    Returns the best timings (system x metrics) and the winning file per
    system (an empty series when no metric has any data at all).
    """
    files = sorted(timings_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No timing files found for {label}: {timings_dir}")

    per_file: dict[str, pd.DataFrame] = {}
    for f in files:
        df = pd.read_csv(f)
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            print(f"WARNING: {label}: skipping {f.name}: missing columns {missing}")
            continue
        avg = df[df["repeat"].astype(str) == "avg"]
        if avg.empty:
            print(f"WARNING: {label}: skipping {f.name}: no 'avg' rows")
            continue

        # Several avg rows of the same system in one file (re-runs) are
        # averaged into one value per system.
        if avg["system"].duplicated().any():
            config_cols = [c for c in ("natoms", "nelectrons", "Ecut", "kgrid")
                           if c in avg.columns]
            inconsistent = avg.groupby("system")[config_cols].nunique().max(axis=1) > 1
            if inconsistent.any():
                print(f"WARNING: {label}/{f.name}: differing configurations for "
                      f"systems {inconsistent[inconsistent].index.tolist()}; they "
                      f"are averaged together anyway")
            per_file[f.name] = avg.groupby("system")[metrics].mean()
        else:
            per_file[f.name] = avg.set_index("system")[metrics]

    if not per_file:
        raise FileNotFoundError(f"No usable timing files for {label} in {timings_dir}")

    # The plotted quantity is the sum of `metrics`; the file (configuration)
    # minimizing that sum wins. Metrics without data are skipped in the sum;
    # min_count=1 gives a NaN total (no winner) for files with no data at all.
    totals = pd.DataFrame({name: frame.sum(axis=1, min_count=1)
                           for name, frame in per_file.items()})

    has_data = totals.notna().any(axis=1)
    if not has_data.any():
        # e.g. the stresses mode before any stress benchmark is run: keep all
        # systems (as NaN rows) so the figure layout matches the other modes.
        print(f"  {label}: no data at all for {metrics} (all values are NaN)")
        systems = sorted(set().union(*(set(f.index) for f in per_file.values())))
        return pd.DataFrame(np.nan, index=systems, columns=metrics), \
            pd.Series(dtype=object, name="fastest_file")

    winner = totals[has_data].idxmin(axis=1)      # fastest file per system
    best = pd.DataFrame.from_dict({system: per_file[fname].loc[system]
                                   for system, fname in winner.items()},
                                  orient="index")[metrics]

    print(f"  {label}: best timings for {len(best)} systems from {len(files)} file(s)")
    for fname, count in winner.value_counts().items():
        print(f"    {fname}: fastest for {count} system(s)")
    return best, winner


def collect_data(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the (system x [architecture, metric]) data with MultiIndex
    columns, keeping only systems present for every architecture, plus a
    (system x architecture) table of the winning file.

    Metrics without any data at all (e.g. t_stresses before the stress
    benchmarks are run) are reported and do not cause systems to be dropped.
    """
    metrics = list(MODE_METRICS[mode])
    print("Loading timings (fastest avg per system):")
    frames = []
    winners: dict[str, pd.Series] = {}
    for label, timings_dir in TIMINGS_DIRS:
        df, winner = load_best_timings(label, timings_dir, metrics)
        df.columns = pd.MultiIndex.from_product([[label], df.columns],
                                                names=("arch", "metric"))
        frames.append(df)
        winners[label] = winner
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
    winner_table = pd.DataFrame(winners).reindex(data.index)
    return data, winner_table


def mode_totals(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Bar values for `mode`: sum of its metrics (system x architecture).

    Metrics without any data (all NaN) are skipped in the sum, so e.g.
    scf_stresses equals the SCF times until stresses data is available.
    Unlike the shared mode_values, an architecture without any data at all
    for the metrics gets NaN (not 0) so the .dat file can report "-" and no
    winner is invented.
    """
    metric_level = data.columns.get_level_values("metric")
    components = data[data.columns[metric_level.isin(MODE_METRICS[mode])]]
    archs = data.columns.get_level_values("arch").unique()
    return pd.DataFrame({arch: components[arch].sum(axis=1, min_count=1)
                         for arch in archs})


def write_dat(path: Path, mode: str, winners: pd.DataFrame,
              totals: pd.DataFrame) -> None:
    """Write a whitespace-aligned table listing, per architecture and system,
    the fastest configuration (file) and the total time it achieved.

    `winners` and `totals` are (system x architecture) tables; architectures
    without data (e.g. stresses before the stress benchmarks) get a "-".
    """
    metrics_label = " + ".join(MODE_METRICS[mode])
    lines = [
        "# fastest configuration per system and architecture",
        f"# mode: {mode} (total = sum of {metrics_label})",
        f"# {'architecture':<12} {'system':<24} {'fastest_file':<28} {'total_s':>10}",
    ]
    for arch in winners.columns:
        for system in winners.index:
            fastest = winners.at[system, arch]
            total = totals.at[system, arch]
            fastest_str = "-" if pd.isna(fastest) else str(fastest)
            total_str = "-" if pd.isna(total) else f"{total:.3f}"
            lines.append(f"{arch:<12} {system:<24} {fastest_str:<28} {total_str:>10}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Plotting ----------------------------------------------------------------


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

    metrics_label = " + ".join(METRIC_LABELS.get(m, m) for m in MODE_METRICS[mode])
    arch_list = " vs ".join(archs)
    ylabel = f"{metrics_label} (s)"
    title = f"{metrics_label} per system ({arch_list}, fastest configuration)"

    values = mode_totals(data, mode)
    label_values = speedup_vs_baseline(values)

    fig, ax = plt.subplots(figsize=(max(10, 0.75 * n_systems), 5.5))

    draw_grouped_bars(ax, values, archs, x, width, label_values)
    ax.legend(loc="upper left")

    ax.set_yscale(y_scale)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x, [SYSTEM_LABELS.get(system, system) for system in data.index])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # Shift x tick labels a tiny bit to the right (screen-space points).
    shift = mtransforms.ScaledTranslation(3 / 72, 0, fig.dpi_scale_trans)
    for tick_label in ax.get_xticklabels():
        tick_label.set_transform(tick_label.get_transform() + shift)
    ax.yaxis.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=tuple(MODE_METRICS), default=MODE,
                        help="timings to plot: SCF, forces, or their sum "
                             f"(default: {MODE})")
    parser.add_argument("--yscale", choices=("log", "linear"), default=Y_SCALE,
                        help=f"y-axis scaling (default: {Y_SCALE})")
    args = parser.parse_args()

    data, winners = collect_data(args.mode)
    n_arch = len(data.columns.get_level_values("arch").unique())
    print(f"\nPlotting best {args.mode} for {len(data)} systems x {n_arch} "
          f"architectures ({args.yscale} y-axis):\n{data.to_string()}\n")

    fig = plot_comparison(data, args.yscale, args.mode)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTPUT_DIR / f"{OUTPUT_NAME}_{args.mode}.pdf"
    out_dat = OUTPUT_DIR / f"{OUTPUT_NAME}_{args.mode}.dat"
    fig.savefig(out_pdf, bbox_inches="tight")
    write_dat(out_dat, args.mode, winners, mode_totals(data, args.mode))
    print(f"Saved {out_pdf}")
    print(f"Saved {out_dat}")


if __name__ == "__main__":
    main()
