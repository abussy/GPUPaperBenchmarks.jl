#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
# ]
# ///
"""Cost of one stress calculation expressed in numbers of SCF steps.

From the single-core / single-GPU timing results (1 rank, 1 thread) this
script reports, for each benchmark system and architecture, how expensive
one stress computation is relative to one SCF iteration:

    stress_cost_in_scf_steps = t_stresses / (t_scf / n_scfiter)

i.e. the stress time divided by the mean time of a single SCF step. The
ratio is computed for every recorded repeat individually and then averaged,
so it is insensitive to differing SCF iteration counts between repeats.

Timings are read from the CSVs written by scripts/run_benchmarks.jl:

  * CPU:   benchmark_results/CPU/timings/mod*_1rank_1thread.csv
           (each modN file covers a subset of the benchmark systems)
  * CUDA:  benchmark_results/CUDA/timings/GPU_1rank_1thread.csv
  * AMDGPU: benchmark_results/AMDGPU/timings/GPU_1rank_1thread.csv

Systems whose stresses were not computed (t_stresses is NaN) are reported
as NaN and excluded from the per-architecture summary.

Run with:
    uv run scripts/stress_cost_in_scf_steps.py

Output: a table on stdout and plots/stress_cost_in_scf_steps.dat with the
same numbers (one column per architecture, plus the SCF step counts).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# --- Configuration -----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark_results"
OUTPUT_DIR = REPO_ROOT / "plots"

OUTPUT_NAME = "stress_cost_in_scf_steps"   # saved as plots/<OUTPUT_NAME>.dat

# One entry per architecture, in column order. Globs are relative to RESULTS_DIR.
# CPU single-core runs are split across the modN files.
ARCHITECTURES = [
    ("CPU", "CPU/timings/mod*_1rank_1thread.csv"),
    ("CUDA", "CUDA/timings/GPU_1rank_1thread.csv"),
    ("AMDGPU", "AMDGPU/timings/GPU_1rank_1thread.csv"),
]


# --- Data loading ------------------------------------------------------------


def load_stress_cost(rel_glob: str) -> pd.DataFrame:
    """Stress cost in SCF steps per system for one architecture.

    Every repeat row of every matching CSV contributes one ratio
    t_stresses * n_scfiter / t_scf; the per-system mean over all repeats
    (and re-runs) is returned, together with the mean SCF step count.
    """
    files = sorted(glob.glob(str(RESULTS_DIR / rel_glob)))
    if not files:
        raise FileNotFoundError(f"No timing files found: {RESULTS_DIR / rel_glob}")

    ratios: dict[str, list[float]] = {}
    steps: dict[str, list[float]] = {}
    for fname in files:
        df = pd.read_csv(fname)
        needed = ("repeat", "system", "t_scf", "t_stresses", "n_scfiter")
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"WARNING: skipping {Path(fname).name}: missing columns {missing}")
            continue
        # The precomputed `avg` row is itself a mean; it is excluded so that
        # every recorded repeat contributes exactly once.
        df = df[df["repeat"].astype(str) != "avg"]
        for system, t_scf, t_stresses, n_iter in zip(
                df["system"], df["t_scf"], df["t_stresses"], df["n_scfiter"]):
            if not (np.isfinite(t_scf) and np.isfinite(n_iter) and n_iter > 0):
                continue
            if not np.isfinite(t_stresses):
                ratios.setdefault(system, []).append(np.nan)
                continue
            ratios.setdefault(system, []).append(t_stresses * n_iter / t_scf)
            steps.setdefault(system, []).append(float(n_iter))

    if not ratios:
        raise FileNotFoundError(f"No usable rows for {RESULTS_DIR / rel_glob}")

    table = pd.DataFrame({
        "stress_cost_scf_steps": {s: np.nanmean(v) for s, v in ratios.items()},
        "n_scfiter": {s: np.nanmean(steps.get(s, [np.nan])) for s in ratios},
    })
    n_missing = table["stress_cost_scf_steps"].isna().sum()
    if n_missing:
        print(f"WARNING: {len(files)} file(s): {n_missing} system(s) without "
              f"stress timings (t_stresses is NaN)")
    return table


def collect_data() -> pd.DataFrame:
    """Return a (system x architecture) table of stress costs in SCF steps."""
    frames = []
    for label, rel_glob in ARCHITECTURES:
        frames.append(load_stress_cost(rel_glob)
                      .rename_axis("system")
                      .add_prefix(f"{label}|"))
    data = pd.concat(frames, axis=1).sort_index()

    cost_cols = [c for c in data.columns if c.endswith("|stress_cost_scf_steps")]
    missing = data[cost_cols].isna().all(axis=1)
    if missing.any():
        print(f"WARNING: dropping systems without stress data in any "
              f"architecture: {list(data.index[missing])}")
        data = data[~missing]
    return data


# --- Output ------------------------------------------------------------------


def print_table(data: pd.DataFrame) -> None:
    """Pretty-print the (system x architecture) stress-cost table."""
    archs = [label for label, _ in ARCHITECTURES]
    cost_cols = [f"{a}|stress_cost_scf_steps" for a in archs]
    step_cols = [f"{a}|n_scfiter" for a in archs]

    header = f"{'system':<22}" + "".join(f"{a + ' (steps)':>16}" for a in archs)
    print(f"\nCost of one stress calculation, in units of SCF steps "
          f"(1 rank / 1 thread):\n\n{header}")
    print("-" * len(header))
    for system, row in data.iterrows():
        cells = ""
        for col in cost_cols:
            value = row[col]
            cells += f"{'-':>16}" if pd.isna(value) else f"{value:>16.2f}"
        print(f"{system:<22}{cells}")

    print("-" * len(header))
    means = [data[c].mean() for c in cost_cols]
    print(f"{'mean over systems':<22}" + "".join(f"{m:>16.2f}" for m in means))

    print(f"\n(SCF iteration counts per architecture: "
          + ", ".join(f"{a} {data[c].min():.0f}-{data[c].max():.0f}"
                      for a, c in zip(archs, step_cols)) + ")")


def write_dat(path: Path, data: pd.DataFrame) -> None:
    """Write the same table as a whitespace-aligned .dat file."""
    archs = [label for label, _ in ARCHITECTURES]
    cost_cols = [f"{a}|stress_cost_scf_steps" for a in archs]
    step_cols = [f"{a}|n_scfiter" for a in archs]

    lines = [
        "# cost of one stress calculation in units of SCF steps (1 rank / 1 thread)",
        "# ratio = t_stresses / (t_scf / n_scfiter), averaged over repeats",
        f"# {'system':<22}"
        + "".join(f"{a + '_steps':>16}" for a in archs)
        + "".join(f"{a + '_nscf':>12}" for a in archs),
    ]
    for system, row in data.iterrows():
        costs = "".join(f"{'-':>16}" if pd.isna(row[c]) else f"{row[c]:>16.2f}"
                        for c in cost_cols)
        steps = "".join(f"{'-':>12}" if pd.isna(row[c]) else f"{row[c]:>12.0f}"
                        for c in step_cols)
        lines.append(f"{system:<22}{costs}{steps}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = collect_data()
    print_table(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dat = OUTPUT_DIR / f"{OUTPUT_NAME}.dat"
    write_dat(out_dat, data)
    print(f"\nSaved {out_dat}")


if __name__ == "__main__":
    main()
