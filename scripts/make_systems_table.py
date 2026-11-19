#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "matplotlib",  # imported transitively by plot_cpu_vs_gpu
# ]
# ///
"""Generate LaTeX and Markdown tables summarizing the benchmark systems.

Scans every timing CSV in benchmark_results/*/timings and collects, per
system, the number of atoms, the plane-wave cutoff, the k-point grid and the
FFT size. These are repeat- and architecture-independent; inconsistent values
across files are reported (and the first is used). The system names are the
same labels as the plot x axes (SYSTEM_LABELS from scripts/plot_cpu_vs_gpu.py).

The free-form notes column (structure, vacuum, magnetization, ...) comes from
the SYSTEM_NOTES dict below; edit it there. Systems without an entry are
marked with a dash (and a warning).

Run with:
    uv run scripts/make_systems_table.py

Output: plots/systems_table.tex (booktabs table) and plots/systems_table.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Shared with the plotting scripts: same labels, same output directory.
from plot_cpu_vs_gpu import OUTPUT_DIR, RESULTS_DIR, SYSTEM_LABELS

# --- Configuration -----------------------------------------------------------

OUTPUT_BASENAME = "systems_table"    # saved as plots/<OUTPUT_BASENAME>.{tex,md}

# Free-form notes per system: structure, vacuum, magnetization, ... Written as
# plain (unicode) text; the LaTeX writer converts ×, Å, μB and subscript
# digits to LaTeX markup automatically. Facts taken from src/systems/*.jl.
SYSTEM_NOTES = {
    "aluminium_primitive": "fcc bulk, metallic",
    "aluminium_supercell": "fcc bulk, metallic; 3×3×3 supercell",
    "copt_primitive": "ferromagnetic L1₀ (P4/mmm) bulk",
    "copt_supercell": "ferromagnetic L1₀ (P4/mmm) bulk; 2×2×2 supercell",
    "gaas_110_slab": "zincblende (110) slab, 10 Å vacuum per side",
    "gold_111_slab": "fcc (111) slab, 10 Å vacuum per side",
    "hbn_primitive": "hexagonal monolayer, 10 Å vacuum per side",
    "hbn_supercell": "hexagonal monolayer, 10 Å vacuum per side; 4×4 in-plane supercell",
    "iron_primitive": "bcc bulk, ferromagnetic (magmom 4 μB)",
    "iron_supercell": "bcc bulk, ferromagnetic (magmom 4 μB); 3×3×3 supercell",
    "mgo_primitive": "rock-salt bulk",
    "mgo_supercell": "rock-salt bulk; 2×2×2 supercell",
    "mos2_primitive": "2H monolayer, 10 Å vacuum per side",
    "mos2_supercell": "2H monolayer, 10 Å vacuum per side; 4×4 in-plane supercell",
    "silicon_primitive": "diamond-cubic bulk",
    "silicon_supercell": "diamond-cubic bulk; 2×2×2 supercell",
    "zro2_primitive": "fluorite bulk",
    "zro2_supercell": "fluorite bulk; 2×2×2 supercell",
}

# --- Data loading ------------------------------------------------------------

TABLE_COLUMNS = ("natoms", "Ecut", "kgrid", "fft_size")


def load_system_configs() -> pd.DataFrame:
    """One row per system with the TABLE_COLUMNS values from all CSVs.

    The configuration is repeat- and architecture-independent; differing
    values across files are reported (and the first is used).
    """
    files = sorted(RESULTS_DIR.glob("*/timings/*.csv"))
    if not files:
        raise FileNotFoundError(f"No timing CSVs found under {RESULTS_DIR}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    nunique = df.groupby("system")[list(TABLE_COLUMNS)].nunique()
    for system, row in nunique[nunique.gt(1).any(axis=1)].iterrows():
        differing = {c: sorted(df.loc[df["system"] == system, c].unique().tolist())
                     for c in TABLE_COLUMNS if row[c] > 1}
        print(f"WARNING: {system}: inconsistent configuration across files "
              f"{differing}; using the first")
    config = df.groupby("system")[list(TABLE_COLUMNS)].first().sort_index()

    print(f"Loaded configuration for {len(config)} systems from {len(files)} file(s)")
    return config


def build_rows(config: pd.DataFrame) -> list[dict]:
    """Assemble the table rows (label, Ecut, natoms, kgrid, fft_size, notes)."""
    rows = []
    for system, cfg in config.iterrows():
        if system not in SYSTEM_NOTES:
            print(f"WARNING: no notes for {system!r}; add an entry to SYSTEM_NOTES")
        ecut = float(cfg["Ecut"])
        rows.append({
            "label": SYSTEM_LABELS.get(system, system),
            "Ecut": str(int(ecut)) if ecut.is_integer() else str(ecut),
            "natoms": str(int(cfg["natoms"])),
            "kgrid": str(cfg["kgrid"]),
            "fft_size": str(cfg["fft_size"]),
            "notes": SYSTEM_NOTES.get(system, "-"),
        })
    return rows


# --- Output: LaTeX -----------------------------------------------------------

# unicode symbols used in SYSTEM_NOTES -> LaTeX markup
LATEX_REPLACEMENTS = {
    "×": r"$\times$",
    "Å": r"\AA{}",
    "μB": r"$\mu_{\mathrm{B}}$",
    "₀": r"$_0$",
}


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters outside of $...$ math segments and
    convert the unicode symbols of LATEX_REPLACEMENTS to LaTeX markup."""
    parts = text.split("$")
    for i in range(0, len(parts), 2):    # even parts are outside math
        for char, repl in (("\\", r"\textbackslash{}"), ("~", r"\textasciitilde{}"),
                           ("^", r"\textasciicircum{}"), ("&", r"\&"),
                           ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
            parts[i] = parts[i].replace(char, repl)
    text = "$".join(parts)
    for char, repl in LATEX_REPLACEMENTS.items():
        text = text.replace(char, repl)
    return text


def latex_grid(value: str) -> str:
    """'2x2x1' -> '$2\\times2\\times1$'."""
    return "$" + value.replace("x", r"\times") + "$"


def write_latex(path: Path, rows: list[dict]) -> None:
    """Write a booktabs table environment."""
    lines = [
        "% Generated by scripts/make_systems_table.py -- edit SYSTEM_NOTES",
        "% there rather than this file.",
        "% The notes column wraps; adjust its p{...} width in the tabular",
        "% preamble if needed (with \\usepackage{array}, a prefix",
        "% >{\\raggedright\\arraybackslash} gives left-aligned wrapping).",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Benchmark systems and their discretization.}",
        "\\label{tab:systems}",
        "\\begin{tabular}{lrrllp{0.32\\textwidth}}",
        "\\toprule",
        "System & $E_{\\mathrm{cut}}$ (Ha) & Atoms & $k$-grid & FFT size & Notes \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [
            latex_escape(row["label"]),
            row["Ecut"],
            row["natoms"],
            latex_grid(row["kgrid"]),
            latex_grid(row["fft_size"]),
            latex_escape(row["notes"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Output: Markdown --------------------------------------------------------


def write_markdown(path: Path, rows: list[dict]) -> None:
    """Write a GitHub-flavored Markdown table."""
    lines = [
        "# Benchmark systems",
        "",
        "| System | Ecut (Ha) | Atoms | k-grid | FFT size | Notes |",
        "|:-------|----------:|------:|:-------|:---------|:------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['Ecut']} | {row['natoms']} | "
            f"{row['kgrid']} | {row['fft_size']} | {row['notes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = load_system_configs()
    rows = build_rows(config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tex = OUTPUT_DIR / f"{OUTPUT_BASENAME}.tex"
    out_md = OUTPUT_DIR / f"{OUTPUT_BASENAME}.md"
    write_latex(out_tex, rows)
    write_markdown(out_md, rows)
    print(f"Saved {out_tex}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
