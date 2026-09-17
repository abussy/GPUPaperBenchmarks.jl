# GPUPaperBenchmarks.jl — DFTK paper benchmarks

Standalone benchmark suite for the DFTK GPU paper. After `using PaperBenchmarks`,
each benchmark system is available as a function (with the same name as the file
in `src/systems/`) and can be run independently. The runner automatically
discovers these files and collects raw timings.

## Scope

- **Systems:** bulk primitive cells, supercells, and surface slabs, with ≤150 atoms.
- **Functional:** PBE.
- **Pseudopotentials:** norm-conserving UPF (`dojo.nc.sr.pbe.v0_4_1.standard.upf`).
  Default kinetic-energy cutoffs (`Ecut`) are the highest recommended value for
  the elements present in each system, as provided by the pseudopotential
  library (see [PseudoDojo](https://www.pseudo-dojo.org/)).
- **Backends:** CPU, NVIDIA CUDA, AMD ROCm.
- **Output:** CSV with raw SCF, forces, and stresses timings.
- **Comparison:** Quantum ESPRESSO input files can be exported from the same
  structures and parameters.

## Pseudopotential cutoffs

The default kinetic-energy cutoff for each system is set to the highest
[PseudoDojo](https://www.pseudo-dojo.org/) recommended `Ecut` among the elements
present in that system. For the `dojo.nc.sr.pbe.v0_4_1.standard.upf` pseudopotential
set used here, the recommended values are listed in the
[`nc-sr-04_pbe_standard.json`](https://github.com/abinit/pseudo_dojo/blob/master/website/nc-sr-04_pbe_standard.json)
table (GitHub mirror of the PseudoDojo website data). The relevant entry for each
element is `"hn"` — the **hints normal** value, which corresponds to the
**standard** accuracy set.

For example, the JSON entry for titanium is:

```json
"Ti": { "hl": 38.0, "hn": 42.0, "hh": 46.0, ... }
```

so the standard recommended cutoff is `42.0` Ha.

The same values can be queried programmatically:

```julia
using PseudoPotentialData
family = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
recommended_cutoff(family, :Ti)  # (Ecut = 42.0, ...)
```

All cutoffs can be overridden per run via the `Ecut` keyword argument.

## Setup

```bash
cd GPUPaperBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.develop(path="/path/to/DFTK.jl"); Pkg.instantiate()'
```

For GPU runs, add the relevant package:

```bash
# NVIDIA
julia --project=. -e 'using Pkg; Pkg.add("CUDA")'

# AMD
julia --project=. -e 'using Pkg; Pkg.add("AMDGPU")'
```

## Running a single system independently

Each file in `src/systems/` defines a function with the same name as the file.
It accepts benchmark variables as keyword arguments (`Ecut`, `kgrid`,
`architecture`, `tol`, ...) and returns a named tuple `(; scfres, forces, stresses)`.

```julia
using PaperBenchmarks

# CPU, default parameters from the system file (forces/stresses not computed)
result = silicon_supercell()
scfres = result.scfres

# Primitive cell of the same material
result = silicon_primitive()

# SrVO₃ perovskite
result = srvo3()

# SrVO₃ supercell (135 atoms by default)
result = srvo3_supercell()

# Custom parameters
result = silicon_supercell(; Ecut=40, kgrid=(2, 2, 2))

# Compute forces and stresses as well
result = silicon_supercell(; compute_forces=true, compute_stresses=true)
forces = result.forces
stresses = result.stresses

# Show SCF progress log
result = silicon_supercell(; callback=DFTK.ScfDefaultCallback())

# NVIDIA GPU
using CUDA
result = silicon_supercell(; architecture=DFTK.GPU(CuArray))

# AMD GPU
using AMDGPU
result = silicon_supercell(; architecture=DFTK.GPU(ROCArray))
```

## Running the full benchmark suite

Run all systems with their default parameters:

```bash
julia --project=. scripts/run_benchmarks.jl
```

Run a subset of systems:

```bash
julia --project=. scripts/run_benchmarks.jl silicon_supercell,diamond_supercell
```

Override benchmark variables:

```bash
julia --project=. scripts/run_benchmarks.jl --Ecut=40 --kgrid=2,2,2 --architecture=CUDA

julia --project=. scripts/run_benchmarks.jl --Ecut=40 --kgrid=2,2,2 --architecture=AMDGPU
```

Run each system multiple times (default is 5) and report individual and average
timings:

```bash
julia --project=. scripts/run_benchmarks.jl --nrepeats=3
```

By default the runner performs one non-recorded warm-up SCF + forces + stresses
run before the timed repeats, so that GPU/CPU JIT compilation time is not
included in the reported averages. Disable it with:

```bash
julia --project=. scripts/run_benchmarks.jl --warmup=false
```

### SCF convergence criterion

The default SCF convergence criterion is density-based (`:density`). You can
switch to energy- or force-based convergence with the `--convergence` flag:

```bash
julia --project=. scripts/run_benchmarks.jl --convergence=energy --tol=1e-10
```

Allowed values are `density` (default), `energy`, and `force`. The same keyword
works when calling a system function directly:

```julia
result = silicon_primitive(; convergence=:energy, tol=1e-10)
```

The runner always measures SCF, forces and stresses as separate timed steps.
The `compute_forces` / `compute_stresses` flags in the individual system files
are intended for independent use; the runner does not rely on them.

Results are written to `results/timings_YYYYmmdd_HHMMSS.csv`. Rows are appended
after each system finishes, so partial results are preserved if the run is
interrupted.

## Running with MPI

A separate MPI-aware runner is provided in `scripts/run_benchmarks_mpi.jl`.
DFTK will distribute the workload across MPI ranks (typically over k-points).
All progress logging and the CSV output are emitted only by the master rank.
If a system fails on any rank, the run aborts immediately.

```bash
mpiexec -n 4 julia --project=. scripts/run_benchmarks_mpi.jl
```

Run a subset of systems or override parameters exactly like the serial runner:

```bash
mpiexec -n 4 julia --project=. scripts/run_benchmarks_mpi.jl silicon_primitive,diamond --architecture=CPU --nrepeats=3
```

As in the serial runner, rows are appended to the CSV after each system
completes, so partial results are preserved if the run aborts.

## Adding a new system

1. Create `src/systems/<my_system>.jl` defining a function `<my_system>(; kwargs...)`.
2. The function should build the system, run the SCF, and return `scfres`.
3. Re-run `scripts/run_benchmarks.jl`; the new system is picked up automatically.

Example:

```julia
function my_system(; Ecut=30, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                     tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                     compute_forces=false, compute_stresses=false,
                     callback=identity, kwargs...)
    system = bulk(:Si; cubic=true) * (2, 2, 2)
    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
```

## Exporting Quantum ESPRESSO inputs

Export `pw.in` files for all systems:

```bash
julia --project=. scripts/export_qe.jl
```

Export only selected systems:

```bash
julia --project=. scripts/export_qe.jl silicon_supercell,diamond_supercell
```

This creates one subdirectory per system with the input file and symlinks to
the required UPF pseudopotentials.

## Benchmark outputs

The CSV contains one row per repeat plus one extra row per system with
`repeat=avg` that reports the mean timing across all repeats:

- `system`: benchmark name
- `repeat`: repeat index (`1` to `N`) or `avg`
- `natoms`: number of atoms
- `nelectrons`: number of electrons
- `Ecut`: kinetic energy cutoff (Hartree)
- `kgrid`: k-point grid
- `architecture`: backend used
- `t_scf`: SCF wall time (s)
- `t_forces`: forces wall time (s)
- `t_stresses`: stresses wall time (s)
- `energy`: total energy (Hartree)
- `n_scfiter`: number of SCF iterations
- `fft_size`: real-space FFT grid size
