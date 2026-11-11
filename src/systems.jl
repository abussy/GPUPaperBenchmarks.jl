# System builders for the DFTK paper benchmark set.
#
# Each file in src/systems/ defines a self-contained function with the same name
# as the file. The function takes benchmark variables as keyword arguments
# (Ecut, kgrid, architecture, ...) and returns a converged SCF result. These
# functions can be run independently or discovered automatically by the runner.

# Default shared settings.
default_pseudopotentials() = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
default_functional() = PBE()
default_smearing() = (temperature=1e-2, smearing=DFTK.Smearing.Gaussian())
default_tol() = 1e-8

"""
    default_scf_convergence(convergence::Symbol, tol)

Return a DFTK convergence object for the requested criterion:
`:density` (default), `:energy`, `:force`, or `:maxiter`.
For `:maxiter` the caller is expected to pass `tol=0.0`, so the criterion never
fires and the SCF stops only when the requested number of iterations is reached.
"""
function default_scf_convergence(convergence::Symbol, tol)
    convergence == :density ? ScfConvergenceDensity(tol) :
    convergence == :energy  ? ScfConvergenceEnergy(tol) :
    convergence == :force   ? ScfConvergenceForce(tol) :
    convergence == :maxiter ? ScfConvergenceDensity(tol) :
    error("Unknown convergence criterion: $convergence (choose :density, :energy, :force, :maxiter)")
end

"""
    _seed_rng!(architecture, seed::Integer)

Seed the random-number generator used by `architecture` with `seed`. CPU runs
use the default Julia RNG and do nothing; GPU backends are seeded via their
vendor-specific API so that SCF initial guesses are reproducible.
"""
function _seed_rng! end
_seed_rng!(::DFTK.CPU, ::Integer) = nothing

"""
    _run_scf(basis; tol, callback, seed=42, kwargs...)

Run a DFTK SCF using the benchmark default convergence criterion.
The caller can select `:density`, `:energy`, `:force`, or `:maxiter` via the
`convergence` keyword, or pass a custom `is_converged` object. A `maxiter`
keyword can be passed to cap the number of SCF iterations.

`seed` controls the RNG seed for the SCF initial density (default 42). Before
the SCF starts, the RNG for the basis architecture is seeded, and
`self_consistent_field` is called with the same `seed`.
"""
function _run_scf(basis; tol, callback, seed::Integer=42, kwargs...)
    _seed_rng!(basis.architecture, seed)
    kwargs_nt = NamedTuple(kwargs)
    convergence = get(kwargs_nt, :convergence, :density)
    is_converged = get(kwargs_nt, :is_converged, default_scf_convergence(convergence, tol))
    remaining = Base.structdiff(kwargs_nt, NamedTuple{(:convergence, :is_converged)})
    self_consistent_field(basis; tol, callback, is_converged, seed, remaining...)
end

"""
    list_systems() -> Vector{String}

Return the names of all benchmark systems, derived from the filenames in
`src/systems/`.
"""
function list_systems()
    dir = joinpath(@__DIR__, "systems")
    [splitext(file)[1] for file in sort(readdir(dir)) if endswith(file, ".jl")]
end

"""
    get_system_function(name::String)

Return the benchmark function defined for a system name.
"""
function get_system_function(name::String)
    sym = Symbol(name)
    isdefined(@__MODULE__, sym) || error("System '$name' not found. " *
        "Make sure src/systems/$name.jl defines a function '$name'.")
    return getfield(@__MODULE__, sym)
end

"""
    _flexible_system_from_fractional(lattice, elements, positions)

Build an `AtomsBase.FlexibleSystem` from a 3×3 lattice matrix in Bohr (columns
are lattice vectors), a vector of element symbols, and a vector of fractional
positions.
"""
function _flexible_system_from_fractional(lattice, elements, positions)
    pos_cart = [lattice * SVector{3, Float64}(p) for p in positions]
    # AtomsBase requires positions and cell vectors to carry length units.
    pos_cart_u = [p * u"bohr" for p in pos_cart]
    lattice_u = lattice * u"bohr"
    atoms = [AtomsBase.Atom(elements[i], pos_cart_u[i]) for i in 1:length(elements)]
    cell_vectors = tuple([lattice_u[:, i] for i in 1:3]...)
    AtomsBase.FlexibleSystem(atoms; cell_vectors, periodicity=(true, true, true))
end

"""
    _flexible_system_fractional(lattice_angstrom, elements, positions_frac)

Build an `AtomsBase.FlexibleSystem` from a 3×3 lattice matrix in Ångström
(columns are lattice vectors), a vector of element symbols, and fractional
positions.
"""
function _flexible_system_fractional(lattice_angstrom, elements, positions_frac)
    lattice_bohr = austrip.(lattice_angstrom * u"Å")
    _flexible_system_from_fractional(lattice_bohr, elements, positions_frac)
end

"""
    _supercell_fractional(lattice, elements, positions_frac, repeat)

Build a supercell from a primitive cell given as a 3×3 lattice matrix (columns
are lattice vectors, in Ångström), element symbols, and fractional positions.
The atom ordering matches ASE's `Atoms.repeat` (the first repeat index varies
slowest).
"""
function _supercell_fractional(lattice, elements, positions_frac,
                               repeat::NTuple{3,Int})
    n1, n2, n3 = repeat
    all(>=(1), (n1, n2, n3)) || error("repeat entries must be positive integers")
    super_lattice = lattice * Diagonal([float(n) for n in (n1, n2, n3)])
    positions = Vector{Vector{Float64}}()
    elements_super = Symbol[]
    for i in 0:(n1 - 1), j in 0:(n2 - 1), k in 0:(n3 - 1)
        offset = (i / n1, j / n2, k / n3)
        for (p, e) in zip(positions_frac, elements)
            push!(positions, p ./ (n1, n2, n3) .+ offset)
            push!(elements_super, e)
        end
    end
    _flexible_system_fractional(super_lattice, elements_super, positions)
end

# ------------------------------------------------------------------
# Generic structure helpers. System-specific geometry builders (monolayers,
# surface slabs) live in the corresponding files in `src/systems/`; each
# replicates the geometry previously generated with `ASE.build` (verified
# against the ASE output), so that no Python/conda environment is required.

# Auto-include all system definition files. Each file is expected to define a
# function with the same name as the file (without the .jl extension).
const _SYSTEMS_DIR = joinpath(@__DIR__, "systems")
for file in sort(readdir(_SYSTEMS_DIR))
    if endswith(file, ".jl")
        include(joinpath(_SYSTEMS_DIR, file))
    end
end
