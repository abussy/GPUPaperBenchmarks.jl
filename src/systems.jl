# System builders for the DFTK paper benchmark set.
#
# Each file in src/systems/ defines a self-contained function with the same name
# as the file. The function takes benchmark variables as keyword arguments
# (Ecut, kgrid, architecture, ...) and returns a converged SCF result. These
# functions can be run independently or discovered automatically by the runner.

# Default shared settings.
default_pseudopotentials() = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
default_functional() = PBE()
default_smearing() = (temperature=1e-3, smearing=DFTK.Smearing.Gaussian())
default_tol() = 1e-8

"""
    default_scf_convergence(convergence::Symbol, tol)

Return a DFTK convergence object for the requested criterion:
`:density` (default), `:energy`, or `:force`.
"""
function default_scf_convergence(convergence::Symbol, tol)
    convergence == :density ? ScfConvergenceDensity(tol) :
    convergence == :energy  ? ScfConvergenceEnergy(tol) :
    convergence == :force   ? ScfConvergenceForce(tol) :
    error("Unknown convergence criterion: $convergence (choose :density, :energy, :force)")
end

"""
    _run_scf(basis; tol, callback, kwargs...)

Run a DFTK SCF using the benchmark default convergence criterion.
The caller can select `:density`, `:energy`, or `:force` via the `convergence`
keyword, or pass a custom `is_converged` object.
"""
function _run_scf(basis; tol, callback, kwargs...)
    kwargs_nt = NamedTuple(kwargs)
    convergence = get(kwargs_nt, :convergence, :density)
    is_converged = get(kwargs_nt, :is_converged, default_scf_convergence(convergence, tol))
    remaining = Base.structdiff(kwargs_nt, NamedTuple{(:convergence, :is_converged)})
    self_consistent_field(basis; tol, callback, is_converged, remaining...)
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

# Auto-include all system definition files. Each file is expected to define a
# function with the same name as the file (without the .jl extension).
const _SYSTEMS_DIR = joinpath(@__DIR__, "systems")
for file in sort(readdir(_SYSTEMS_DIR))
    if endswith(file, ".jl")
        include(joinpath(_SYSTEMS_DIR, file))
    end
end
