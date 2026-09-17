module PaperBenchmarks

using DFTK
using AtomsBase
using AtomsBuilder
using PseudoPotentialData
using ASEconvert
using CSV
using TOML
using Dates
using LinearAlgebra
using Printf
using Random
using StaticArrays
using Statistics
using Unitful
using UnitfulAtomic

include("systems.jl")
include("timing.jl")
include("export_qe.jl")

# Re-export names used by the individual system builders so that
# `using PaperBenchmarks` is enough to run/tweak systems interactively.
using AtomsBuilder: bulk, rattle!
using ASEconvert: AbstractSystem, pyconvert
using UnitfulAtomic: austrip
using LinearAlgebra: diagm
using PseudoPotentialData: PseudoFamily

export DFTK, bulk, rattle!, AbstractSystem, pyconvert, austrip, diagm, @u_str, PseudoFamily

# Auto-export each benchmark system function so users can call e.g. silicon_primitive().
for sys in list_systems()
    sym = Symbol(sys)
    if isdefined(@__MODULE__, sym)
        @eval export $sym
    end
end

export list_systems, get_system_function, benchmark_system, select_architecture,
       write_results_csv, append_results_csv, export_qe_inputs, export_qe_system,
       default_pseudopotentials, default_functional, default_smearing, default_tol

end
