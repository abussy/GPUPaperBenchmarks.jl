# Quantum ESPRESSO input exporter.

struct _QEBasisOnlySolver end

# DFTK unconditionally reads these solver-state fields when assembling the
# final SCF result, even if no fixed-point iteration was performed.
function (::_QEBasisOnlySolver)(fixpoint_map, x0, info; kwargs...)
    return x0, merge(info, (; n_bands_converge=0, diagonalization=[]))
end

"""
    export_qe_system(name, system_dir; mode=:converged, kwargs...)

Write a Quantum ESPRESSO input for one benchmark system. `mode=:converged`
writes `pw_comp.in` for a fully converged SCF and `mode=:timing` writes
`pw_perf.in` for exactly 10 SCF iterations followed by forces.

The exporter constructs the model and the exact benchmark plane-wave basis
without performing an eigensolve; it does not run a converged DFTK calculation.
The required UPF files are copied into a local `pseudo/` directory so that the
system directory is self-contained. Remaining keyword arguments are forwarded to
the system builder and therefore allow the same overrides as a DFTK benchmark
run.
"""
function export_qe_system(name::String, system_dir::String;
                          mode::Symbol=:converged, magnetic_moments=nothing, kwargs...)
    mode in (:converged, :timing) ||
        throw(ArgumentError("mode must be :converged or :timing, got :$mode"))
    f = get_system_function(name)
    export_kwargs = Dict{Symbol, Any}(kwargs)
    export_kwargs[:architecture] = DFTK.CPU()
    export_kwargs[:compute_forces] = false
    export_kwargs[:compute_stresses] = false
    export_kwargs[:tol] = 0.0
    export_kwargs[:convergence] = :maxiter
    # Avoid an eigensolve, which can exhaust memory for the larger supercells.
    # The builder still constructs the model and the exact benchmark basis.
    export_kwargs[:maxiter] = 0
    export_kwargs[:solver] = _QEBasisOnlySolver()
    result = f(; export_kwargs...)
    moments = _qe_initial_magnetic_moments(name, result.scfres.basis.model,
                                           export_kwargs, magnetic_moments)
    return _write_qe_input(result.scfres, system_dir; mode, magnetic_moments=moments)
end

function _qe_initial_magnetic_moments(name, model, kwargs, override)
    model.spin_polarization == :collinear || return nothing
    isnothing(override) || return override

    if name in ("iron_primitive", "iron_supercell")
        return fill(get(kwargs, :magmom, 4.0), length(model.atoms))
    elseif name in ("copt_primitive", "copt_supercell")
        co_magmom = get(kwargs, :co_magmom, 1.8)
        pt_magmom = get(kwargs, :pt_magmom, 0.0)
        return [AtomsBase.element_symbol(atom) == :Co ? co_magmom : pt_magmom
                for atom in model.atoms]
    end
    throw(ArgumentError(
        "No default initial magnetic moments are known for $name; pass magnetic_moments"))
end

function export_qe_inputs(systems::Vector{String}, output_dir::String; kwargs...)
    mkpath(output_dir)
    return [export_qe_system(name, joinpath(output_dir, name); kwargs...)
            for name in systems]
end

function _qe_smearing(model)
    smearing = model.smearing
    smearing isa DFTK.Smearing.Gaussian && return "gaussian"
    smearing isa DFTK.Smearing.FermiDirac && return "fermi-dirac"
    smearing isa DFTK.Smearing.MarzariVanderbilt && return "marzari-vanderbilt"
    if smearing isa DFTK.Smearing.MethfesselPaxton
        smearing.order == 0 && return "gaussian"
        smearing.order == 1 && return "methfessel-paxton"
    end
    error("Unsupported QE equivalent for DFTK smearing $(typeof(smearing))")
end

_qe_input_filename(mode::Symbol) =
    mode == :converged ? "pw_comp.in" : "pw_perf.in"

function _qe_kshift(kgrid)
    return map(kgrid.kshift) do shift
        shift == 0 && return 0
        shift == 1 // 2 && return 1
        error("QE only supports DFTK k-grid shifts of 0 or 1/2, got $shift")
    end
end

function _write_qe_input(scfres, system_dir::String; mode::Symbol=:converged,
                         magnetic_moments=nothing)
    mode in (:converged, :timing) ||
        throw(ArgumentError("mode must be :converged or :timing, got :$mode"))
    basis = scfres.basis
    model = basis.model
    basis.kgrid isa DFTK.MonkhorstPack ||
        error("QE automatic k-point export requires a DFTK MonkhorstPack grid")
    family = something(DFTK.pseudofamily(first(model.atoms)), default_pseudopotentials())

    # DFTK stores lattice vectors in Bohr and positions in fractional coordinates.
    lattice_bohr = model.lattice
    positions_frac = model.positions

    unique_elements = unique(AtomsBase.element_symbol.(model.atoms))
    ntyp = length(unique_elements)

    magnetic = model.spin_polarization == :collinear
    if magnetic && (isnothing(magnetic_moments) ||
                    length(magnetic_moments) != length(model.atoms))
        error("Per-atom magnetic moments are required to export a collinear QE input")
    end

    # QE interprets |starting_magnetization| >= 1 as a site magnetic moment in
    # mu_B, matching the DFTK initial magnetic moments.
    starting_magnetization = zeros(Float64, ntyp)
    for (itype, elem) in enumerate(unique_elements)
        indices = findall(a -> AtomsBase.element_symbol(a) == elem, model.atoms)
        magnetic || continue
        moments = [m isa Number ? m : m[3] for m in magnetic_moments[indices]]
        all(==(first(moments)), moments) || error(
            "Different initial moments for element $elem require separate QE species")
        starting_magnetization[itype] = first(moments)
    end

    pseudo_dir = joinpath(system_dir, "pseudo")
    mkpath(pseudo_dir)
    for elem in unique_elements
        src = family[Symbol(elem)]
        dst = joinpath(pseudo_dir, basename(src))
        # Replace old symlinks as well as stale pseudopotential copies.
        if ispath(dst) || islink(dst)
            rm(dst; force=true)
        end
        cp(src, dst)
    end

    input_path = joinpath(system_dir, _qe_input_filename(mode))
    open(input_path, "w") do io
        println(io, "! Generated from the corresponding DFTK benchmark setup.")
        println(io, "! DFTK uses Hartree; QE cutoffs and smearing below use Rydberg.")
        println(io, "&CONTROL")
        println(io, "  calculation = 'scf'")
        println(io, "  prefix = '", basename(abspath(system_dir)), "'")
        println(io, "  pseudo_dir = './pseudo/'")
        println(io, "  outdir = './out/'")
        println(io, "  tprnfor = ", mode == :timing ? ".true." : ".false.")
        println(io, "  tstress = .false.")
        println(io, "  disk_io = 'none'")
        println(io, "/")
        println(io)
        println(io, "&SYSTEM")
        println(io, "  ibrav = 0")
        println(io, "  nat = ", length(model.atoms))
        println(io, "  ntyp = ", ntyp)
        @printf(io, "  ecutwfc = %.10f\n", 2 * basis.Ecut)
        @printf(io, "  ecutrho = %.10f\n", 8 * basis.Ecut)
        @printf(io, "  nr1 = %d\n  nr2 = %d\n  nr3 = %d\n", basis.fft_size...)
        println(io, "  input_dft = 'PBE'")
        if iszero(model.temperature)
            println(io, "  occupations = 'fixed'")
        else
            println(io, "  occupations = 'smearing'")
            println(io, "  smearing = '", _qe_smearing(model), "'")
            @printf(io, "  degauss = %.12e\n", 2 * model.temperature)
        end
        println(io, "  nspin = ", magnetic ? 2 : 1)
        if magnetic
            for (itype, magnetization) in enumerate(starting_magnetization)
                @printf(io, "  starting_magnetization(%d) = %.12f\n",
                        itype, magnetization)
            end
        end
        println(io, "/")
        println(io)
        println(io, "&ELECTRONS")
        if mode == :timing
            # Prevent an early exit and continue to forces after ten iterations.
            println(io, "  electron_maxstep = 10")
            println(io, "  conv_thr = 0.0d0")
            println(io, "  scf_must_converge = .false.")
        else
            println(io, "  conv_thr = 2.0d-10")
            println(io, "  scf_must_converge = .true.")
        end
        println(io, "/")
        println(io)
        println(io, "ATOMIC_SPECIES")
        for elem in unique_elements
            idx = findfirst(a -> AtomsBase.element_symbol(a) == elem, model.atoms)
            atom = model.atoms[idx]
            mass_amu = ustrip(u"u", AtomsBase.mass(atom))
            pp_file = basename(family[Symbol(elem)])
            @printf(io, "  %s  %.8f  %s\n", string(elem), mass_amu, pp_file)
        end
        println(io)
        println(io, "ATOMIC_POSITIONS crystal")
        for (atom, position) in zip(model.atoms, positions_frac)
            sym = AtomsBase.element_symbol(atom)
            @printf(io, "  %s  %.16f  %.16f  %.16f\n", string(sym), position...)
        end
        println(io)
        println(io, "K_POINTS automatic")
        kshift = _qe_kshift(basis.kgrid)
        @printf(io, "  %d %d %d  %d %d %d\n", basis.kgrid.kgrid_size..., kshift...)
        println(io)
        println(io, "CELL_PARAMETERS bohr")
        for i in 1:3
            @printf(io, "  %.16f  %.16f  %.16f\n", lattice_bohr[:, i]...)
        end
    end
    return input_path
end
