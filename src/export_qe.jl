# Quantum ESPRESSO input exporter.

function export_qe_system(name::String, system_dir::String; kwargs...)
    f = get_system_function(name)
    export_kwargs = copy(kwargs)
    get!(export_kwargs, :tol, 1e-4)
    get!(export_kwargs, :architecture, DFTK.CPU())
    result = f(; export_kwargs...)
    _write_qe_input(result.scfres, system_dir)
end

function export_qe_inputs(systems::Vector{String}, output_dir::String; kwargs...)
    mkpath(output_dir)
    for name in systems
        system_dir = joinpath(output_dir, name)
        export_qe_system(name, system_dir; kwargs...)
    end
end

function _write_qe_input(scfres, system_dir::String)
    basis = scfres.basis
    model = basis.model
    family = something(DFTK.pseudofamily(first(model.atoms)), default_pseudopotentials())

    lattice_bohr = model.lattice
    positions_frac = [lattice_bohr \ SVector{3, Float64}(p) for p in model.positions]

    unique_elements = unique(AtomsBase.element_symbol.(model.atoms))
    ntyp = length(unique_elements)

    pseudo_dir = joinpath(system_dir, "pseudo")
    mkpath(pseudo_dir)
    for elem in unique_elements
        src = family[Symbol(elem)]
        dst = joinpath(pseudo_dir, basename(src))
        if !ispath(dst)
            symlink(abspath(src), dst)
        end
    end

    magnetic = model.n_spin == 2
    nspin = magnetic ? 2 : 1
    temperature = model.temperature
    qe_smear = "gaussian"

    input_path = joinpath(system_dir, "pw.in")
    open(input_path, "w") do io
        println(io, "&CONTROL")
        println(io, "  calculation = 'scf'")
        println(io, "  prefix = '", splitdir(system_dir)[2], "'")
        println(io, "  pseudo_dir = './pseudo/'")
        println(io, "  outdir = './out/'")
        println(io, "  tprnfor = .true.")
        println(io, "  tstress = .true.")
        println(io, "  disk_io = 'none'")
        println(io, "/")
        println(io)
        println(io, "&SYSTEM")
        println(io, "  ibrav = 0")
        println(io, "  nat = ", length(model.atoms))
        println(io, "  ntyp = ", ntyp)
        println(io, "  ecutwfc = ", round(basis.Ecut; digits=4))
        println(io, "  ecutrho = ", round(4 * basis.Ecut; digits=4))
        println(io, "  occupations = 'smearing'")
        println(io, "  smearing = '", qe_smear, "'")
        println(io, "  degauss = ", temperature)
        println(io, "  nspin = ", nspin)
        if magnetic
            magmom = model.spin_polarization == :collinear ? 0.5 : 0.0
            println(io, "  starting_magnetization = ", magmom)
        end
        println(io, "/")
        println(io)
        println(io, "&ELECTRONS")
        println(io, "  conv_thr = 1.0d-8")
        println(io, "  mixing_beta = 0.3")
        println(io, "/")
        println(io)
        println(io, "ATOMIC_SPECIES")
        for elem in unique_elements
            idx = findfirst(a -> AtomsBase.element_symbol(a) == elem, model.atoms)
            at = model.atoms[idx]
            mass_amu = ustrip(u"u", AtomsBase.mass(at))
            pp_file = basename(family[Symbol(elem)])
            @printf(io, "  %s  %.4f  %s\n", string(elem), mass_amu, pp_file)
        end
        println(io)
        println(io, "ATOMIC_POSITIONS crystal")
        for (i, at) in enumerate(model.atoms)
            sym = AtomsBase.element_symbol(at)
            p = positions_frac[i]
            @printf(io, "  %s  %.10f  %.10f  %.10f\n", string(sym), p...)
        end
        println(io)
    println(io, "K_POINTS automatic")
    @printf(io, "  %s  0 0 0\n", join(basis.kgrid.kgrid_size, " "))
        println(io)
        println(io, "CELL_PARAMETERS bohr")
        for i in 1:3
            @printf(io, "  %.10f  %.10f  %.10f\n", lattice_bohr[:, i]...)
        end
    end
end
