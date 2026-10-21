"""
    nio_supercell(; Ecut=42, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                     tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                     lattice_constant_angstrom=4.17, magmom=1.7, repeat=(2, 2, 2),
                     compute_forces=false, compute_stresses=false,
                     callback=identity, kwargs...)

NiO antiferromagnetic supercell (rock-salt, AFM-II ordering). Default is a
2×2×2 repetition of the 4-atom primitive cell (32 atoms). Returns
`(; scfres, forces, stresses)`.
"""
function nio_supercell(; Ecut=42, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          lattice_constant_angstrom=4.17, magmom=1.7, repeat=(2, 2, 2),
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    a = austrip(lattice_constant_angstrom * u"Å")
    lattice = a * [0.0 0.5 1.0;
                   0.5 0.0 1.0;
                   0.5 0.5 1.0]

    primitive_positions = [[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.5],
                           [0.25, 0.25, 0.25],
                           [0.75, 0.75, 0.75]]
    primitive_elements = [:Ni, :Ni, :O, :O]
    primitive_moments = [magmom, -magmom, 0.0, 0.0]

    n1, n2, n3 = repeat
    scale = diagm([Float64(n1), Float64(n2), Float64(n3)])
    super_lattice = lattice * scale

    positions = Vector{Vector{Float64}}()
    elements = Symbol[]
    magnetic_moments = Float64[]
    for i in 0:(n1 - 1), j in 0:(n2 - 1), k in 0:(n3 - 1)
        offset = [i, j, k]
        for idx in 1:length(primitive_positions)
            f = primitive_positions[idx]
            push!(positions, (f .+ offset) ./ repeat)
            push!(elements, primitive_elements[idx])
            push!(magnetic_moments, primitive_moments[idx])
        end
    end

    system = _flexible_system_from_fractional(super_lattice, elements, positions)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()..., magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    ρ0 = guess_density(basis, magnetic_moments)
    scfres = _run_scf(basis; tol, callback, ρ=ρ0, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
