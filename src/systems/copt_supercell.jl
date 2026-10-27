"""
    copt_supercell(; Ecut=48, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                      tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                      lattice_constant_a_angstrom=3.83, lattice_constant_c_angstrom=3.71,
                      co_magmom=1.8, pt_magmom=0.0, repeat=(2, 2, 2),
                      compute_forces=false, compute_stresses=false,
                      callback=identity, kwargs...)

CoPt ferromagnetic supercell (L1₀ ordered tetragonal structure). Default is a
2×2×2 repetition of the 2-atom primitive cell (16 atoms). Returns
`(; scfres, forces, stresses)`.
"""
function copt_supercell(; Ecut=48, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                           tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                           lattice_constant_a_angstrom=3.83,
                           lattice_constant_c_angstrom=3.71,
                           co_magmom=1.8, pt_magmom=0.0, repeat=(2, 2, 2),
                           compute_forces=false, compute_stresses=false,
                           callback=identity, kwargs...)
    a = austrip(lattice_constant_a_angstrom * u"Å")
    c = austrip(lattice_constant_c_angstrom * u"Å")
    lattice = diagm([a, a, c])

    primitive_positions = [[0.0, 0.0, 0.0],
                           [0.5, 0.5, 0.5]]
    primitive_elements = [:Co, :Pt]
    primitive_moments = [co_magmom, pt_magmom]

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
