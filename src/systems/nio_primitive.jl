"""
    nio_primitive(; Ecut=42, kgrid=(8, 8, 8), architecture=DFTK.CPU(),
                     tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                     lattice_constant_angstrom=4.17, magmom=1.7,
                     compute_forces=false, compute_stresses=false,
                     callback=identity, kwargs...)

4-atom antiferromagnetic primitive NiO cell (rock-salt, AFM-II ordering).
The two Ni atoms carry opposite magnetic moments. Returns
`(; scfres, forces, stresses)`.
"""
function nio_primitive(; Ecut=42, kgrid=(8, 8, 8), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          lattice_constant_angstrom=4.17, magmom=1.7,
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    a = austrip(lattice_constant_angstrom * u"Å")
    # Primitive rhombohedral cell doubled along [111] for AFM-II ordering.
    lattice = a * [0.0 0.5 1.0;
                   0.5 0.0 1.0;
                   0.5 0.5 1.0]

    positions = [[0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.5],
                 [0.25, 0.25, 0.25],
                 [0.75, 0.75, 0.75]]
    elements = [:Ni, :Ni, :O, :O]
    magnetic_moments = [magmom, -magmom, 0.0, 0.0]
    system = _flexible_system_from_fractional(lattice, elements, positions)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    ρ0 = guess_density(basis, magnetic_moments)
    scfres = _run_scf(basis; tol, callback, ρ=ρ0, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
