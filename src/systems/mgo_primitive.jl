"""
    mgo_primitive(; Ecut=42, kgrid=(4, 4, 4), architecture=DFTK.CPU(),
                     tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                     lattice_constant_angstrom=4.21,
                     compute_forces=false, compute_stresses=false,
                     kwargs...)

2-atom primitive MgO cell (rock-salt). Returns `(; scfres, forces, stresses)`.
"""
function mgo_primitive(; Ecut=42, kgrid=(4, 4, 4), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          lattice_constant_angstrom=4.21,
                          compute_forces=false, compute_stresses=false,
                          kwargs...)
    a = austrip(lattice_constant_angstrom * u"Å")
    # Primitive FCC lattice vectors for rock-salt.
    lattice = a / 2 * [0.0 1.0 1.0;
                       1.0 0.0 1.0;
                       1.0 1.0 0.0]

    positions = [[0.0, 0.0, 0.0],
                 [0.5, 0.5, 0.5]]
    elements = [:Mg, :O]
    system = _flexible_system_from_fractional(lattice, elements, positions)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = self_consistent_field(basis; tol, callback=identity, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
