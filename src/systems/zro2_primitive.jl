"""
    zro2_primitive(; Ecut=42, kgrid=(4, 4, 4), architecture=DFTK.CPU(),
                      tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                      lattice_constant_angstrom=5.08,
                      compute_forces=false, compute_stresses=false,
                      callback=identity, kwargs...)

3-atom primitive ZrO₂ cell (fluorite). Returns `(; scfres, forces, stresses)`.
"""
function zro2_primitive(; Ecut=42, kgrid=(4, 4, 4), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          lattice_constant_angstrom=5.08,
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    a = austrip(lattice_constant_angstrom * u"Å")
    # Primitive FCC lattice vectors for fluorite.
    lattice = a / 2 * [0.0 1.0 1.0;
                       1.0 0.0 1.0;
                       1.0 1.0 0.0]

    positions = [[0.0, 0.0, 0.0],
                 [0.25, 0.25, 0.25],
                 [0.75, 0.75, 0.75]]
    elements = [:Zr, :O, :O]
    system = _flexible_system_from_fractional(lattice, elements, positions)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
