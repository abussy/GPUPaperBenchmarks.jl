"""
    aluminium_primitive(; Ecut=20, kgrid=(8, 8, 8), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)

1-atom primitive aluminium cell (fcc). Returns `(; scfres, forces, stresses)`.
"""
function aluminium_primitive(; Ecut=20, kgrid=(8, 8, 8), architecture=DFTK.CPU(),
                              tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                              compute_forces=false, compute_stresses=false,
                              callback=identity, kwargs...)
    system = bulk(:Al; cubic=false)
    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
