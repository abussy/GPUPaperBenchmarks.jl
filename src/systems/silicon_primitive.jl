"""
    silicon_primitive(; Ecut=18, kgrid=(4, 4, 4), architecture=DFTK.CPU(),
                       tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                       compute_forces=false, compute_stresses=false,
                       callback=identity, kwargs...)

2-atom primitive silicon cell (diamond cubic). Returns `(; scfres, forces, stresses)`.
"""
function silicon_primitive(; Ecut=18, kgrid=(4, 4, 4), architecture=DFTK.CPU(),
                            tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                            compute_forces=false, compute_stresses=false,
                            callback=identity, kwargs...)
    system = bulk(:Si; cubic=false)
    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
