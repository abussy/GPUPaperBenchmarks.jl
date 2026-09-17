"""
    aluminium_rattled_108(; Ecut=20, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                             tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                             rattle_amplitude=0.03,
                             compute_forces=false, compute_stresses=false,
                             callback=identity, kwargs...)

108-atom aluminium supercell with a small random rattle. Returns `(; scfres, forces, stresses)`.
"""
function aluminium_rattled_108(; Ecut=20, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                                  tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                                  rattle_amplitude=0.03,
                                  compute_forces=false, compute_stresses=false,
                                  callback=identity, kwargs...)
    system = bulk(:Al; cubic=true) * (3, 3, 3)
    rattle!(system, rattle_amplitude * u"Å")
    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
