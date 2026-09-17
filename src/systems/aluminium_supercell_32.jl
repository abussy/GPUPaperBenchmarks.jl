"""
    aluminium_supercell_32(; Ecut=20, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                             tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                             compute_forces=false, compute_stresses=false,
                             callback=identity, kwargs...)

32-atom aluminium supercell (fcc, 2×2×2 repetition of the 4-atom cubic cell).
Returns `(; scfres, forces, stresses)`.
"""
function aluminium_supercell_32(; Ecut=20, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                                  tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                                  compute_forces=false, compute_stresses=false,
                                  callback=identity, kwargs...)
    system = bulk(:Al; cubic=true) * (2, 2, 2)
    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
