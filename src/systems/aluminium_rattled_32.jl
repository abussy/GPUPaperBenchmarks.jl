"""
    aluminium_rattled_32(; Ecut=20, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                           tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                           rattle_amplitude=0.03,
                           compute_forces=false, compute_stresses=false,
                           kwargs...)

32-atom aluminium supercell with a small random rattle (useful for force
benchmarks). Returns `(; scfres, forces, stresses)`.
"""
function aluminium_rattled_32(; Ecut=20, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                                tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                                rattle_amplitude=0.03,
                                compute_forces=false, compute_stresses=false,
                                kwargs...)
    system = bulk(:Al; cubic=true) * (2, 2, 2)
    rattle!(system, rattle_amplitude * u"Å")
    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = self_consistent_field(basis; tol, callback=identity, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
