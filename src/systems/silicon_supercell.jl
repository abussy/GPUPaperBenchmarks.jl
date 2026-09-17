"""
    silicon_supercell(; Ecut=18, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                        tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                        compute_forces=false, compute_stresses=false,
                        callback=identity, kwargs...)

64-atom silicon supercell (diamond cubic, 2×2×2 repetition of the 8-atom
cubic cell). Returns `(; scfres, forces, stresses)`.
"""
function silicon_supercell(; Ecut=18, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                            tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                            compute_forces=false, compute_stresses=false,
                            callback=identity, kwargs...)
    system = bulk(:Si; cubic=true) * (2, 2, 2)
    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = self_consistent_field(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
