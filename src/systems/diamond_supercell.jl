"""
    diamond_supercell(; Ecut=41, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                        tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                        compute_forces=false, compute_stresses=false,
                        callback=identity, kwargs...)

64-atom diamond supercell. Returns `(; scfres, forces, stresses)`.
"""
function diamond_supercell(; Ecut=41, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                            tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                            compute_forces=false, compute_stresses=false,
                            callback=identity, kwargs...)
    system = bulk(:C; cubic=true) * (2, 2, 2)
    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = self_consistent_field(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
