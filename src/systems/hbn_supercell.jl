"""
    hbn_supercell(; Ecut=42, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                     tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                     repeat_xy=(4, 4),
                     compute_forces=false, compute_stresses=false,
                     callback=identity, kwargs...)

hBN supercell with 10 Å of vacuum on each side (~20 Å total separation).
Default is a 4×4 in-plane repetition of the primitive 2-atom monolayer
(32 atoms). Returns `(; scfres, forces, stresses)`.

The geometry replicates the ASE `ase.build.graphene(formula="BN", a=2.5,
vacuum=10.0)` output followed by `repeat`, without requiring ASE.
"""
function hbn_supercell(; Ecut=42, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          repeat_xy=(4, 4),
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    lattice, elements, positions = _hbn_monolayer_data()
    system = _supercell_fractional(lattice, elements, positions, (repeat_xy..., 1))

    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
