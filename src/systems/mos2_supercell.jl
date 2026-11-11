"""
    mos2_supercell(; Ecut=40, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                      tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                      repeat_xy=(4, 4),
                      compute_forces=false, compute_stresses=false,
                      callback=identity, kwargs...)

MoS₂ supercell with 10 Å of vacuum on each side (~20 Å total separation).
Default is a 4×4 in-plane repetition of the primitive 3-atom monolayer
(48 atoms). Returns `(; scfres, forces, stresses)`.

The geometry replicates the ASE `ase.build.mx2(formula="MoS2", vacuum=10.0)`
output followed by `repeat`, without requiring ASE.
"""
function mos2_supercell(; Ecut=40, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                           tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                           repeat_xy=(4, 4),
                           compute_forces=false, compute_stresses=false,
                           callback=identity, kwargs...)
    lattice, elements, positions = _mos2_monolayer_data()
    system = _supercell_fractional(lattice, elements, positions, (repeat_xy..., 1))

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
