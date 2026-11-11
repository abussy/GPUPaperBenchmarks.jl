"""
    _hbn_monolayer_data(; a=2.5, vacuum=10.0)

hBN monolayer data `(lattice, elements, positions_frac)` replicating ASE's
`ase.build.graphene(formula="BN", a, vacuum)` (hexagonal cell, `vacuum` Å of
vacuum on each side).
"""
function _hbn_monolayer_data(; a=2.5, vacuum=10.0)
    c = 2 * vacuum
    lattice = [a     -a / 2      0.0
               0.0    a * √3 / 2 0.0
               0.0    0.0        c]
    positions = [[0.0, 0.0, 0.5], [2 / 3, 1 / 3, 0.5]]
    (lattice, [:B, :N], positions)
end

"""
    hbn_primitive(; Ecut=42, kgrid=(8, 8, 1), architecture=DFTK.CPU(),
                     tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                     compute_forces=false, compute_stresses=false,
                     callback=identity, kwargs...)

2-atom hexagonal boron nitride (hBN) monolayer cell with 10 Å of vacuum on
each side (~20 Å total separation). Returns `(; scfres, forces, stresses)`.

The geometry replicates the ASE `ase.build.graphene(formula="BN", a=2.5,
vacuum=10.0)` output that was used previously, without requiring ASE.
"""
function hbn_primitive(; Ecut=42, kgrid=(8, 8, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    lattice, elements, positions = _hbn_monolayer_data()
    system = _flexible_system_fractional(lattice, elements, positions)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
