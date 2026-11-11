"""
    _mos2_monolayer_data(; vacuum=10.0)

MoS₂ monolayer data `(lattice, elements, positions_frac)` replicating ASE's
`ase.build.mx2(formula="MoS2", vacuum)` (hexagonal cell, `vacuum` Å of vacuum
on each side). Uses the ASE defaults `a=3.18` Å and slab thickness 3.19 Å.
"""
function _mos2_monolayer_data(; vacuum=10.0)
    a = 3.18          # ASE mx2 default lattice constant
    thickness = 3.19  # ASE mx2 default slab thickness
    c = 2 * vacuum + thickness
    lattice = [a     -a / 2      0.0
               0.0    a * √3 / 2 0.0
               0.0    0.0        c]
    δ = (thickness / 2) / c
    positions = [[0.0, 0.0, 0.5],
                 [2 / 3, 1 / 3, 0.5 + δ],
                 [2 / 3, 1 / 3, 0.5 - δ]]
    (lattice, [:Mo, :S, :S], positions)
end

"""
    mos2_primitive(; Ecut=40, kgrid=(8, 8, 1), architecture=DFTK.CPU(),
                      tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                      compute_forces=false, compute_stresses=false,
                      callback=identity, kwargs...)

3-atom primitive MoS₂ monolayer cell with 10 Å of vacuum on each side
(~20 Å total separation). Returns `(; scfres, forces, stresses)`.

The geometry replicates the ASE `ase.build.mx2(formula="MoS2", vacuum=10.0)`
output that was used previously, without requiring ASE.
"""
function mos2_primitive(; Ecut=40, kgrid=(8, 8, 1), architecture=DFTK.CPU(),
                         tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                         compute_forces=false, compute_stresses=false,
                         callback=identity, kwargs...)
    lattice, elements, positions = _mos2_monolayer_data()
    system = _flexible_system_fractional(lattice, elements, positions)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
