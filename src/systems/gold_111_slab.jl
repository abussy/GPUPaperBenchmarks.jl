"""
    _fcc111_slab_data(sym, a; n_layers, vacuum)

fcc(111) slab data `(lattice, elements, positions_frac)` for element `sym`
with fcc lattice constant `a` (Å), `n_layers` atomic layers in ABC stacking,
and `vacuum` Å of vacuum on each side. Replicates ASE's
`ase.build.surface(ase.build.bulk(sym, "fcc"), (1, 1, 1), n_layers, vacuum)`.
"""
function _fcc111_slab_data(sym::Symbol, a::Real; n_layers::Int, vacuum::Real)
    d = a / √2    # in-plane lattice constant of the (111) surface mesh
    dz = a / √3   # (111) interlayer spacing
    c = 2 * vacuum + (n_layers - 1) * dz
    lattice = [d     d / 2      0.0
               0.0   d * √3 / 2 0.0
               0.0   0.0        c]
    # ABC stacking: in-plane fractional positions cycle with period 3.
    xy_cycle = ((0.0, 0.0), (2 / 3, 2 / 3), (1 / 3, 1 / 3))
    positions = [[xy_cycle[mod(k, 3) + 1][1], xy_cycle[mod(k, 3) + 1][2],
                  (vacuum + k * dz) / c] for k in 0:(n_layers - 1)]
    (lattice, fill(sym, n_layers), positions)
end

"""
    gold_111_slab(; Ecut=38, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                    tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                    n_layers=24, vacuum=10.0,
                    compute_forces=false, compute_stresses=false,
                    callback=identity, kwargs...)

Au(111) slab. Default is a 24-layer slab (no surface repeat, 24 atoms) with
10 Å of vacuum on each side (~20 Å total separation). Returns
`(; scfres, forces, stresses)`.

The geometry replicates the ASE
`ase.build.surface(ase.build.bulk("Au", "fcc"), (1, 1, 1), n_layers, vacuum)`
output that was used previously, without requiring ASE.
"""
function gold_111_slab(; Ecut=38, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          n_layers=24, vacuum=10.0,
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    # ASE default fcc lattice constant for Au (consistent with `bulk(:Au)`).
    a = 4.08
    lattice, elements, positions = _fcc111_slab_data(:Au, a; n_layers, vacuum)
    system = _flexible_system_fractional(lattice, elements, positions)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
