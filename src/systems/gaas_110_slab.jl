"""
    _gaas_surface_slab_data(; a, n_layers, vacuum)

GaAs slab data `(lattice, elements, positions_frac)` replicating ASE's
`ase.build.surface(ase.build.bulk("GaAs", "zincblende"; a), (1, 1, 0),
n_layers, vacuum)`. Note that ASE's `bulk(..., "zincblende")` returns the
2-atom primitive cell, so the Miller indices (and hence the resulting slab
orientation and stacking) refer to that primitive cell. `n_layers` counts
repetitions of the two-atomic-layer repeat unit along the stack.
"""
function _gaas_surface_slab_data(; a::Real, n_layers::Int, vacuum::Real)
    n_atomic = 2 * n_layers
    s = a / 4  # spacing between consecutive atomic layers along the stack
    c = 2 * vacuum + (n_atomic - 1) * s
    b = a / √2
    lattice = diagm([b, b, c])
    # Stacking pattern cycles with period 4 atomic layers (one bulk repeat
    # along the stacking direction); species alternate Ga/As.
    xy_cycle = ((0.0, 0.0), (0.0, 0.5), (0.5, 0.5), (0.5, 0.0))
    positions = [[xy_cycle[mod(k, 4) + 1][1], xy_cycle[mod(k, 4) + 1][2],
                  (vacuum + k * s) / c] for k in 0:(n_atomic - 1)]
    elements = [k % 2 == 0 ? :Ga : :As for k in 0:(n_atomic - 1)]
    (lattice, elements, positions)
end

"""
    gaas_110_slab(; Ecut=42, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                    tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                    n_layers=24, vacuum=10.0, a=5.6537,
                    compute_forces=false, compute_stresses=false,
                    callback=identity, kwargs...)

GaAs slab. Default is a 24-layer slab (no surface repeat) with 10 Å of vacuum
on each side (~20 Å total separation). Returns `(; scfres, forces, stresses)`.

The geometry replicates the ASE
`ase.build.surface(ase.build.bulk("GaAs", "zincblende"; a), (1, 1, 0),
n_layers, vacuum)` output that was used previously, without requiring ASE.
Note that ASE's `bulk(..., "zincblende")` returns the 2-atom primitive cell,
so the Miller indices (and thus the resulting stacking) refer to that
primitive cell.
"""
function gaas_110_slab(; Ecut=42, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          n_layers=24, vacuum=10.0, a=5.6537,
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    lattice, elements, positions = _gaas_surface_slab_data(; a, n_layers, vacuum)
    system = _flexible_system_fractional(lattice, elements, positions)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
