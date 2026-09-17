"""
    tio2_110_slab(; Ecut=42, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                    tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                    n_layers=24, vacuum=10.0, a=4.594, c=2.958,
                    compute_forces=false, compute_stresses=false,
                    callback=identity, kwargs...)

Rutile TiO₂(110) slab built with ASE. Default is a 24-layer slab (no surface
repeat) with 10 Å of vacuum on each side (~20 Å total separation). Returns
`(; scfres, forces, stresses)`.
"""
function tio2_110_slab(; Ecut=42, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          n_layers=24, vacuum=10.0, a=4.594, c=2.958,
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    ase_spacegroup = ASEconvert.ase.spacegroup
    ase_build = ASEconvert.ase.build

    # Rutile TiO2: space group P42/mnm (136).
    # Ti at (0, 0, 0), O at (0.305, 0.305, 0).
    bulk_tio2 = ase_spacegroup.crystal(symbols=["Ti", "O"],
                                       basis=[0.0 0.0 0.0; 0.305 0.305 0.0],
                                       spacegroup=136,
                                       cellpar=[a, a, c, 90.0, 90.0, 90.0])

    surface = ase_build.surface(bulk_tio2, (1, 1, 0), n_layers, vacuum; periodic=true)
    system = pyconvert(AbstractSystem, surface)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
