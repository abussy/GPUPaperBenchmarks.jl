"""
    gaas_110_slab(; Ecut=42, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                    tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                    n_layers=24, n_vacuum=12, a=5.6537,
                    compute_forces=false, compute_stresses=false,
                    callback=identity, kwargs...)

GaAs(110) slab built with ASE. Default is a 24-layer slab (no surface repeat).
Returns `(; scfres, forces, stresses)`.
"""
function gaas_110_slab(; Ecut=42, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          n_layers=24, n_vacuum=12, a=5.6537,
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    ase_build = ASEconvert.ase.build
    bulk_ase = ase_build.bulk("GaAs", "zincblende"; a)
    surface = ase_build.surface(bulk_ase, (1, 1, 0), n_layers, 0; periodic=true)
    d_vacuum = maximum(maximum, surface.cell) / n_layers * n_vacuum
    surface = ase_build.surface(bulk_ase, (1, 1, 0), n_layers, d_vacuum; periodic=true)
    system = pyconvert(AbstractSystem, surface)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
