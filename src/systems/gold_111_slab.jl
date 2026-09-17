"""
    gold_111_slab(; Ecut=38, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                    tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                    n_layers=8, n_vacuum=16, repeat_surface=(1, 1),
                    compute_forces=false, compute_stresses=false,
                    callback=identity, kwargs...)

Au(111) slab built with ASE. Default is a 24-layer slab repeated 2×2 in the
surface plane to better approximate bulk behaviour. Returns `(; scfres, forces, stresses)`.
"""
function gold_111_slab(; Ecut=38, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          n_layers=24, n_vacuum=16, repeat_surface=(2, 2),
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    ase_build = ASEconvert.ase.build
    bulk_ase = ase_build.bulk("Au", "fcc")
    surface = ase_build.surface(bulk_ase, (1, 1, 1), n_layers, 0; periodic=true)
    d_vacuum = maximum(maximum, surface.cell) / n_layers * n_vacuum
    surface = ase_build.surface(bulk_ase, (1, 1, 1), n_layers, d_vacuum; periodic=true)
    if repeat_surface != (1, 1)
        surface = surface.repeat((repeat_surface..., 1))
    end
    system = pyconvert(AbstractSystem, surface)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
