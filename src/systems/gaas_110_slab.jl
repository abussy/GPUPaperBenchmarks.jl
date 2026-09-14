"""
    gaas_110_slab(; Ecut=25, kgrid=(4, 4, 1), architecture=DFTK.CPU(),
                    tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                    n_layers=6, n_vacuum=12, a=5.6537,
                    compute_forces=false, compute_stresses=false,
                    kwargs...)

GaAs(110) slab built with ASE. Default is a 16-layer slab repeated 2×2 in the
surface plane to better approximate bulk behaviour. Returns `(; scfres, forces, stresses)`.
"""
function gaas_110_slab(; Ecut=25, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          n_layers=16, n_vacuum=12, a=5.6537,
                          compute_forces=false, compute_stresses=false,
                          kwargs...)
    ase_build = pyimport("ase.build")
    bulk_ase = ase_build.bulk("GaAs", "zincblende"; a)
    surface = ase_build.surface(bulk_ase, (1, 1, 0), n_layers, 0; periodic=true)
    d_vacuum = maximum(maximum, surface.cell) / n_layers * n_vacuum
    surface = ase_build.surface(bulk_ase, (1, 1, 0), n_layers, d_vacuum; periodic=true)
    system = pyconvert(AbstractSystem, surface)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = self_consistent_field(basis; tol, callback=identity, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
