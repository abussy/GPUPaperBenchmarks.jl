"""
    mos2_monolayer_supercell(; Ecut=40, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                                tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                                repeat_xy=(4, 4),
                                compute_forces=false, compute_stresses=false,
                                callback=identity, kwargs...)

MoS₂ monolayer supercell built with ASE, with 10 Å of vacuum on each side
(~20 Å total separation). Default is a 4×4 in-plane repetition of the
primitive 3-atom monolayer (48 atoms). Returns `(; scfres, forces, stresses)`.
"""
function mos2_monolayer_supercell(; Ecut=40, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                                     tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                                     repeat_xy=(4, 4),
                                     compute_forces=false, compute_stresses=false,
                                     callback=identity, kwargs...)
    ase_build = ASEconvert.ase.build
    monolayer = ase_build.mx2(formula="MoS2", vacuum=10.0)
    monolayer.pbc = (true, true, true)
    supercell = monolayer.repeat((repeat_xy..., 1))
    system = pyconvert(AbstractSystem, supercell)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
