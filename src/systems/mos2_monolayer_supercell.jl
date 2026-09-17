"""
    mos2_monolayer_supercell(; Ecut=40, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                                tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                                repeat_xy=(3, 3),
                                compute_forces=false, compute_stresses=false,
                                callback=identity, kwargs...)

MoS₂ monolayer supercell built with ASE. Default is a 3×3 in-plane repetition
of the primitive 3-atom monolayer (27 atoms). Returns `(; scfres, forces, stresses)`.
"""
function mos2_monolayer_supercell(; Ecut=40, kgrid=(2, 2, 1), architecture=DFTK.CPU(),
                                    tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                                    repeat_xy=(3, 3),
                                    compute_forces=false, compute_stresses=false,
                                    callback=identity, kwargs...)
    ase_build = ASEconvert.ase.build
    monolayer = ase_build.mx2(formula="MoS2")
    system = pyconvert(AbstractSystem, monolayer)
    system = system * (repeat_xy..., 1)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = self_consistent_field(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
