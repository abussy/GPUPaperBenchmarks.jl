"""
    mos2_primitive(; Ecut=40, kgrid=(6, 6, 1), architecture=DFTK.CPU(),
                      tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                      compute_forces=false, compute_stresses=false,
                      kwargs...)

3-atom primitive MoS₂ monolayer cell built with ASE. Returns
`(; scfres, forces, stresses)`.
"""
function mos2_primitive(; Ecut=40, kgrid=(6, 6, 1), architecture=DFTK.CPU(),
                         tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                         compute_forces=false, compute_stresses=false,
                         kwargs...)
    ase_build = pyimport("ase.build")
    monolayer = ase_build.mx2(formula="MoS2")
    system = pyconvert(AbstractSystem, monolayer)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = self_consistent_field(basis; tol, callback=identity, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
