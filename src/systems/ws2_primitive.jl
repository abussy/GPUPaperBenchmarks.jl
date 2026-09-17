"""
    ws2_primitive(; Ecut=37, kgrid=(8, 8, 1), architecture=DFTK.CPU(),
                     tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                     compute_forces=false, compute_stresses=false,
                     callback=identity, kwargs...)

3-atom WS₂ monolayer cell built with ASE, with 10 Å of vacuum on each side
(~20 Å total separation). Returns `(; scfres, forces, stresses)`.
"""
function ws2_primitive(; Ecut=37, kgrid=(8, 8, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    ase_build = ASEconvert.ase.build
    monolayer = ase_build.mx2(formula="WS2", vacuum=10.0)
    monolayer.pbc = (true, true, true)
    system = pyconvert(AbstractSystem, monolayer)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
