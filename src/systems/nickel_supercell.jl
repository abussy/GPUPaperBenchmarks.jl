"""
    nickel_supercell(; Ecut=49, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                       tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                       magmom=1.0, rattle_amplitude=0.03,
                       compute_forces=false, compute_stresses=false,
                       callback=identity, kwargs...)

32-atom ferromagnetic fcc nickel supercell (2×2×2 repetition of the 4-atom cubic
cell). Returns `(; scfres, forces, stresses)`.
"""
function nickel_supercell(; Ecut=49, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                           tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                           magmom=1.0, rattle_amplitude=0.03,
                           compute_forces=false, compute_stresses=false,
                           callback=identity, kwargs...)
    system = bulk(:Ni; cubic=true) * (2, 2, 2)
    rattle!(system, rattle_amplitude * u"Å")
    magnetic_moments = fill(magmom, length(system))
    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()..., magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    ρ0 = guess_density(basis, magnetic_moments)
    scfres = _run_scf(basis; tol, callback, ρ=ρ0, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
