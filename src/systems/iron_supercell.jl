"""
    iron_supercell(; Ecut=25, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                     tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                     magmom=4.0, rattle_amplitude=0.03,
                     compute_forces=false, compute_stresses=false,
                     kwargs...)

54-atom ferromagnetic bcc iron supercell (3×3×3 repetition of the 2-atom cubic
cell). Returns `(; scfres, forces, stresses)`.
"""
function iron_supercell(; Ecut=25, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          magmom=4.0, rattle_amplitude=0.03,
                          compute_forces=false, compute_stresses=false,
                          kwargs...)
    system = bulk(:Fe; cubic=true) * (3, 3, 3)
    rattle!(system, rattle_amplitude * u"Å")
    magnetic_moments = fill(magmom, length(system))
    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()..., magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    ρ0 = guess_density(basis, magnetic_moments)
    scfres = self_consistent_field(basis; tol, callback=identity, ρ=ρ0, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
