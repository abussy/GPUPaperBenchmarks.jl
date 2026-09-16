"""
    nickel_primitive(; Ecut=49, kgrid=(8, 8, 8), architecture=DFTK.CPU(),
                        tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                        magmom=1.0,
                        compute_forces=false, compute_stresses=false,
                        kwargs...)

1-atom primitive fcc nickel cell. Returns `(; scfres, forces, stresses)`.
"""
function nickel_primitive(; Ecut=49, kgrid=(8, 8, 8), architecture=DFTK.CPU(),
                           tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                           magmom=1.0,
                           compute_forces=false, compute_stresses=false,
                           kwargs...)
    system = bulk(:Ni; cubic=false)
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
