"""
    copt_primitive(; Ecut=48, kgrid=(8, 8, 8), architecture=DFTK.CPU(),
                      tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                      lattice_constant_a_angstrom=3.83, lattice_constant_c_angstrom=3.71,
                      co_magmom=1.8, pt_magmom=0.0,
                      compute_forces=false, compute_stresses=false,
                      callback=identity, kwargs...)

2-atom ferromagnetic primitive CoPt cell (L1₀ ordered tetragonal structure).
Co sits at the origin and Pt at the body centre. Returns
`(; scfres, forces, stresses)`.
"""
function copt_primitive(; Ecut=48, kgrid=(8, 8, 8), architecture=DFTK.CPU(),
                           tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                           lattice_constant_a_angstrom=3.83,
                           lattice_constant_c_angstrom=3.71,
                           co_magmom=1.8, pt_magmom=0.0,
                           compute_forces=false, compute_stresses=false,
                           callback=identity, kwargs...)
    a = austrip(lattice_constant_a_angstrom * u"Å")
    c = austrip(lattice_constant_c_angstrom * u"Å")
    lattice = diagm([a, a, c])

    positions = [[0.0, 0.0, 0.0],
                 [0.5, 0.5, 0.5]]
    elements = [:Co, :Pt]
    magnetic_moments = [co_magmom, pt_magmom]
    system = _flexible_system_from_fractional(lattice, elements, positions)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()..., magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    ρ0 = guess_density(basis, magnetic_moments)
    scfres = _run_scf(basis; tol, callback, ρ=ρ0, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
