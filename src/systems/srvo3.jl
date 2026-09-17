"""
    srvo3(; Ecut=42, kgrid=(6, 6, 6), architecture=DFTK.CPU(),
             tol=default_tol(), pseudopotentials=default_pseudopotentials(),
             lattice_constant_angstrom=3.84,
             compute_forces=false, compute_stresses=false,
             callback=identity, kwargs...)

5-atom cubic SrVO₃ perovskite primitive cell. Returns `(; scfres, forces, stresses)`.
"""
function srvo3(; Ecut=42, kgrid=(6, 6, 6), architecture=DFTK.CPU(),
                  tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                  lattice_constant_angstrom=3.84,
                  compute_forces=false, compute_stresses=false,
                  callback=identity, kwargs...)
    a = austrip(lattice_constant_angstrom * u"Å")
    lattice = a * diagm([1.0, 1.0, 1.0])

    positions = [[0.0, 0.0, 0.0],    # Sr
                 [0.5, 0.5, 0.5],    # V
                 [0.5, 0.5, 0.0],    # O
                 [0.5, 0.0, 0.5],    # O
                 [0.0, 0.5, 0.5]]    # O
    elements = [:Sr, :V, :O, :O, :O]
    system = _flexible_system_from_fractional(lattice, elements, positions)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = self_consistent_field(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
