"""
    srvo3_supercell(; Ecut=42, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                      tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                      lattice_constant_angstrom=3.84, repeat=(3, 3, 3),
                      compute_forces=false, compute_stresses=false,
                      kwargs...)

SrVO₃ perovskite supercell. Default is a 3×3×3 repetition of the 5-atom cubic
primitive cell (135 atoms). Returns `(; scfres, forces, stresses)`.
"""
function srvo3_supercell(; Ecut=42, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                           tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                           lattice_constant_angstrom=3.84, repeat=(3, 3, 3),
                           compute_forces=false, compute_stresses=false,
                           kwargs...)
    a = austrip(lattice_constant_angstrom * u"Å")
    lattice = a * diagm([1.0, 1.0, 1.0])

    positions = [[0.0, 0.0, 0.0],    # Sr
                 [0.5, 0.5, 0.5],    # V
                 [0.5, 0.5, 0.0],    # O
                 [0.5, 0.0, 0.5],    # O
                 [0.0, 0.5, 0.5]]    # O
    elements = [:Sr, :V, :O, :O, :O]
    system = _flexible_system_from_fractional(lattice, elements, positions)
    system = system * Tuple(repeat)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials,
                      default_smearing()...)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = self_consistent_field(basis; tol, callback=identity, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
