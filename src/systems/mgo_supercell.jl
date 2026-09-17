"""
    mgo_supercell(; Ecut=42, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                     tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                     lattice_constant_angstrom=4.21, repeat=(2, 2, 2),
                     compute_forces=false, compute_stresses=false,
                     callback=identity, kwargs...)

MgO rock-salt supercell. Default is a 2×2×2 repetition of the 8-atom
conventional cell (64 atoms). Returns `(; scfres, forces, stresses)`.
"""
function mgo_supercell(; Ecut=42, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          lattice_constant_angstrom=4.21, repeat=(2, 2, 2),
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    a = austrip(lattice_constant_angstrom * u"Å")
    lattice = a * diagm([1.0, 1.0, 1.0])

    mg_frac = [[0.0, 0.0, 0.0],
               [0.0, 0.5, 0.5],
               [0.5, 0.0, 0.5],
               [0.5, 0.5, 0.0]]
    o_frac = [[0.5, 0.5, 0.5],
              [0.5, 0.0, 0.0],
              [0.0, 0.5, 0.0],
              [0.0, 0.0, 0.5]]

    positions = [mg_frac; o_frac]
    elements = vcat(fill(:Mg, 4), fill(:O, 4))
    system = _flexible_system_from_fractional(lattice, elements, positions)
    system = system * Tuple(repeat)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = self_consistent_field(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
