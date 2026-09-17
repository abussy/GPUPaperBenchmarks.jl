"""
    zro2_supercell(; Ecut=42, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                      tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                      lattice_constant_angstrom=5.08, repeat=(2, 2, 2),
                      compute_forces=false, compute_stresses=false,
                      callback=identity, kwargs...)

Cubic fluorite ZrO₂ supercell. Default is a 2×2×2 repetition of the 12-atom
conventional cell (96 atoms). Returns `(; scfres, forces, stresses)`.
"""
function zro2_supercell(; Ecut=42, kgrid=(1, 1, 1), architecture=DFTK.CPU(),
                          tol=default_tol(), pseudopotentials=default_pseudopotentials(),
                          lattice_constant_angstrom=5.08, repeat=(2, 2, 2),
                          compute_forces=false, compute_stresses=false,
                          callback=identity, kwargs...)
    a = austrip(lattice_constant_angstrom * u"Å")
    lattice = a * diagm([1.0, 1.0, 1.0])

    zr_frac = [[0.0, 0.0, 0.0],
               [0.0, 0.5, 0.5],
               [0.5, 0.0, 0.5],
               [0.5, 0.5, 0.0]]
    o_frac = [[0.25, 0.25, 0.25],
              [0.25, 0.75, 0.75],
              [0.75, 0.25, 0.75],
              [0.75, 0.75, 0.25],
              [0.75, 0.75, 0.75],
              [0.75, 0.25, 0.25],
              [0.25, 0.75, 0.25],
              [0.25, 0.25, 0.75]]

    positions = [zr_frac; o_frac]
    elements = vcat(fill(:Zr, 4), fill(:O, 8))
    system = _flexible_system_from_fractional(lattice, elements, positions)
    system = system * Tuple(repeat)

    model = model_DFT(system; functionals=default_functional(), pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid, architecture)
    scfres = _run_scf(basis; tol, callback, kwargs...)
    forces = compute_forces ? compute_forces_cart(scfres) : nothing
    stresses = compute_stresses ? compute_stresses_cart(scfres) : nothing
    return (; scfres, forces, stresses)
end
