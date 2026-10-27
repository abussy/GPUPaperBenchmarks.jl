# Precompile workload for PaperBenchmarks.
#
# Runs a short CPU SCF for silicon_primitive with forces and stresses so that
# the package loads with the hot path already compiled.

@compile_workload begin
    silicon_primitive(;
        Ecut=18,
        kgrid=(8, 8, 8),
        architecture=DFTK.CPU(),
        maxiter=5,
        convergence=:maxiter,
        tol=0.0,
        compute_forces=true,
        compute_stresses=true,
    )
end
