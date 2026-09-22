# Timing and runner logic.

const _CUDA_PKGID = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
const _AMDGPU_PKGID = Base.PkgId(Base.UUID("21141c5a-9bdb-4563-92ae-f87d6854732e"), "AMDGPU")

"""
    cuda_architecture()

Return the CUDA architecture object. Extended by `PaperBenchmarksCUDAExt` when
CUDA is loaded.
"""
function cuda_architecture end

"""
    amdgpu_architecture()

Return the AMDGPU architecture object. Extended by `PaperBenchmarksAMDGPUExt` when
AMDGPU is loaded.
"""
function amdgpu_architecture end

"""
    reclaim_memory(architecture)

Reclaim memory associated with `architecture`. The base implementation calls the
Julia garbage collector. GPU extensions overload this method to additionally
call the vendor-specific memory-pool reclaim function after the GC.
"""
function reclaim_memory end

reclaim_memory(::DFTK.CPU) = GC.gc(true)

function _require_extension(pkgid::Base.PkgId, name::String)
    try
        Base.require(pkgid)
    catch e
        error("Could not load $name. Make sure $(pkgid.name).jl is installed and loadable.")
    end
end

"""
    select_architecture(name::AbstractString)

Return a DFTK architecture object for the requested backend.
"""
function select_architecture(name::AbstractString)
    name = lowercase(name)
    if name == "cpu"
        return DFTK.CPU()
    elseif name == "cuda"
        _require_extension(_CUDA_PKGID, "CUDA")
        return Base.invokelatest(cuda_architecture)
    elseif name == "amdgpu"
        _require_extension(_AMDGPU_PKGID, "AMDGPU")
        return Base.invokelatest(amdgpu_architecture)
    else
        error("Unknown architecture: $name (choose CPU, CUDA, or AMDGPU)")
    end
end

"""
    benchmark_system(name::String; Ecut=nothing, kgrid=nothing,
                     architecture=DFTK.CPU(), tol=default_tol(),
                     convergence::Symbol=:density,
                     compute_forces=true, compute_stresses=true,
                     callback=identity,
                     nrepeats=5, warmup=true, maxiter=nothing, kwargs...)

Run SCF, forces, and stresses for a single benchmark system `nrepeats` times
and return a vector of named tuples. The last entry reports the average timing
of all repeats; the remaining entries report the individual repeats.

Use `compute_forces=false` or `compute_stresses=false` to skip the corresponding
calculation; the associated timing columns are filled with `NaN`.

The `convergence` keyword selects the SCF convergence criterion:
`:density` (default), `:energy`, or `:force`. The tolerance is set by `tol`.

Set `callback=DFTK.ScfDefaultCallback()` to display the SCF progress log for
each repeat.

If `warmup=true` (default), a single non-recorded SCF run (plus forces and
stresses, when requested) is performed first to warm up the GPU/CPU code path
and avoid including JIT compilation time in the reported results.

Set `convergence=:maxiter` together with `maxiter` to run the SCF for a fixed
number of iterations. In this mode `tol` is forced to `0.0` so that the SCF
stops only when `maxiter` iterations have been performed. `maxiter` can also
be used with the other convergence criteria as a hard cap on the number of
iterations.
"""
function benchmark_system(name::String; Ecut=nothing, kgrid=nothing,
                          architecture=DFTK.CPU(), tol=default_tol(),
                          convergence::Symbol=:density,
                          compute_forces=true, compute_stresses=true,
                          callback=identity,
                          nrepeats=5, warmup=true, maxiter=nothing, kwargs...)
    f = get_system_function(name)
    nrepeats >= 1 || error("nrepeats must be at least 1")
    convergence in (:density, :energy, :force, :maxiter) ||
        error("Unknown convergence criterion: $convergence (choose :density, :energy, :force, :maxiter)")
    if convergence == :maxiter
        isnothing(maxiter) && error("maxiter must be specified when convergence=:maxiter")
        maxiter >= 1 || error("maxiter must be at least 1")
        tol = 0.0
    end

    # Build the argument list dynamically so that unspecified Ecut/kgrid use
    # the defaults encoded in each system file.
    call_kwargs = Dict{Symbol, Any}(:architecture => architecture, :tol => tol,
                                    :convergence => convergence, :callback => callback)
    if !isnothing(maxiter)
        call_kwargs[:maxiter] = maxiter
    end
    merge!(call_kwargs, kwargs)
    if !isnothing(Ecut)
        call_kwargs[:Ecut] = Ecut
    end
    if !isnothing(kgrid)
        call_kwargs[:kgrid] = kgrid
    end

    # Non-recorded warm-up run to avoid measuring GPU JIT / CPU compilation.
    if warmup
        result_warmup = f(; call_kwargs...)
        scfres_warmup = result_warmup.scfres
        compute_forces && compute_forces_cart(scfres_warmup)
        compute_stresses && compute_stresses_cart(scfres_warmup)
        reclaim_memory(architecture)
    end

    repeats = []
    for rep in 1:nrepeats
        t_scf = @elapsed result = f(; call_kwargs...)
        scfres = result.scfres
        t_forces = if compute_forces
            @elapsed compute_forces_cart(scfres)
        else
            NaN
        end
        t_stresses = if compute_stresses
            @elapsed compute_stresses_cart(scfres)
        else
            NaN
        end
        reclaim_memory(architecture)

        basis = scfres.basis
        model = basis.model
        push!(repeats, (
            system = name,
            repeat = string(rep),
            natoms = length(model.atoms),
            nelectrons = model.n_electrons,
            Ecut = basis.Ecut,
            kgrid = join(basis.kgrid.kgrid_size, "x"),
            architecture = string(typeof(architecture)),
            t_scf = t_scf,
            t_forces = t_forces,
            t_stresses = t_stresses,
            energy = Float64(scfres.energies.total),
            n_scfiter = scfres.n_iter,
            fft_size = join(basis.fft_size, "x"),
        ))
    end

    # Append an average row. Non-timing fields are taken from the last repeat.
    last = repeats[end]
    avg = (
        system = name,
        repeat = "avg",
        natoms = last.natoms,
        nelectrons = last.nelectrons,
        Ecut = last.Ecut,
        kgrid = last.kgrid,
        architecture = last.architecture,
        t_scf = mean(r.t_scf for r in repeats),
        t_forces = compute_forces ? mean(r.t_forces for r in repeats) : NaN,
        t_stresses = compute_stresses ? mean(r.t_stresses for r in repeats) : NaN,
        energy = last.energy,
        n_scfiter = last.n_scfiter,
        fft_size = last.fft_size,
    )
    return [repeats; avg]
end

const _CSV_HEADER = [:system, :repeat, :natoms, :nelectrons, :Ecut, :kgrid, :architecture,
                       :t_scf, :t_forces, :t_stresses, :energy, :n_scfiter, :fft_size]

# Convert a single result or a vector of result rows into named tuples with a
# deterministic column order.
function _result_rows(result)
    flat = result isa AbstractVector ? result : [result]
    return [(; (k => r[k] for k in _CSV_HEADER)...) for r in flat]
end

"""
    write_results_csv(path::String, results)

Write a (possibly nested) vector of result named tuples to a CSV file.
Nested vectors are flattened so that each repeat and each average occupies one
row.
"""
function write_results_csv(path::String, results)
    mkpath(dirname(path))
    flat = collect(Any, iterate_results(results))
    rows = _result_rows(flat)
    CSV.write(path, rows; header=string.(_CSV_HEADER))
end

"""
    append_results_csv(path::String, result)

Append the result rows of a single benchmark system to `path`. The header is
written only if the file does not yet exist or is empty, making this safe to
call repeatedly as systems complete.
"""
function append_results_csv(path::String, result)
    mkpath(dirname(path))
    rows = _result_rows(result)
    if !isfile(path) || filesize(path) == 0
        CSV.write(path, rows; header=string.(_CSV_HEADER))
    else
        CSV.write(path, rows; append=true)
    end
end

"""
    iterate_results(results)

Flatten a vector of results or a vector of result vectors into a single
iterator of rows.
"""
function iterate_results(results)
    isempty(results) && return Any[]
    if results[1] isa AbstractVector
        return Iterators.flatten(results)
    else
        return results
    end
end
