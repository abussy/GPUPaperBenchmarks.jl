#!/usr/bin/env julia

# Load the requested GPU backend at top level, before PaperBenchmarks, so that
# CUDA / AMDGPU (and the corresponding DFTK extensions) are available in the
# same world age as the benchmark code.
function _load_backend_from_args()
    for arg in ARGS
        startswith(arg, "--architecture=") || continue
        arch = lowercase(split(arg, "=", limit=2)[2])
        if arch == "cuda"
            @eval using CUDA
        elseif arch == "amdgpu"
            @eval using AMDGPU
        end
        break
    end
end
_load_backend_from_args()

# MPI is a project dependency. Initialise it only when the script was launched
# through an MPI launcher; otherwise run in serial mode with rank-0 semantics.
using MPI

function _detect_mpi_launch()
    MPI.Initialized() || any(haskey(ENV, k) for k in (
        "OMPI_COMM_WORLD_SIZE",      # OpenMPI
        "PMI_SIZE",                  # MPICH / Intel MPI / Cray PALS
        "MPICH_RANK_REORDER_DISPLAY",
        "SLURM_NTASKS",              # Slurm srun
        "ALPS_APP_PE",               # Cray ALPS
        "PALS_NODE_ID",              # Cray PALS
    ))
end

function _init_mpi()
    if _detect_mpi_launch()
        if !MPI.Initialized()
            MPI.Init()
        end
        return MPI.COMM_WORLD
    else
        return nothing
    end
end

const COMM = _init_mpi()
const ISMASTER = isnothing(COMM) || MPI.Comm_rank(COMM) == 0
const NPROCS = isnothing(COMM) ? 1 : MPI.Comm_size(COMM)

# MPI-aware helpers that degrade to serial no-ops when COMM === nothing.
_mpi_bcast(val, root::Integer=0) = isnothing(COMM) ? val : MPI.bcast(val, root, COMM)
_mpi_allreduce(val, op) = isnothing(COMM) ? val : MPI.Allreduce(val, op, COMM)
_mpi_barrier() = isnothing(COMM) || MPI.Barrier(COMM)
_mpi_abort(code::Integer) = isnothing(COMM) ? exit(code) : MPI.Abort(COMM, code)

macro run_info(args...)
    esc(:( if ISMASTER; @info $(args...); end ))
end

macro run_error(args...)
    esc(:( if ISMASTER; @error $(args...); end ))
end

using PaperBenchmarks
using Dates
using DFTK

function parse_kwargs(args)
    kwargs = Dict{Symbol, Any}()
    for arg in args
        startswith(arg, "--") || continue
        kv = split(arg[3:end], "=", limit=2)
        if length(kv) == 2
            key = Symbol(kv[1])
            val_str = kv[2]
            val = if key == :kgrid
                Tuple(parse.(Int, split(val_str, ",")))
            elseif key == :convergence
                Symbol(lowercase(val_str))
            elseif lowercase(val_str) in ("true", "false")
                parse(Bool, val_str)
            elseif occursin(r"^\d+$", val_str)
                parse(Int, val_str)
            elseif occursin(r"^\d+(\.\d+)?([eE][+-]?\d+)?$", val_str)
                parse(Float64, val_str)
            else
                val_str
            end
            kwargs[key] = val
        end
    end
    return kwargs
end

function parse_architecture(kwargs)
    arch_name = get(kwargs, :architecture, "CPU")
    return select_architecture(arch_name)
end

function main()
    # Configure BLAS, FFTW and DFTK threading once at startup.
    DFTK.setup_threading()

    # First positional argument can be a comma-separated list of system names.
    # Remaining arguments are --key=value kwargs forwarded to the system functions.
    system_arg = length(ARGS) >= 1 ? ARGS[1] : ""
    system_names = if isempty(system_arg) || startswith(system_arg, "--")
        list_systems()
    else
        String.(split(system_arg, ","))
    end

    kwargs = parse_kwargs(ARGS)
    architecture = parse_architecture(kwargs)
    kwargs[:architecture] = architecture

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_dir = pop!(kwargs, :output_dir, "results")
    output_path = if ISMASTER
        mkpath(string(output_dir))
        joinpath(string(output_dir), "timings_$(timestamp).csv")
    else
        ""
    end
    output_path = _mpi_bcast(output_path, 0)

    nrepeats = pop!(kwargs, :nrepeats, 5)
    warmup = pop!(kwargs, :warmup, true)
    compute_forces = get(kwargs, :compute_forces, true)
    compute_stresses = get(kwargs, :compute_stresses, true)
    verbose = pop!(kwargs, :verbose, false)
    if verbose
        kwargs[:callback] = DFTK.ScfDefaultCallback()
    end
    @run_info "Running DFTK paper benchmarks" systems=system_names output=output_path nrepeats=nrepeats warmup=warmup compute_forces=compute_forces compute_stresses=compute_stresses verbose=verbose mpi_ranks=NPROCS

    results = []
    nsystems = length(system_names)
    for (isys, name) in enumerate(system_names)
        progress = "[$isys/$nsystems]"
        @run_info "$progress Benchmarking $name ($nrepeats repeats, warmup=$warmup, forces=$compute_forces, stresses=$compute_stresses) ..."

        local result
        local err = nothing
        ok = true
        try
            result = benchmark_system(name; nrepeats=nrepeats, warmup=warmup, kwargs...)
        catch e
            ok = false
            err = e
        end

        if !isnothing(COMM)
            # MPI mode: all ranks must agree on success before proceeding.
            ok_int = ok ? 1 : 0
            ok_int = _mpi_allreduce(ok_int, MPI.LAND)
            ok = ok_int != 0
        end

        if !ok
            if isnothing(COMM)
                # Serial mode: log the error and continue with the next system.
                @run_error "$progress failed for $name" exception=err
                continue
            else
                # MPI mode: abort immediately so that no rank continues alone.
                if ISMASTER && err !== nothing
                    @run_error "$progress failed for $name" exception=err
                else
                    @run_error "$progress failed for $name (failed on at least one MPI rank)"
                end
                _mpi_abort(1)
            end
        end

        if ISMASTER
            append_results_csv(output_path, result)
            push!(results, result)
            avg = result[end]
            @run_info "$progress done for $name" avg.t_scf avg.t_forces avg.t_stresses
        end
    end

    @run_info "Results written incrementally to $output_path"
    _mpi_barrier()

    if ISMASTER && !isempty(results)
        flat = collect(PaperBenchmarks.iterate_results(results))
        avg_rows = filter(r -> r.repeat == "avg", flat)
        @run_info "Average timings summary ($(length(avg_rows))/$nsystems systems)"
        for r in avg_rows
            @run_info "  $(r.system)" t_scf=r.t_scf t_forces=r.t_forces t_stresses=r.t_stresses
        end
    end
    _mpi_barrier()
end

main()
