#!/usr/bin/env julia

# MPI-aware variant of scripts/run_benchmarks.jl.
# Launch with e.g.:
#     mpiexec -n 4 julia --project=. scripts/run_benchmarks_mpi.jl silicon_primitive --architecture=CPU --nrepeats=1
#
# The calculations run in parallel over MPI ranks; all logging and CSV output
# is emitted only by the master rank (rank 0). If any rank fails during a
# system, the run aborts immediately via MPI_Abort.

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

using MPI
if !MPI.Initialized()
    MPI.Init()
end
comm = MPI.COMM_WORLD
ismaster = MPI.Comm_rank(comm) == 0
nprocs = MPI.Comm_size(comm)

using PaperBenchmarks
using Dates
using DFTK

# Logging helpers that emit only on the master MPI rank.
macro master_info(args...)
    esc(:( if ismaster; @info $(args...); end ))
end
macro master_error(args...)
    esc(:( if ismaster; @error $(args...); end ))
end

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
            elseif lowercase(val_str) in ("true", "false")
                parse(Bool, val_str)
            elseif occursin(r"^\d+$", val_str)
                parse(Int, val_str)
            elseif occursin(r"^\d+\.\d+$", val_str)
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
    # DFTK.setup_threading uses Threads.nthreads() for DFTK/BLAS and 1 for FFTW.
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
    output_path = if ismaster
        mkpath(string(output_dir))
        joinpath(string(output_dir), "timings_$(timestamp).csv")
    else
        ""
    end
    output_path = MPI.bcast(output_path, 0, comm)

    nrepeats = pop!(kwargs, :nrepeats, 5)
    warmup = pop!(kwargs, :warmup, true)
    @master_info "Running DFTK paper benchmarks (MPI)" systems=system_names output=output_path nrepeats=nrepeats warmup=warmup mpi_ranks=nprocs

    results = []
    nsystems = length(system_names)
    for (isys, name) in enumerate(system_names)
        progress = "[$isys/$nsystems]"
        @master_info "$progress Benchmarking $name ($nrepeats repeats, warmup=$warmup) ..."

        local result
        local err = nothing
        ok = true
        try
            result = benchmark_system(name; kwargs...)
        catch e
            ok = false
            err = e
        end

        # Synchronise success across all MPI ranks and abort if any rank failed.
        ok_int = ok ? 1 : 0
        ok_int = MPI.Allreduce(ok_int, MPI.LAND, comm)
        ok = ok_int != 0

        if !ok
            if ismaster && err !== nothing
                @master_error "$progress failed for $name" exception=err
            else
                @master_error "$progress failed for $name (failed on at least one MPI rank)"
            end
            MPI.Abort(comm, 1)
        end

        if ismaster
            push!(results, result)
            avg = result[end]
            @master_info "$progress done for $name" avg.t_scf avg.t_forces avg.t_stresses
        end
    end

    if ismaster
        write_results_csv(output_path, results)
        @master_info "Results written to $output_path"
    end
    MPI.Barrier(comm)

    if ismaster && !isempty(results)
        flat = collect(PaperBenchmarks.iterate_results(results))
        avg_rows = filter(r -> r.repeat == "avg", flat)
        @master_info "Average timings summary ($(length(avg_rows))/$nsystems systems)"
        for r in avg_rows
            @master_info "  $(r.system)" t_scf=r.t_scf t_forces=r.t_forces t_stresses=r.t_stresses
        end
    end
    MPI.Barrier(comm)
end

main()
