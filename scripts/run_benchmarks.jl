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
    output_path = joinpath(string(output_dir), "timings_$(timestamp).csv")

    nrepeats = pop!(kwargs, :nrepeats, 5)
    warmup = pop!(kwargs, :warmup, true)
    @info "Running DFTK paper benchmarks" systems=system_names output=output_path nrepeats=nrepeats warmup=warmup

    results = []
    nsystems = length(system_names)
    for (isys, name) in enumerate(system_names)
        progress = "[$isys/$nsystems]"
        @info "$progress Benchmarking $name ($nrepeats repeats, warmup=$warmup) ..."
        try
            result = benchmark_system(name; kwargs...)
            push!(results, result)
            avg = result[end]
            @info "$progress done for $name" avg.t_scf avg.t_forces avg.t_stresses
        catch e
            @error "$progress failed for $name" exception=e
        end
    end

    write_results_csv(output_path, results)
    @info "Results written to $output_path"

    if !isempty(results)
        flat = collect(PaperBenchmarks.iterate_results(results))
        avg_rows = filter(r -> r.repeat == "avg", flat)
        @info "Average timings summary ($(length(avg_rows))/$nsystems systems)"
        for r in avg_rows
            @info "  $(r.system)" t_scf=r.t_scf t_forces=r.t_forces t_stresses=r.t_stresses
        end
    end
end

main()
