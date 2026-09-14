#!/usr/bin/env julia
using PaperBenchmarks
using Dates

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
    # First positional argument can be a comma-separated list of system names.
    # Remaining arguments are --key=value kwargs forwarded to the system functions.
    system_arg = length(ARGS) >= 1 ? ARGS[1] : ""
    system_names = if isempty(system_arg) || startswith(system_arg, "--")
        list_systems()
    else
        split(system_arg, ",")
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
    for name in system_names
        @info "Benchmarking $name ($nrepeats repeats, warmup=$warmup) ..."
        try
            result = benchmark_system(name; kwargs...)
            push!(results, result)
            avg = result[end]
            @info "  done" avg.t_scf avg.t_forces avg.t_stresses
        catch e
            @error "  failed for $name" exception=e
        end
    end

    write_results_csv(output_path, results)
    @info "Results written to $output_path"

    if !isempty(results)
        flat = collect(PaperBenchmarks.iterate_results(results))
        avg_rows = filter(r -> r.repeat == "avg", flat)
        @info "Average timings summary"
        for r in avg_rows
            @info "  $(r.system)" t_scf=r.t_scf t_forces=r.t_forces t_stresses=r.t_stresses
        end
    end
end

main()
