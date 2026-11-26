#!/usr/bin/env julia
using PaperBenchmarks

function print_usage(io=stdout)
    println(io, "Usage:")
    println(io, "  julia --project=. scripts/export_qe.jl [converged|timing] [systems] [output_dir]")
    println(io)
    println(io, "Modes:")
    println(io, "  converged  Fully converge the SCF for total-energy comparisons (default).")
    println(io, "  timing     Run exactly 10 SCF iterations, then compute forces.")
    println(io)
    println(io, "systems is an optional comma-separated list; the default is all systems.")
    println(io, "output_dir defaults to qe_inputs/<system>.")
    println(io, "Inputs are named pw_comp.in (converged) or pw_perf.in (timing).")
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        print_usage()
        return
    end
    length(args) <= 3 || throw(ArgumentError("Too many arguments; use --help for usage"))

    mode_name = isempty(args) ? "converged" : lowercase(args[1])
    mode_name in ("converged", "timing") || throw(ArgumentError(
        "Unknown mode '$mode_name'; choose 'converged' or 'timing'"))
    mode = Symbol(mode_name)

    system_arg = length(args) >= 2 ? args[2] : ""
    system_names = isempty(system_arg) ? list_systems() : String.(split(system_arg, ","))
    output_dir = length(args) >= 3 ? args[3] : "qe_inputs"

    @info "Exporting Quantum ESPRESSO inputs" mode systems=system_names output=output_dir
    failed = String[]
    for name in system_names
        system_dir = joinpath(output_dir, name)
        mkpath(system_dir)
        try
            input_path = export_qe_system(name, system_dir; mode)
            isfile(input_path) && filesize(input_path) > 0 ||
                error("No nonempty QE input was produced at $input_path")
            @info "  exported $name"
        catch e
            push!(failed, name)
            @error "  failed for $name" exception=(e, catch_backtrace())
        end
    end
    isempty(failed) || error("QE export failed for: $(join(failed, ", "))")
    @info "Done. Inputs written to $output_dir"
end

main()
