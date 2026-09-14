#!/usr/bin/env julia
using PaperBenchmarks

function main()
    system_arg = length(ARGS) >= 1 ? ARGS[1] : ""
    system_names = if isempty(system_arg)
        list_systems()
    else
        split(system_arg, ",")
    end
    output_dir = length(ARGS) >= 2 ? ARGS[2] : "qe_inputs"

    @info "Exporting Quantum ESPRESSO inputs" systems=system_names output=output_dir
    for name in system_names
        system_dir = joinpath(output_dir, name)
        mkpath(system_dir)
        try
            export_qe_system(name, system_dir)
            @info "  exported $name"
        catch e
            @error "  failed for $name" exception=e
        end
    end
    @info "Done. Inputs written to $output_dir"
end

main()
