#!/usr/bin/env julia

using HDF5

length(ARGS) >= 3 || error("usage: extract_h5.jl <input.h5> <output.dat> <dataset> [<dataset> ...]")

input_path, output_path = ARGS[1:2]
dataset_paths = ARGS[3:end]

function read_column(file, path)
    haskey(file, path) || error("dataset not found: $path")
    value = read(file[path])
    value isa AbstractArray &&
        ndims(value) > 1 &&
        error("dataset must be scalar or one-dimensional: $path")
    return value isa AbstractArray ? collect(value) : [value]
end

function write_value(io, value)
    if value isa AbstractString
        escaped = replace(value, '\\' => "\\\\", '"' => "\\\"")
        print(io, '"', escaped, '"')
    else
        print(io, value)
    end
end

h5open(input_path, "r") do file
    columns = [read_column(file, path) for path in dataset_paths]
    row_count = maximum(length, columns)
    for (path, column) in zip(dataset_paths, columns)
        length(column) in (1, row_count) ||
            error("dataset length must be 1 or $row_count: $path has $(length(column))")
    end

    open(output_path, "w") do io
        for row in 1:row_count
            for (column_index, column) in enumerate(columns)
                column_index > 1 && print(io, '\t')
                write_value(io, column[length(column) == 1 ? 1 : row])
            end
            println(io)
        end
    end
end
