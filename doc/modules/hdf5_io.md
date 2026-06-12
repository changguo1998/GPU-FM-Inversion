# Module: HDF5 I/O

## Purpose

HDF5 file read/write operations shared across all stages. Provides typed accessors that abstract HDF5 group/dataset traversal.

## Used By

| Stage | Language | Usage |
|-------|----------|-------|
| `input.jl` | Julia | Read `raw.h5`, write `database.h5` + `status_0.h5` (strategy only) |
| `preprocess.jl` | Julia | Read `status_{N}.h5` (strategy), write `status_{N}.h5` (trials) |
| `forward.cpp` | C++ | Read `database.h5`, read/write `status_{N}.h5` |
| `assess.jl` | Julia | Read `status_{N}.h5`, `database.h5`, write `status_{N+1}.h5` |
| `output.jl` | Julia | Read all `status_{*}.h5` + `database.h5`, write `output.h5` |
| `driver.sh` | Bash | Check dataset/group existence |

## Julia Interface (`HDF5.jl`)

```julia
# Reading
read_event(file)::EventInfo
read_phase_picks(file)::Vector{PhasePick}
read_stations(file)::Vector{StationInfo}
read_waveform(file, phase_id)::Vector{Float64}
read_config(file)::Dict{String, Any}
read_trials(file)::TrialSet
read_strategy(file)::Strategy
read_misfits(file)::Dict{Symbol, Matrix{Float64}}  # module → data
read_greens(file, phase_id, depth_idx)::Matrix{Float64}
read_index(file)::Index

# Writing
write_database(file, greens, data, index, config)
write_trials(file, trials::TrialSet)
write_misfits(file, module::Symbol, data::AbstractArray)
write_strategy(file, strategy::Strategy)
write_output(file, solution, uncertainty, per_station, summary)

# Structure helpers
h5create_group(file, path)
h5exists(file, path)::Bool
```

## C++ Interface (HDF5 C API)

```cpp
struct Hdf5Handle {
    hid_t file_id;

    void open(const char* path, unsigned flags);
    void close();

    int    read_int_scalar(const char* path);
    double read_double_scalar(const char* path);
    std::vector<int>    read_int_1d(const char* path);
    std::vector<double> read_double_1d(const char* path);
    std::vector<double> read_double_2d(const char* path, int& rows, int& cols);

    bool group_exists(const char* path);
    void create_group(const char* path);
    void write_double_2d(const char* path, const double* data,
                         hsize_t dim1, hsize_t dim2);
};
```

No HighFive dependency — raw HDF5 C API only.

## Key Design Decisions

- **No append mode**: Each stage writes complete datasets.
- **Error handling**: C++ checks all HDF5 return codes. Julia uses exceptions.

## Testing Strategy

- Julia ↔ C++ cross-language: write with Julia, read with C++, verify identical data
- NaN round-trip: verify NaN values survive HDF5 read/write cycle
