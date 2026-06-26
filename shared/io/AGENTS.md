# AGENTS.md — IO module (`shared/io/src/IO.jl`)

## Role

HDF5 I/O abstractions for the pipeline. Reads external data files, writes pipeline HDF5 files, provides geophysics utilities. All stages depend on this module.

Used by: `input.jl`, `preprocess.jl`, `assess.jl`, `output.jl`, `Grid` (via `H5IO` alias).

## Type structs

| Struct | Fields | Schema |
|---------------|-----------------------------------------------------------------------------------------------|--------------------------------------|
| `EventInfo` | `longitude, latitude, depth, magnitude, origintime` | From external `raw.h5` `/event` |
| `StationInfo` | `id, network, station, component, latitude, longitude, elevation, dt, begin_time` | From `/stations` |
| `PhasePick` | `station_id, P_time, S_time, P_polarity` (Int8) | From `/phase_picks` |
| `Index` | `phase_ids, phase_type, station_idx, distance, azimuth, greens_depth_idx` | Written to `database.h5` `/index` |
| `TrialSet` | `strike, dip, rake, depth, depth_idx, freq_idx` | Written to `status_{N}.h5` `/trials` |
| `Strategy` | Grid params, masks, weights, best results, freq/depth accumulators, iteration, converged flag | Written to `/strategy` (20+ fields) |

## Exports

### Readers (external HDF5)

- `read_event(h5file)` → `EventInfo`
- `read_phase_picks(h5file)` → `Vector{PhasePick}`
- `read_stations(h5file)` → `Vector{StationInfo}`
- `read_waveform(h5file, phase_id)` → `Vector{Float64}`

### Readers (pipeline HDF5)

- `read_trials(h5file)` → `TrialSet`
- `read_strategy(h5file)` → `Strategy`
- `read_misfits(h5file)` → `Dict{Symbol, Matrix{Float64}}`
- `read_greens(h5file, phase_id, depth_idx)` → `Matrix{Float64}`
- `read_index(h5file)` → `Index`
- `read_config(h5file)` → `Dict{String, Any}` (recursive group reader)

### Writers

- `write_database(h5file, greens, data, index, config)` — creates `database.h5` from scratch
- `write_trials(h5file, trials::TrialSet)` — overwrites `/trials` in existing file
- `write_strategy(h5file, strategy::Strategy)` — overwrites `/strategy`
- `write_output(h5file, solution, uncertainty, per_phase, per_station_summary, summary)` — creates `output.h5`
- `write_misfits(h5file, modname::Symbol, data)` — writes `/misfits/{modname}` (not used by current pipeline — forward.cpp writes misfits directly via HDF5 C API)

### Geophysics utilities

- `parse_time_iso(t_str)` → `Float64` seconds-since-epoch (NaN on empty)
- `haversine_distance(lat1, lon1, lat2, lon2)` → distance in km
- `compute_azimuth(lat1, lon1, lat2, lon2)` → azimuth in degrees \[0, 360)
- `extract_station(phase_id)` → `"NET.STA"` from `"NET.STA.COMP.TYPE"`
- `extract_phase_type(phase_id)` → `"P"` or `"S"` from last segment
- `find_latest_status(status_dir)` → `(filepath, iteration_int)` or error if none found

### Helpers

- `h5create_group(h5file, path)` — create group with intermediates
- `h5exists(h5file, path)` → `Bool`
- `_read_group_recursive(gr)` → `Dict{String, Any}` — recursive group reader
- `_write_group_recursive(parent, data)` — recursive Dict→HDF5 writer

## Coding conventions

- All reader functions accept a file path string and return Julia objects.
- Writer functions accept a file path string and data, open `"w"` (create) or `"r+"` (append).
- `write_trials` and `write_strategy` delete-and-recreate datasets (no append).
- `write_output` creates `output.h5` from scratch (`"w"` mode).
- String datasets use variable-length HDF5 strings.
- Helper utility functions are pure — no HDF5 I/O, no side effects.
