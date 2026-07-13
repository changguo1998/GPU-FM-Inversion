# IO Module — HDF5 I/O, Type Structs, Geophysics Utilities

## Types

| Struct | Export | Fields | Notes |
|---------------|-------------|---------------------------------------------------------------------------------|-----------------------------------------|
| `EventInfo` | Yes | `longitude`, `latitude`, `depth`, `magnitude`, `origintime` | Event location and magnitude |
| `StationInfo` | Yes | `id`, `network`, `station`, `channel`, `lat`, `lon`, `elev`, `dt`, `begin_time` | Station metadata |
| `ModuleData` | Yes | `obs`, `obs_norm2`, `gf`, `synamp`, `channel_id`, `station_idx` | Unified per-module preprocessing output |
| `PhasePick` | Yes | `station_id`, `P_time`, `S_time`, `P_polarity` | Phase arrival picks |
| `TrialSet` | Yes | `strike`, `dip`, `rake`, `depth`, `depth_idx`, `freq_idx` | Grid trial generation output |
| `Strategy` | No | `depth_indices`, `freq_low_idx`, `freq_high_idx`, `iteration` | Integer indices into `/paraspace` |
| `ConfigError` | No (Config) | `func`, `msg` | Config interface error |

`ModuleData` replaced the earlier `XCorrObs`, `XCorrGF`, `PolarityGF` structs.
It stores per-band observation data and per-depth per-band GF data, with
optional `obs_norm2` and `synamp` fields that are populated only for XCorr-type
modules.

## Key Functions

### Database Writing

- `write_database(h5file, config, event, station, channel_data, gf_data, module_data::Dict{String, ModuleData}; paraspace=nothing)` — creates `database.h5` from scratch; writes `/paraspace`, `/config`, `/event`, `/station`, `/channel`, `/gf`, and per-module groups `/{ModuleName}/` with obs and gf data
- `write_strategy(h5file, strategy)` — writes `/strategy` to `status_N.h5`
- `write_trials(h5file, trials)` — writes `/trials` to `status_N.h5`
- `write_misfits(h5file, modname, data)` — writes per-module misfit matrix to `/misfits`
- `write_paraspace(h5file, paraspace)` — writes `/paraspace` group
- `write_output(h5file, solution, uncertainty, per_phase, per_station_summary, summary)` — writes final `output.h5`

### Database Reading

- `read_config(h5file) -> Dict` — recursive read of `/config`
- `read_event(h5file) -> EventInfo`
- `read_stations(h5file) -> Vector{StationInfo}`
- `read_phase_picks(h5file) -> Vector{PhasePick}`
- `read_waveform(h5file, phase_id) -> Vector{Float64}`
- `read_trials(h5file) -> TrialSet`
- `read_strategy(h5file) -> Strategy`
- `read_misfits(h5file) -> Dict{Symbol, Matrix{Float64}}` — reads all misfit matrices from `/misfits`
- `read_greens(h5file, phase_id, depth_idx) -> Matrix{Float64}` — reads raw GF; uses `/paraspace/depth` to resolve depth value
- `read_paraspace(h5file) -> Dict{String, Any}` — reads `/paraspace`

### Utilities

- `parse_time_iso(t_str) -> Float64` — ISO 8601 to seconds since epoch
- `haversine_distance(lat1, lon1, lat2, lon2) -> Float64` — great-circle distance (km)
- `compute_azimuth(lat1, lon1, lat2, lon2) -> Float64` — azimuth (degrees, 0 = N)
- `extract_station(phase_id) -> String` — station key from phase identifier
- `extract_phase_type(phase_id) -> String` — phase type ("P" or "S") from phase identifier
- `find_latest_status(status_dir) -> (filepath, iteration)` — finds highest `status_N.h5`

## HDF5 Schema

See `doc/schema.md` for the full schema specification.

Key points:

- Per-module data is written to `/{ModuleName}/` — group name matches the module instance name from `misfit_modules`
- `/{ModuleName}/obs/{band}/` contains preprocessed observation data
- `/{ModuleName}/gf/{depth}/{band}/` contains preprocessed Green's functions
- Metadata (`channel_id`, `station_idx`) is written at the module group root
