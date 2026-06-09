# Dependencies

## Julia packages (from fminv/Project.toml)

| Package | Role | Source |
|---------|------|--------|
| JuliaSourceMechanism | Core inversion framework | Local (gitee.com mirror) |
| DWN | Discrete Wavenumber Method waveform synthesis | Local (gitee.com mirror) |
| SeisTools | Seismic data I/O + processing | Local (gitee.com mirror) |
| SeismicRayTrace | Ray tracing for travel times | Local (gitee.com mirror) |
| JLD2 | Julia HDF5-like data serialization | Julia registry |
| DelimitedFiles | CSV/TSV I/O | Julia stdlib |
| Dates | DateTime handling | Julia stdlib |
| LinearAlgebra | Matrix operations, SVD | Julia stdlib |
| Printf | Formatted string output | Julia stdlib |
| Statistics | Mean, std, normalize | Julia stdlib |
| TOML | TOML config file parsing | Julia stdlib |

## External Julia packages (used by sub-packages)

| Package | Used By | Role |
|---------|---------|------|
| FFTW | JuliaSourceMechanism | FFT (Green's function, cross-correlation alternative) |
| DSP | JuliaSourceMechanism | Digital filters (Butterworth bandpass) |
| Mmap | JuliaSourceMechanism | Memory-mapped I/O for large Green's function libs |
| CairoMakie | fminv/plot.jl | Visualization |

## Binary data dependencies

| Path | Format | Content |
|------|--------|---------|
| `/home/ustc/event/setting.toml` | TOML | Global RTS configuration (algorithm params, Green's fun name) |
| `/home/ustc/gf/<name>/glib_1.bin` | Binary | Pre-computed Green's function library (SEM/FD) |
| `/home/ustc/gf/<name>/setting.toml` | TOML | Green's lib metadata (grid extents, receiver locations) |
| `/tmp/<eventname>/sac/*.sac` | SAC binary | Observed seismic waveforms |
| `/tmp/<eventname>/event.toml` | TOML | Event parameters (lat, lon, depth, magnitude, phase picks) |
| `<dataroot>/greenfun/.../*.gf` | Custom binary | Per-station Green's functions (computed or cached) |
| `JuliaSourceMechanism/src/dat/crust1.bin` | Binary | Crust1.0 global velocity model (360×180×9 layers) |

## SAC file format

Binary format with 158-header-variable fixed header (70 float, 40 int, 24 string, 24 logical) followed by single-precision float time series data. Read via `SeisTools.SAC.readhead()` and `SeisTools.SAC.read()`.

## Green's function file format (.gf)

Custom binary with header + 6-column float matrix:
- Header: TOML-encoded metadata (model description, station info, dt, tp, ts)
- Data: N×6 Float32 matrix [Mxx, Myy, Mzz, Mxy, Mxz, Myz]

## glib binary format (Green's function library)

Fixed-format binary for pre-computed 6D Green's function array:
```
Header: rt(Float32), n(4×Int32), x(n[1]×Float32), y(n[2]×Float32), z(n[3]×Float32), t(n[4]×Float32)
Data:   H(Float32, n[4], 6, 3, n[3], n[2], n[1])
```
Dimensions: [time_samples, MT_components(6), spatial_components(3), z_grid, y_grid, x_grid]
Trilinear interpolation in `_glib_readlocation()` reads 8 corner values per (x,y,z).

## JLD2 data format

Intermediate file `auto.jld2` contains preprocessed env dict:
```julia
jldsave("auto.jld2"; env, status)
```
Result files (`result_stage*.jld2`) contain:
```julia
jldsave("result_stageN.jld2"; env, status, result)
```

## Environment setup

All fminv scripts activate the shared environment at `/home/ustc/app/`:
```julia
using Pkg
Pkg.activate("/home/ustc/app"; io=devnull)
```

Local packages (JuliaSourceMechanism, DWN, SeisTools, SeismicRayTrace) are installed from gitee.com mirrors but local copies exist in `old_codes/` for reference.