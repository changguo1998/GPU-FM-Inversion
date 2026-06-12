# Stage: `setup.jl` — (Superseded)

**This stage has been split into two stages:**

| Former | New |
|--------|-----|
| `setup.jl` (first run, once) | `input.jl` — data ingestion, preprocessing, database creation |
| `setup.jl` (subsequent runs, each loop) | `preprocess.jl` — trial generation from strategy |

See:
- [`doc/stages/input.md`](input.md) — the once-only initialization stage
- [`doc/stages/preprocess.md`](preprocess.md) — the per-loop trial generation stage