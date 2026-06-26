#!/usr/bin/env bash
set -euo pipefail

# gen_synthetic.sh - One-time synthetic test data generation
#
# Generates minimal external data file (raw.h5) and config.jl for pipeline testing.
# Use this before running driver.sh on a fresh data directory.
# Usage: bash examples/gen_synthetic.sh [data-dir]  (default: CWD)
# Then:  bash driver.sh --data-dir /path/to/data

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTDIR="${1:-.}"

echo "[gen_synthetic] Generating test data in: $(realpath "${OUTDIR}")"
julia "${SCRIPT_DIR}/tests/synthetic_data.jl" "${OUTDIR}"
echo "[gen_synthetic] Done. Run: bash driver.sh --data-dir $(realpath "${OUTDIR}")"
