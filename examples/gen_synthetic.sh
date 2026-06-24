#!/usr/bin/env bash
#
# gen_synthetic.sh — One-time synthetic test data generation
#
# Generates minimal external data file (raw.h5) and config.jl for pipeline testing.
# Use this before running driver.sh on a fresh data directory.
#
# Usage:
#   bash examples/gen_synthetic.sh              # writes to CWD
#   bash examples/gen_synthetic.sh /tmp/my_data  # writes to /tmp/my_data
#
# Then run:
#   bash driver.sh --data-dir /tmp/my_data
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTDIR="${1:-.}"

echo "[gen_synthetic] Generating test data in: $(realpath "$OUTDIR")"
julia "$SCRIPT_DIR/tests/synthetic_data.jl" "$OUTDIR"
echo "[gen_synthetic] Done. Run: bash driver.sh --data-dir $(realpath "$OUTDIR")"