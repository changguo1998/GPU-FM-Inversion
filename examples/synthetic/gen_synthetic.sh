#!/usr/bin/env bash
set -euo pipefail

# gen_synthetic.sh — Generate synthetic test data for pipeline testing.
#
# Outputs (in OUTDIR):
#   stations.txt       — station list (station_id, lat, lon)
#   {station}.{ch}.dat — waveform per station+channel (one column)
#   phases.txt         — phase picks (station_id, P_time, S_time)
#
# Usage:
#   bash examples/synthetic/gen_synthetic.sh [data-dir]   # default: examples/synthetic/
#   bash examples/synthetic/gen_synthetic.sh /tmp/test_event --nsta 6
#   bash examples/synthetic/gen_synthetic.sh --strike 30 --dip 60 --rake 90

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# First non-flag arg = output dir; remaining --flags passed to Julia
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
	OUTDIR="$1"
	shift
else
	OUTDIR="${SCRIPT_DIR}"
fi

mkdir -p "${OUTDIR}"
echo "[gen_synthetic] Generating test data in: $(realpath "${OUTDIR}")"
julia "${SCRIPT_DIR}/../../tests/synthetic_data.jl" "${OUTDIR}" "$@"
echo "[gen_synthetic] Done. Files in $(realpath "${OUTDIR}"):"
echo "  stations.txt  phases.txt  *.dat"
