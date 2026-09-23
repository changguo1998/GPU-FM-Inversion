#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

help() {
    cat << 'USAGE'
Usage: bash plot_waveforms.sh --data-dir <dir> [--backend gnuplot] [--output <file>]

Plot observed and best-trial synthetic waveforms in station rows and phase-component columns.
USAGE
}

DATA_DIR=""
BACKEND="gnuplot"
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            [[ $# -ge 2 ]] || {
                help >&2
                exit 1
            }
            DATA_DIR="$2"
            shift 2
            ;;
        --backend)
            [[ $# -ge 2 ]] || {
                help >&2
                exit 1
            }
            BACKEND="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || {
                help >&2
                exit 1
            }
            OUTPUT="$2"
            shift 2
            ;;
        -h | --help)
            help
            exit 0
            ;;
        *)
            help >&2
            exit 1
            ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "plot_waveforms.sh: --data-dir is required" >&2
    exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
    echo "plot_waveforms.sh: data directory not found: $DATA_DIR" >&2
    exit 1
fi
if [[ "$BACKEND" != "gnuplot" ]]; then
    echo "plot_waveforms.sh: unsupported backend '$BACKEND' (only gnuplot is available)" >&2
    exit 1
fi
if ! command -v gnuplot > /dev/null 2>&1; then
    echo "plot_waveforms.sh: gnuplot executable not found" >&2
    exit 1
fi

DATABASE="$DATA_DIR/database.h5"
RESULT="$DATA_DIR/output.h5"
STATUS_FILE="$(find "$DATA_DIR/status" -maxdepth 1 -type f -name 'status_*.h5' -print 2> /dev/null | sort -V | tail -1)"
for path in "$DATABASE" "$RESULT" "$STATUS_FILE"; do
    if [[ -z "$path" || ! -f "$path" ]]; then
        echo "plot_waveforms.sh: required pipeline output not found: ${path:-status file}" >&2
        exit 1
    fi
done

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$DATA_DIR/figures/waveform_comparison.png"
fi
mkdir -p "$(dirname "$OUTPUT")"

JULIA_PROJECT="$SCRIPT_DIR"
if [[ -f "$DATA_DIR/Project.toml" ]]; then
    JULIA_PROJECT="$DATA_DIR"
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/scripts/extract_waveform_comparison.jl" \
    "$DATABASE" "$STATUS_FILE" "$RESULT" "$TEMP_DIR"

STATION_COUNT="$(wc -l < "$TEMP_DIR/stations.txt")"
STATION_IDS="$(paste -sd ' ' "$TEMP_DIR/stations.txt")"
gnuplot -c "$SCRIPT_DIR/scripts/gnuplot/waveform_comparison.gp" \
    "$OUTPUT" "$TEMP_DIR" "$STATION_COUNT" "$STATION_IDS"

echo "Wrote $OUTPUT"
