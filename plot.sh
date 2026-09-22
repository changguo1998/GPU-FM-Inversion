#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

help() {
    cat << 'USAGE'
Usage: bash plot.sh --data-dir <dir> [--backend gnuplot] [--output <file>]

Plot the event and station distribution from database.h5 without map projection.
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
    echo "plot.sh: --data-dir is required" >&2
    exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
    echo "plot.sh: data directory not found: $DATA_DIR" >&2
    exit 1
fi
if [[ "$BACKEND" != "gnuplot" ]]; then
    echo "plot.sh: unsupported backend '$BACKEND' (only gnuplot is available)" >&2
    exit 1
fi
if ! command -v gnuplot > /dev/null 2>&1; then
    echo "plot.sh: gnuplot executable not found" >&2
    exit 1
fi

DATABASE="$DATA_DIR/database.h5"
if [[ ! -f "$DATABASE" ]]; then
    echo "plot.sh: database not found: $DATABASE" >&2
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$DATA_DIR/figures/event_stations.png"
fi
mkdir -p "$(dirname "$OUTPUT")"

JULIA_PROJECT="$SCRIPT_DIR"
if [[ -f "$DATA_DIR/Project.toml" ]]; then
    JULIA_PROJECT="$DATA_DIR"
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/scripts/extract_h5.jl" \
    "$DATABASE" "$TEMP_DIR/stations.dat" \
    /station/longitude /station/latitude /station/id
julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/scripts/extract_h5.jl" \
    "$DATABASE" "$TEMP_DIR/event.dat" \
    /event/longitude /event/latitude /event/magnitude

gnuplot -c "$SCRIPT_DIR/scripts/gnuplot/event_stations.gp" \
    "$OUTPUT" "$TEMP_DIR/stations.dat" "$TEMP_DIR/event.dat"

echo "Wrote $OUTPUT"
