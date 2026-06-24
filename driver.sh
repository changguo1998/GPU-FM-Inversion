#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# driver.sh — Pipeline Orchestration for Focal Mechanism Inversion
#
# Stages:
#   input.jl      (once) → database.h5 + status_0.h5
#   preprocess.jl (loop) → adds /trials to status_{N}.h5
#   forward.cpp   (loop) → adds /misfits to status_{N}.h5
#   assess.jl     (loop) → writes status_{N+1}.h5 (refined strategy)
#   output.jl     (once) → output.h5
#
# Usage:
#   bash driver.sh --data-dir <dir> [--dry-run]
# ==============================================================================

# ── Defaults ───────────────────────────────────────────────────────────────────
DRY_RUN=false

# Project root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FORWARD_BIN="$SCRIPT_DIR/build/forward/forward"

# ── Parse CLI ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
    --data-dir)
        DATA_DIR="$2"
        CONFIG_FILE="$DATA_DIR/config.jl"
        DATABASE_H5="$DATA_DIR/database.h5"
        shift 2
        ;;

    --dry-run)
        DRY_RUN=true
        shift
        ;;
    *)
        echo "[driver] ERROR: Unknown argument: $1" >&2
        echo "Usage: bash driver.sh --data-dir <dir> [--dry-run]" >&2
        exit 1
        ;;
    esac
done

# ── Validate --data-dir ───────────────────────────────────────────────────────
if [[ -z "${DATA_DIR:-}" ]]; then
    echo "[driver] ERROR: --data-dir is required" >&2
    echo "Usage: bash driver.sh --data-dir <dir> [--dry-run]" >&2
    exit 1
fi

# ── Ensure data directory exists ───────────────────────────────────────────────
if [[ ! -d "$DATA_DIR" ]]; then
    echo "[driver] ERROR: data directory not found: $DATA_DIR" >&2
    exit 1
fi

STATUS_DIR="$DATA_DIR/status"
mkdir -p "$STATUS_DIR"

# ==============================================================================
# Helper functions
# ==============================================================================

# Find the latest status_{N}.h5 file and return N (or -1 if none found)
find_latest_n() {
    local max_n=-1
    for f in "$STATUS_DIR"/status_*.h5; do
        [[ -f "$f" ]] || continue
        local basename_f
        basename_f=$(basename "$f")
        if [[ "$basename_f" =~ ^status_([0-9]+)\.h5$ ]]; then
            local n="${BASH_REMATCH[1]}"
            if ((n > max_n)); then
                max_n=$n
            fi
        fi
    done
    echo "$max_n"
}

# Check whether an HDF5 group exists (delegated to assess.jl --query)
h5_group_exists() {
    julia --project="$SCRIPT_DIR" "$SCRIPT_DIR/scripts/assess.jl" --query group-exists "$1" "$2"
}

# Read /strategy/converged value from an HDF5 file (delegated to assess.jl --query)
h5_read_converged() {
    julia --project="$SCRIPT_DIR" "$SCRIPT_DIR/scripts/assess.jl" --query read-converged "$1"
}

# Run a stage; if DRY_RUN, just echo the label
run_stage() {
    local label="$1"
    shift
    if $DRY_RUN; then
        echo "[DRY-RUN] $label"
        return 0
    fi
    echo "[driver] Running $label ..."
    "$@"
    local ret=$?
    if [[ $ret -ne 0 ]]; then
        echo "[driver] ERROR: $label failed (exit code $ret)" >&2
        exit $ret
    fi
    echo "[driver] $label complete."
}

# ==============================================================================
# Main state detection and stage dispatch (loop until converged)
#
# State machine:
#   no database.h5                → input.jl (once)
#   status_N, converged=1         → output.jl (done)
#   status_N, has /misfits        → assess.jl
#   status_N, has /trials         → forward.cpp
#   status_N, has /strategy       → preprocess.jl
#
# After assess writes status_{N+1} with converged=1, that becomes latest
# and triggers output. With converged=0, next loop preprocesses it.
# ==============================================================================

while true; do

    # ── Stage 1: input.jl (once, no database.h5) ──────────────────────────────
    if [[ ! -f "$DATABASE_H5" ]]; then
        # Validate input files exist
        if [[ ! -f "$CONFIG_FILE" ]]; then
            echo "[driver] ERROR: config file not found: $CONFIG_FILE" >&2
            exit 1
        fi
        run_stage "input.jl → database.h5 + status_0.h5" \
            julia --project="$SCRIPT_DIR" "$SCRIPT_DIR/scripts/input.jl" "$CONFIG_FILE"
        # Move status_0.h5 from data dir to status dir
        if [[ -f "$DATA_DIR/status_0.h5" ]]; then
            mv "$DATA_DIR/status_0.h5" "$STATUS_DIR/"
        fi
        if $DRY_RUN; then break; fi
        continue
    fi

    # Find the latest status_{N}.h5
    N=$(find_latest_n)
    if [[ $N -lt 0 ]]; then
        echo "[driver] ERROR: database.h5 exists but no status_N.h5 found." >&2
        exit 1
    fi
    SRC_STATUS="$STATUS_DIR/status_${N}.h5"

    # ── Check converged flag on LATEST status file ────────────────────────────
    # After assess writes status_{N+1} with converged=1, it becomes the latest.
    converged_val=$(h5_read_converged "$SRC_STATUS")
    if [[ "$converged_val" =~ ^1 ]]; then
        echo "[driver] Converged=1 detected in status_${N}.h5"
        run_stage "output.jl → output.h5" \
            julia --project="$SCRIPT_DIR" \
            "$SCRIPT_DIR/output/src/output.jl" \
            "$DATABASE_H5" --status-dir "$STATUS_DIR"
        break
    fi

    # ── Stage 4: assess.jl (has /misfits) ─────────────────────────────────────
    has_misfits=$(h5_group_exists "$SRC_STATUS" "misfits")
    if [[ "$has_misfits" == "true" ]]; then
        NEXT_N=$((N + 1))
        run_stage "assess.jl → status_${NEXT_N}.h5" \
            julia --project="$SCRIPT_DIR" \
            "$SCRIPT_DIR/assess/src/assess.jl" \
            "$SRC_STATUS" "$DATABASE_H5"
        if $DRY_RUN; then
            echo "[DRY-RUN] Would loop to preprocess for status_${NEXT_N}.h5"
            break
        fi
        continue
    fi

    # ── Stage 3: forward.cpp (has /trials, no /misfits) ───────────────────────
    has_trials=$(h5_group_exists "$SRC_STATUS" "trials")
    if [[ "$has_trials" == "true" ]]; then
        run_stage "forward.cpp → /misfits into status_${N}.h5" \
            "$FORWARD_BIN" "$DATABASE_H5" "$SRC_STATUS"
        if $DRY_RUN; then break; fi
        continue
    fi

    # ── Stage 2: preprocess.jl (has /strategy, no /trials) ────────────────────
    run_stage "preprocess.jl → /trials into status_${N}.h5" \
        julia --project="$SCRIPT_DIR" \
        "$SCRIPT_DIR/preprocess/src/preprocess.jl" \
        "$SRC_STATUS" "$DATABASE_H5"
    if $DRY_RUN; then break; fi

done

echo "[driver] Pipeline complete."
