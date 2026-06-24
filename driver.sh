#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# driver.sh — Pipeline Orchestration for Focal Mechanism Inversion
#
# Stages:
#   input (once) → loop: [preprocess → forward → assess] → output
#
# State machine (assess.jl decides converged via exit code):
#   exit 0  = continue → next loop iteration
#   exit 10 = converged → proceed to output.jl
#
# Usage:
#   bash driver.sh --data-dir <dir>
# ==============================================================================

# ── Defaults ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FORWARD_BIN="$SCRIPT_DIR/forward/build/forward"

# ── Parse CLI ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
	case "$1" in
	--data-dir)
		DATA_DIR="$2"
		shift 2
		;;
	*)
		echo "[driver] ERROR: Unknown argument: $1" >&2
		echo "Usage: bash driver.sh --data-dir <dir>" >&2
		exit 1
		;;
	esac
done

if [[ -z "${DATA_DIR:-}" ]]; then
	echo "[driver] ERROR: --data-dir is required" >&2
	exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
	echo "[driver] ERROR: data directory not found: $DATA_DIR" >&2
	exit 1
fi

CONFIG_FILE="$DATA_DIR/config.jl"
DATABASE_H5="$DATA_DIR/database.h5"
STATUS_DIR="$DATA_DIR/status"
mkdir -p "$STATUS_DIR"

# ==============================================================================
# Helpers
# ==============================================================================

find_latest_n() {
	local max_n=-1
	for f in "$STATUS_DIR"/status_*.h5; do
		[[ -f "$f" ]] || continue
		local basename_f
		basename_f=$(basename "$f")
		if [[ "$basename_f" =~ ^status_([0-9]+)\.h5$ ]]; then
			local n="${BASH_REMATCH[1]}"
			((n > max_n)) && max_n=$n
		fi
	done
	echo "$max_n"
}

run_stage() {
	local label="$1"
	shift
	echo "[driver] $label ..."
	"$@"
	echo "[driver] $label done."
}

# ==============================================================================
# Main
# ==============================================================================

# ── Stage 1: input (once) ─────────────────────────────────────────────────────
if [[ ! -f "$DATABASE_H5" ]]; then
	if [[ ! -f "$CONFIG_FILE" ]]; then
		echo "[driver] ERROR: config file not found: $CONFIG_FILE" >&2
		exit 1
	fi
	run_stage "input.jl → database.h5 + status_0.h5" \
		julia --project="$SCRIPT_DIR" "$SCRIPT_DIR/scripts/input.jl" "$CONFIG_FILE"

	if [[ -f "$DATA_DIR/status_0.h5" ]]; then
		mv "$DATA_DIR/status_0.h5" "$STATUS_DIR/"
	fi
fi

# ── Loop: preprocess → forward → assess ───────────────────────────────────────
while true; do
	N=$(find_latest_n)
	if [[ $N -lt 0 ]]; then
		echo "[driver] ERROR: no status_N.h5 found." >&2
		exit 1
	fi
	SRC_STATUS="$STATUS_DIR/status_${N}.h5"

	run_stage "preprocess.jl (iteration $N)" \
		julia --project="$SCRIPT_DIR" \
		"$SCRIPT_DIR/scripts/preprocess.jl" "$SRC_STATUS" "$DATABASE_H5"

	run_stage "forward.cpp (iteration $N)" \
		"$FORWARD_BIN" "$DATABASE_H5" "$SRC_STATUS"

	# assess.jl decides:
	#   exit 0  = continue (writes status_{N+1}.h5)
	#   exit 10 = converged (sets converged=1 on current file)
	set +e
	run_stage "assess.jl (iteration $N)" \
		julia --project="$SCRIPT_DIR" \
		"$SCRIPT_DIR/scripts/assess.jl" "$SRC_STATUS" "$DATABASE_H5"
	ASSESS_EXIT=$?
	set -e

	if [[ $ASSESS_EXIT -eq 10 ]]; then
		echo "[driver] assess.jl signalled converged."
		break
	elif [[ $ASSESS_EXIT -ne 0 ]]; then
		echo "[driver] ERROR: assess.jl failed (exit code $ASSESS_EXIT)." >&2
		exit $ASSESS_EXIT
	fi
done

# ── Stage 5: output ───────────────────────────────────────────────────────────
run_stage "output.jl → output.h5" \
	julia --project="$SCRIPT_DIR" \
	"$SCRIPT_DIR/scripts/output.jl" \
	"$DATABASE_H5" --status-dir "$STATUS_DIR"

echo "[driver] Pipeline complete."
