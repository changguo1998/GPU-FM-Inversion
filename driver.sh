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

help() {
	echo "Usage: bash driver.sh --data-dir <dir>"
}

# ── Defaults ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CMD_CALL_JULIA="julia --project=$SCRIPT_DIR"
CALL_INPUT="$CMD_CALL_JULIA $SCRIPT_DIR/scripts/input.jl"
CALL_PREPROCESS="$CMD_CALL_JULIA $SCRIPT_DIR/scripts/preprocess.jl"
CALL_FORWARD="$SCRIPT_DIR/forward/build/forward"
CALL_ASSESS="$CMD_CALL_JULIA $SCRIPT_DIR/scripts/assess.jl"
CALL_OUTPUT="$CMD_CALL_JULIA $SCRIPT_DIR/scripts/output.jl"

# ── Parse CLI ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
	case "$1" in
	--data-dir)
		DATA_DIR="$2"
		shift 2
		;;
	-h | --help)
		help
		exit 0
		;;
	*)
		help
		exit 1
		;;
	esac
done

CONFIG_FILE="$DATA_DIR/config.jl"
LOG_FILE="$DATA_DIR/driver.log"
ASSESS_DECISION_FILE="$DATA_DIR/.decision.txt"
DATABASE_H5="$DATA_DIR/database.h5"
STATUS_DIR="$DATA_DIR/status"

# logging functions
if [[ -t 1 && -z ${NO_COLOR:-} ]]; then
	_WHITE=$'\033[0;37m'
	_YELLOW=$'\033[0;33m'
	_RED=$'\033[0;31m'
	_END=$'\033[0m'
else
	_WHITE=""
	_YELLOW=""
	_RED=""
	_END=""
fi

_msg() {
	local level color txt
	level="$1"
	color="$2"
	shift 2
	txt="$*"
	if [[ -n ${LOG_FILE:-} ]]; then
		if [[ -f $LOG_FILE ]]; then
			printf '[%s] %s%s: %s%s\n' "$(date '+%F %T')" "$color" "$level" "$txt" "$_END" | tee -a "$LOG_FILE"
		fi
	else
		printf '[driver %s] %s%s: %s%s\n' "$(date '+%F %T')" "$color" "$level" "$txt" "$_END"
	fi
}

info() {
	_msg "INFO" "$_WHITE" "$@"
}

warn() {
	_msg "WARN" "$_YELLOW" "$@"
}

error() {
	_msg " ERR" "$_RED" "$@" >&2
}

if [[ -z "${DATA_DIR:-}" ]]; then
	error "--data-dir is required"
	exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
	error "data directory not found: $DATA_DIR"
	exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
	error "config file not found: $CONFIG_FILE"
	exit 1
fi

if [[ -f "$DATABASE_H5" ]]; then
	warn "database.h5 already exists"
fi

# ==============================================================================
# Main
# ==============================================================================

# ── Stage 1: input (once) ─────────────────────────────────────────────────────
mkdir -p "$STATUS_DIR"
: >"$LOG_FILE"
: >"$ASSESS_DECISION_FILE"
info "input"
$CALL_INPUT "$CONFIG_FILE"
if [[ -f "$DATA_DIR/status_0.h5" ]]; then
	mv "$DATA_DIR/status_0.h5" "$STATUS_DIR/"
else
	error "failed to generate status_0.h5"
	exit 1
fi

iteration=1
# ── Loop: preprocess → forward → assess ───────────────────────────────────────
while true; do
	if [[ ! -f "$ASSESS_DECISION_FILE" ]]; then
		break
	fi
	DECISION="$(cat "$ASSESS_DECISION_FILE")"
	if [[ -z $DECISION ]]; then
		break
	fi

	info "($iteration) preprocess"
	$CALL_PREPROCESS

	info "($iteration) misfit"
	$CALL_FORWARD

	info "($iteration) assess"
	$CALL_ASSESS

	((iteration += 1))
done

# ── Stage 5: output ───────────────────────────────────────────────────────────
info "output"
$CALL_OUTPUT

info "complete"
