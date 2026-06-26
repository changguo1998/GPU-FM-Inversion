#!/usr/bin/env bash
set -euo pipefail

# format.sh - Format all source files in the project (parallel)
#
# Formatters:
#   Julia   → jlfmt        (JuliaFormatter.jl CLI)
#   Bash    → shfmt
#   C++/CUDA → clang-format (LLVM; optional, skips C++ if not found)
#   Markdown → mdformat + markdown-table-formatter (npm)
# Usage:
#   bash format.sh              # format all files (parallel)
#   bash format.sh --check      # dry-run, exit 1 if any file would change
#   JOBS=8 bash format.sh       # override parallelism (default: nproc)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CHECK=false

while [[ $# -gt 0 ]]; do
	case "$1" in
	--check)
		CHECK=true
		shift
		;;
	*)
		echo "Usage: bash format.sh [--check]"
		exit 1
		;;
	esac
done

source ~/.bashrc
echo "spack setup file: $SPACK_SETUP_ENV"
source "$SPACK_SETUP_ENV"

checkexe() {
	local e
	e="$1"
	if [[ -z "${e}" ]]; then
		return 0
	fi
	echo "${e}: $(command -v "${e}")"
}
checkexe jlfmt
checkexe shfmt
checkexe mdformat
checkexe fnm
checkexe markdown-table-formatter

if ! command -v clang-format >/dev/null 2>&1; then
	spack load llvm
fi
checkexe clang-format

# Fail counter: race-free via marker files in temp dir (one per failing file)
_faildir="$(mktemp -d)"
trap 'rm -rf "$_faildir"' EXIT

mark_fail() { touch "${_faildir}/$(printf '%s' "$1" | tr '/' '_')"; }

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'
warn() { echo -e "${YELLOW}WARN${NC} $1"; }
fail() { echo -e "${RED}FAIL${NC} $1"; }

# Formatter helpers (silent on success; fail + mark on check failure)
fmt_julia() {
	local f="$1"
	if [[ "$CHECK" == true ]]; then
		if jlfmt --check "$f" &>/dev/null; then return 0; fi
		fail "$f (would be formatted)"
		mark_fail "$f"
	else
		local tmp
		tmp=$(mktemp /tmp/jlfmt_XXXXXX)
		jlfmt "$f" 2>/dev/null >"$tmp"
		if [[ -s "$tmp" ]]; then
			mv "$tmp" "$f"
		else
			rm -f "$tmp"
		fi
	fi
}

fmt_bash() {
	local f="$1"
	if [[ "$CHECK" == true ]]; then
		if shfmt -d "$f" &>/dev/null; then return 0; fi
		fail "$f (would be formatted)"
		mark_fail "$f"
	else
		shfmt -w "$f" 2>/dev/null || true
	fi
}

fmt_cxx() {
	local f="$1"
	if [[ "$CHECK" == true ]]; then
		if clang-format --dry-run --Werror "$f" &>/dev/null; then return 0; fi
		fail "$f (would be formatted)"
		mark_fail "$f"
	else
		clang-format -i --style=file "$f" 2>/dev/null || clang-format -i --style=LLVM "$f" 2>/dev/null || true
	fi
}

fmt_md() {
	local f="$1"
	if [[ "$CHECK" == true ]]; then
		if mdformat --check "$f" &>/dev/null; then return 0; fi
		fail "$f (would be formatted)"
		mark_fail "$f"
	else
		markdown-table-formatter "$f" 2>/dev/null || true
		mdformat "$f" 2>/dev/null || true
	fi
}

# Export for parallel xargs execution
export -f fmt_julia fmt_bash fmt_cxx fmt_md mark_fail warn fail
export CHECK _faildir RED YELLOW NC

# File discovery
project_files() {
	local ext="$1"
	find "$SCRIPT_DIR" -type f -name "*$ext" \
		-not -path '*/node_modules/*' \
		-not -path '*/build/*' \
		-not -path '*/.git/*' \
		-not -path '*/Manifest.toml' |
		sort
}

# Parallelism
JOBS="${JOBS:-$(nproc)}"

# Main
echo "=== Formatting refactor-fm (parallel, ${JOBS} jobs) ==="

if command -v jlfmt &>/dev/null; then
	project_files ".jl" | xargs -r -P "$JOBS" -n 1 bash -c 'fmt_julia "$1"' _
else
	warn "jlfmt not found — skipping Julia"
fi

if command -v shfmt &>/dev/null; then
	project_files ".sh" | xargs -r -P "$JOBS" -n 1 bash -c 'fmt_bash "$1"' _
else
	warn "shfmt not found — skipping shell scripts"
fi

if command -v mdformat &>/dev/null; then
	project_files ".md" | xargs -r -P "$JOBS" -n 1 bash -c 'fmt_md "$1"' _
else
	warn "mdformat not found — skipping markdown"
fi

if command -v clang-format &>/dev/null; then
	{
		project_files ".cpp"
		project_files ".hpp"
		project_files ".h"
		project_files ".cu"
		project_files ".cuh"
	} | xargs -r -P "$JOBS" -n 1 bash -c 'fmt_cxx "$1"' _
else
	warn "clang-format not found — skipping C++/CUDA"
fi

# Summary
echo ""
if [[ "$CHECK" == true ]]; then
	_errors=$(find "$_faildir" -type f | wc -l)
	if [[ ${_errors} -eq 0 ]]; then
		echo "✓ All files formatted."
		exit 0
	else
		echo "✗ ${_errors} file(s) need formatting."
		exit 1
	fi
else
	echo "Formatting complete."
fi
