#!/usr/bin/env bash
# ==============================================================================
# format.sh — Format all source files in the project
#
# Formatters:
#   Julia   → jlfmt        (JuliaFormatter.jl CLI)
#   Bash    → shfmt
#   C++/CUDA → clang-format (LLVM; optional, skips C++ if not found)
#   Markdown → mdformat + markdown-table-formatter (npm)
#
# Usage:
#   bash format.sh              # format all files
#   bash format.sh --check      # dry-run, exit 1 if any file would change
# ==============================================================================
set -euo pipefail

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
	if [[ -z $e ]]; then
		return 0
	fi
	echo "$e: $(command -v "$e")"
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

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
ok() { echo -e "${GREEN}OK${NC} $1"; }
warn() { echo -e "${YELLOW}WARN${NC} $1"; }
fail() { echo -e "${RED}FAIL${NC} $1"; }

errors=0

# ── Formatter helpers ────────────────────────────────────────────────────────

fmt_julia() {
	local f="$1"
	if [[ "$CHECK" == true ]]; then
		if jlfmt --check "$f" &>/dev/null; then return 0; fi
		fail "$f (would be formatted)"
		errors=$((errors + 1))
	else
		local tmp
		tmp=$(mktemp /tmp/jlfmt_XXXXXX)
		jlfmt "$f" 2>/dev/null >"$tmp"
		if [[ -s "$tmp" ]]; then
			mv "$tmp" "$f" && ok "jlfmt  $f"
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
		errors=$((errors + 1))
	else
		shfmt -w "$f" 2>/dev/null && ok "shfmt  $f"
	fi
}

fmt_cxx() {
	local f="$1"
	if [[ "$CHECK" == true ]]; then
		if clang-format --dry-run --Werror "$f" &>/dev/null; then return 0; fi
		fail "$f (would be formatted)"
		errors=$((errors + 1))
	else
		clang-format -i --style=file "$f" 2>/dev/null || clang-format -i --style=LLVM "$f" 2>/dev/null
		ok "clang-format $f"
	fi
}

fmt_md() {
	local f="$1"
	if [[ "$CHECK" == true ]]; then
		if mdformat --check "$f" &>/dev/null && markdown-table-formatter --check "$f" &>/dev/null; then return 0; fi
		fail "$f (would be formatted)"
		errors=$((errors + 1))
	else
		mdformat "$f" 2>/dev/null && ok "mdfmt   $f"
		markdown-table-formatter "$f" 2>/dev/null && ok "mtf     $f"
	fi
}

# ── File discovery ───────────────────────────────────────────────────────────

project_files() {
	local ext="$1"
	find "$SCRIPT_DIR" -type f -name "*$ext" \
		-not -path '*/node_modules/*' \
		-not -path '*/build/*' \
		-not -path '*/.git/*' \
		-not -path '*/Manifest.toml' |
		sort
}

# ── Main ─────────────────────────────────────────────────────────────────────

echo "=== Formatting refactor-fm ==="

if command -v jlfmt &>/dev/null; then
	while IFS= read -r f; do fmt_julia "$f"; done < <(project_files ".jl")
else
	warn "jlfmt not found — skipping Julia"
fi

if command -v shfmt &>/dev/null; then
	while IFS= read -r f; do fmt_bash "$f"; done < <(project_files ".sh")
else
	warn "shfmt not found — skipping shell scripts"
fi

if command -v markdown-table-formatter &>/dev/null; then
	while IFS= read -r f; do fmt_md "$f"; done < <(project_files ".md")
else
	warn "markdown-table-formatter not found — skipping markdown"
fi

if command -v clang-format &>/dev/null; then
	while IFS= read -r f; do fmt_cxx "$f"; done < <(
		project_files ".cpp"
		project_files ".hpp"
		project_files ".h"
		project_files ".cu"
		project_files ".cuh"
	)
else
	warn "clang-format not found — skipping C++/CUDA"
fi

echo ""
if [[ "$CHECK" == true ]]; then
	if [[ $errors -eq 0 ]]; then
		echo "✓ All files formatted."
		exit 0
	else
		echo "✗ $errors file(s) need formatting."
		exit 1
	fi
else
	echo "Formatting complete."
fi
