#!/usr/bin/env bash
#
# tests/run_cross_lang.sh
#
# Orchestrates all cross‑language verification tests.
# Compiles C++ binaries if needed, runs Julia test orchestrator,
# and reports PASS / SKIP / FAIL for each test.
#
# Usage:  ./tests/run_cross_lang.sh
#
# Requirements: C++17 compiler, HDF5 development libraries.
# If either is missing, C++ tests are SKIP'd gracefully.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORWARD_DIR="$PROJECT_ROOT/forward"
TESTS_DIR="$FORWARD_DIR/tests"

# ── colours ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # no colour

pass() { echo -e "${GREEN}PASS${NC} — $*"; }
fail() { echo -e "${RED}FAIL${NC} — $*"; }
skip() { echo -e "${YELLOW}SKIP${NC} — $*"; }
info() { echo -e "INFO — $*"; }

failures=0
passes=0
skips=0

record_pass() { passes=$((passes + 1)); }
record_fail() { failures=$((failures + 1)); }
record_skip() { skips=$((skips + 1)); }

# ── helpers ──────────────────────────────────────────────
have_command() { command -v "$1" &>/dev/null; }

have_hdf5_cflags() {
	if have_command pkg-config && pkg-config --cflags hdf5 &>/dev/null; then
		return 0
	fi
	# fallback: try to find hdf5.h
	if [ -f /usr/include/hdf5.h ] || [ -f /usr/include/hdf5/serial/hdf5.h ]; then
		return 0
	fi
	return 1
}

get_cxx() {
	if have_command g++; then
		echo "g++"
		return 0
	fi
	if have_command clang++; then
		echo "clang++"
		return 0
	fi
	echo ""
}

get_cxxflags() {
	local flags="-std=c++17 -O0 -g -Wall -Wextra"
	# HDF5 include paths
	if have_command pkg-config && pkg-config --cflags hdf5 &>/dev/null; then
		flags="$flags $(pkg-config --cflags hdf5)"
	fi
	echo "$flags"
}

get_ldflags() {
	local ld="-lm"
	if have_command pkg-config && pkg-config --libs hdf5 &>/dev/null; then
		ld="$ld $(pkg-config --libs hdf5)"
	else
		ld="$ld -lhdf5"
	fi
	echo "$ld"
}

# ── compile test_cross_lang ─────────────────────────────
compile_cross_lang() {
	local src="$FORWARD_DIR/tests/test_cross_lang.cpp"
	local mt_utils="$FORWARD_DIR/src/mt_utils.cpp"
	local hdf5_io="$FORWARD_DIR/src/hdf5_io.cpp"
	local binary="$TESTS_DIR/test_cross_lang"

	local cxx
	cxx=$(get_cxx)
	if [ -z "$cxx" ]; then
		skip "no C++ compiler found (g++ or clang++). C++ cross‑lang tests will be SKIP'd."
		return 1
	fi

	if ! have_hdf5_cflags; then
		skip "HDF5 development headers not found. C++ cross‑lang tests will be SKIP'd."
		return 1
	fi

	local cxxflags
	cxxflags=$(get_cxxflags)
	local ldflags
	ldflags=$(get_ldflags)

	info "compiling test_cross_lang…"
	$cxx $cxxflags -I"$FORWARD_DIR/src" -o "$binary" \
		"$src" "$mt_utils" "$hdf5_io" \
		$ldflags

	if [ $? -ne 0 ]; then
		fail "compilation of test_cross_lang"
		return 1
	fi

	info "test_cross_lang compiled → $binary"
	return 0
}

# ── run Julia cross‑lang test ──────────────────────────
run_julia_cross_lang() {
	local jl_script="$SCRIPT_DIR/test_cross_lang.jl"

	if ! have_command julia; then
		skip "julia not found. Cross‑lang tests will be SKIP'd."
		record_skip
		return
	fi

	info "running Julia cross‑language test suite…"
	echo ""

	# Uses MT (shared/mt) and HDF5 (stdlib-ish)
	if julia "$jl_script"; then
		pass "Julia cross‑language test suite"
		record_pass
	else
		fail "Julia cross‑language test suite"
		record_fail
	fi
}

# ════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════"
echo "  Cross‑language verification suite (T22)"
echo "═══════════════════════════════════════════════"
echo ""

# Step 1 — compile.  If this cannot be done then the Julia
#          script will SKIP each C++‑backed check gracefully.
if compile_cross_lang; then
	record_pass
else
	record_skip
fi

echo ""

# Step 2 — run Julia orchestrator (which calls into C++).
run_julia_cross_lang

echo ""
echo "═══════════════════════════════════════════════"
echo -n "  Summary: "
echo -e "${GREEN}${passes} passed${NC}  ${YELLOW}${skips} skipped${NC}  ${RED}${failures} failed${NC}"
echo "═══════════════════════════════════════════════"

if [ "$failures" -gt 0 ]; then
	exit 1
fi
exit 0
