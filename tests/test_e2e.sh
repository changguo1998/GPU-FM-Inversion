#!/usr/bin/env bash
set -euo pipefail

# test_e2e.sh - Synthetic event end-to-end test
#
# Tests complete pipeline: input → preprocess → (fake misfits) → assess →
# (loop) → output. No GPU or compiled forward binary required.
# Usage: bash tests/test_e2e.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Temp directory for test artifacts
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

DATA_DIR="$TMPDIR/test_event"
mkdir -p "$DATA_DIR"

PASS=0
FAIL=0

pass() {
	echo "  ✓ $1"
	PASS=$((PASS + 1))
}
fail() {
	echo "  ✗ $1"
	FAIL=$((FAIL + 1))
}

echo "=== E2E Test: Synthetic event end-to-end pipeline (temp: ${TMPDIR}) ==="

# Step 1: Generate synthetic data
echo ""
echo "[Step 1] Generating synthetic data ..."
julia --project="$PROJECT_DIR/shared/io" \
	"$PROJECT_DIR/tests/synthetic_data.jl" "$DATA_DIR"
echo "  raw.h5: $(ls -lh "$DATA_DIR/raw.h5" | awk '{print $5}')"
echo "  config.jl: $(wc -l <"$DATA_DIR/config.jl") lines"

# Step 2: input.jl → database.h5 + status_0.h5
echo ""
echo "[Step 2] input.jl → database.h5 + status_0.h5 ..."
julia --project="$PROJECT_DIR" \
	"$PROJECT_DIR/scripts/input.jl" \
	"$DATA_DIR/config.jl"

[[ -f "$DATA_DIR/database.h5" ]] && pass "database.h5 created" ||
	fail "database.h5 missing"

[[ -f "$DATA_DIR/status_0.h5" ]] && pass "status_0.h5 created" ||
	fail "status_0.h5 missing"

# Step 3: preprocess.jl → /trials in status_0.h5
echo ""
echo "[Step 3] preprocess.jl → /trials into status_0.h5 ..."
julia --project="$PROJECT_DIR" \
	"$PROJECT_DIR/scripts/preprocess.jl" \
	"$DATA_DIR/status_0.h5" "$DATA_DIR/database.h5"

N_TRIALS_0=$(julia --project="$PROJECT_DIR/shared/io" -e "
    using HDF5
    h5open(\"$DATA_DIR/status_0.h5\", \"r\") do f
        println(length(read(f[\"trials/strike\"])))
    end
")
echo "  Iter 0 trials: $N_TRIALS_0"
[[ "$N_TRIALS_0" -gt 0 ]] && pass "status_0 has /trials ($N_TRIALS_0 trials)" ||
	fail "status_0 missing /trials"

# Step 4: Inject fake misfits into status_0.h5
echo ""
echo "[Step 4] Injecting fake misfits into status_0.h5 ..."

# Dimensions: 6 phases, 3 stations, 81 trials
# Place best misfit at trial 14 (strike=45,dip=30,rake=20,depth_idx=2,10km)
# so depth refinement narrows to depth 2 only, and best SDR = (45,30,20)
julia --project="$PROJECT_DIR/shared/io" -e '
using HDF5

fname = "'"$DATA_DIR"'/status_0.h5"
n_ph = 6
n_st = 3
n_tr = 81
best = 14   # 0-based: trial 14 (1-based Julia)

# xcorr [6×81]: all 1.0 except best trial
xcorr = fill(1.0, n_ph, n_tr)
xcorr[:, best] .= 0.1

# polarity [3×81]: all 0.5 except best trial
polarity = fill(0.5, n_st, n_tr)
polarity[:, best] .= 0.05

h5open(fname, "r+") do f
    if haskey(f, "misfits")
        delete_object(f, "misfits")
    end
    mg = create_group(f, "misfits")
    write(mg, "xcorr", xcorr)
    write(mg, "polarity", polarity)
end
println("  Written xcorr[$(size(xcorr))], polarity[$(size(polarity))]")
'

has_xcorr=$(julia --project="$PROJECT_DIR/shared/io" -e "
    using HDF5
    h5open(\"$DATA_DIR/status_0.h5\", \"r\") do f
        println(haskey(f, \"misfits/xcorr\"))
    end
")
[[ "$has_xcorr" == "true" ]] && pass "status_0 has /misfits/xcorr" ||
	fail "status_0 missing /misfits/xcorr"

# Step 5: assess.jl (iteration 1, answer "y" to continue)
echo ""
echo "[Step 5] assess.jl (echo y → continue) ..."
echo "y" | julia --project="$PROJECT_DIR" \
	"$PROJECT_DIR/scripts/assess.jl" \
	"$DATA_DIR/status_0.h5" "$DATA_DIR/database.h5"

# After refinement, best was (45,30,20) at depth 2 with all other depths poor
# → depth_indices=[2] only, step sizes halved to 10°, nstrike/ndip/nrake=3

[[ -f "$DATA_DIR/status_1.h5" ]] && pass "status_1.h5 created" ||
	fail "status_1.h5 missing"

CONVERGED_1=$(julia --project="$PROJECT_DIR/shared/io" -e "
    using HDF5
    h5open(\"$DATA_DIR/status_1.h5\", \"r\") do f
        println(read(f[\"strategy/converged\"]))
    end
")
[[ "$CONVERGED_1" == "0" ]] && pass "status_1 converged=0 (continue)" ||
	fail "status_1 converged=$CONVERGED_1, expected 0"

# Verify step sizes decreased
NEW_DSTRIKE=$(julia --project="$PROJECT_DIR/shared/io" -e "
    using HDF5
    h5open(\"$DATA_DIR/status_1.h5\", \"r\") do f
        println(read(f[\"strategy/dstrike\"]))
    end
")
if [[ -n "$NEW_DSTRIKE" ]]; then
	OLD_DSTRIKE=20.0
	# new step should be 10.0 (halved)
	if (($(echo "$NEW_DSTRIKE < $OLD_DSTRIKE" | bc -l))); then
		pass "Step sizes decreased: dstrike=$OLD_DSTRIKE → $NEW_DSTRIKE"
	else
		fail "Step sizes did not decrease: dstrike=$OLD_DSTRIKE → $NEW_DSTRIKE"
	fi
fi

# Step 6: preprocess.jl → /trials in status_1.h5
echo ""
echo "[Step 6] preprocess.jl → /trials into status_1.h5 ..."
julia --project="$PROJECT_DIR" \
	"$PROJECT_DIR/scripts/preprocess.jl" \
	"$DATA_DIR/status_1.h5" "$DATA_DIR/database.h5"

N_TRIALS_1=$(julia --project="$PROJECT_DIR/shared/io" -e "
    using HDF5
    h5open(\"$DATA_DIR/status_1.h5\", \"r\") do f
        println(length(read(f[\"trials/strike\"])))
    end
")
echo "  Iter 1 trials: $N_TRIALS_1"
[[ "$N_TRIALS_1" -gt 0 ]] && pass "status_1 has /trials ($N_TRIALS_1 trials)" ||
	fail "status_1 missing /trials"

# Step 7: Inject fake misfits into status_1.h5
echo ""
echo "[Step 7] Injecting fake misfits into status_1.h5 ..."

# Iter 1: 3×3×3×1×1 = 27 trials (only 1 depth). Best at trial 14 again.
julia --project="$PROJECT_DIR/shared/io" -e '
using HDF5

fname = "'"$DATA_DIR"'/status_1.h5"
n_ph = 6
n_st = 3
n_tr = 27
best = 14   # trial 14 (1-based)

xcorr = fill(1.0, n_ph, n_tr)
xcorr[:, best] .= 0.1

polarity = fill(0.5, n_st, n_tr)
polarity[:, best] .= 0.05

h5open(fname, "r+") do f
    if haskey(f, "misfits")
        delete_object(f, "misfits")
    end
    mg = create_group(f, "misfits")
    write(mg, "xcorr", xcorr)
    write(mg, "polarity", polarity)
end
println("  Written xcorr[$(size(xcorr))], polarity[$(size(polarity))]")
'

pass "Injected misfits into status_1.h5"

# Step 8: assess.jl (iteration 2, answer "N" to stop → converged=1)
echo ""
echo "[Step 8] assess.jl (echo N → converged) ..."
echo "N" | julia --project="$PROJECT_DIR" \
	"$PROJECT_DIR/scripts/assess.jl" \
	"$DATA_DIR/status_1.h5" "$DATA_DIR/database.h5"

[[ -f "$DATA_DIR/status_2.h5" ]] && pass "status_2.h5 created" ||
	fail "status_2.h5 missing"

CONVERGED_2=$(julia --project="$PROJECT_DIR/shared/io" -e "
    using HDF5
    h5open(\"$DATA_DIR/status_2.h5\", \"r\") do f
        println(read(f[\"strategy/converged\"]))
    end
")
[[ "$CONVERGED_2" == "1" ]] && pass "status_2 converged=1 (stopped)" ||
	fail "status_2 converged=$CONVERGED_2, expected 1"

# Step 9: output.jl → output.h5
echo ""
echo "[Step 9] output.jl → output.h5 ..."
julia --project="$PROJECT_DIR" \
	"$PROJECT_DIR/scripts/output.jl" \
	"$DATA_DIR/database.h5" --status-dir "$DATA_DIR"

[[ -f "$DATA_DIR/output.h5" ]] && pass "output.h5 created" ||
	fail "output.h5 missing"

# Step 10: Verify output.h5 structure
echo ""
echo "[Step 10] Verifying output.h5 structure ..."
julia --project="$PROJECT_DIR/shared/io" \
	"$PROJECT_DIR/tests/test_e2e.jl" \
	"$DATA_DIR/output.h5"

# Summary
echo ""
echo "E2E Test Results: ${PASS} passed, ${FAIL} failed"

if [[ ${FAIL} -gt 0 ]]; then
	echo "FAILURE: ${FAIL} check(s) failed."
	exit 1
else
	echo "SUCCESS: All checks passed."
	exit 0
fi
