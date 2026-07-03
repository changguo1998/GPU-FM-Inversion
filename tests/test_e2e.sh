#!/usr/bin/env bash
set -euo pipefail

# test_e2e.sh — Synthetic event end-to-end test
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
cp "$PROJECT_DIR/examples/synthetic/config.jl" "$DATA_DIR/config.jl"
echo "  stations: $(ls -1 "$DATA_DIR"/*.dat | wc -l) waveform files"
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

# Step 4: Inject fake misfits into status_0.h5 (dynamic trial count)
echo ""
echo "[Step 4] Injecting fake misfits into status_0.h5 ..."

julia --project="$PROJECT_DIR/shared/io" -e '
using HDF5

fname = "'"$DATA_DIR"'/status_0.h5"

h5open(fname, "r") do f
    global n_tr  = length(read(f["trials/strike"]))
    global n_ph  = length(read(f["strategy/xcorr_phase_mask"]))
    global n_st  = length(read(f["strategy/polarity_channel_mask"]))
end

# Best misfit at middle trial so refinement has a clear minimum
best = Int(floor(n_tr / 2))

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
OLD_DSTRIKE=$(julia --project="$PROJECT_DIR/shared/io" -e "
    using HDF5
    h5open(\"$DATA_DIR/status_0.h5\", \"r\") do f
        println(read(f[\"strategy/dstrike\"]))
    end
")
if [[ -n "$NEW_DSTRIKE" && -n "$OLD_DSTRIKE" ]]; then
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

# Step 7: Inject fake misfits into status_1.h5 (dynamic trial count)
echo ""
echo "[Step 7] Injecting fake misfits into status_1.h5 ..."

julia --project="$PROJECT_DIR/shared/io" -e '
using HDF5

fname = "'"$DATA_DIR"'/status_1.h5"

h5open(fname, "r") do f
    global n_tr  = length(read(f["trials/strike"]))
    global n_ph  = length(read(f["strategy/xcorr_phase_mask"]))
    global n_st  = length(read(f["strategy/polarity_channel_mask"]))
end

best = Int(floor(n_tr / 2))

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
set +e
echo "N" | julia --project="$PROJECT_DIR" \
	"$PROJECT_DIR/scripts/assess.jl" \
	"$DATA_DIR/status_1.h5" "$DATA_DIR/database.h5"
ASSESS_EXIT=$?
set -e
[[ "$ASSESS_EXIT" -eq 10 ]] && pass "assess exited with 10 (converged)" ||
	fail "assess exited with $ASSESS_EXIT, expected 10"

# When converged, assess writes converged=1 to status_1.h5 (same file), not a new status file
CONVERGED_FINAL=$(julia --project="$PROJECT_DIR/shared/io" -e "
    using HDF5
    h5open(\"$DATA_DIR/status_1.h5\", \"r\") do f
        println(read(f[\"strategy/converged\"]))
    end
")
[[ "$CONVERGED_FINAL" == "1" ]] && pass "status_1 converged=1 (stopped)" ||
	fail "status_1 converged=$CONVERGED_FINAL, expected 1"

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
