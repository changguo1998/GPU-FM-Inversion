#!/usr/bin/env bash
# check_comment_compression.sh — verify a docs/comments-only change is a NET
# compression with zero functional-code modification.
#
# Compares `git diff BASE...HEAD` (default BASE = HEAD~1, override with argv[1]).
# For source files (.jl/.cpp/.h/.sh): lines removed/added inside comments,
# docstrings and blank lines count toward the "removable" pool; any other
# removed/added line is a functional change (fail).  .md lines all count
# toward the pool.  Net compression = removables removed - added.
#
# Exit codes: 0 = verified; 1 = functional change; 2 = no net compression;
# 3 = script/diff error.

set -u

BASE="${1:-HEAD~1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 3

DIFF_FILE="$(mktemp)"
trap 'rm -f "$DIFF_FILE"' EXIT

if ! git diff --quiet "$BASE" -- . ':(exclude)forward/build' ':(exclude)scripts/check_comment_compression.sh'; then
    if ! git diff --unified=0 "$BASE" -M --no-color -- . ':(exclude)forward/build' ':(exclude)scripts/check_comment_compression.sh' > "$DIFF_FILE"; then
        echo "check: cannot diff $BASE" >&2
        exit 3
    fi
else
    echo "check: no diff vs $BASE — nothing to verify" >&2
    exit 3
fi

python3 - "$DIFF_FILE" "$BASE" <<'PYEOF'
import re
import sys

diff_file, base = sys.argv[1], sys.argv[2]
REMOVED = 0     # removable lines (comments/docstrings/blank) deleted
ADDED = 0       # removable lines added
FUNC = 0        # functional lines changed (must be 0)
MD_REM = MD_ADD = 0
md = False
in_block = False   # C /* */ or Julia #= =#
in_doc = False     # Julia """ ... """

def removable(src):
    """True if src (trimmed line body) is comment/docstring/blank."""
    global in_block, in_doc
    if src == "":
        return True
    if md:
        return True
    t = src.lstrip()
    if in_block:
        if "*/" in t or "#=" in t:
            in_block = False
        return True
    if in_doc:
        if t.count('"""') >= 1:
            in_doc = t.count('"""') % 2 == 0
        return True
    if t.startswith("/*") or t.startswith("#="):
        if not (t.rstrip().endswith("*/") or t.rstrip().endswith("#=")):
            in_block = True
        return True
    if t.startswith('"""'):
        n = t.count('"""')
        if n == 1:
            in_doc = True
        return True
    if t.startswith("//") or t.startswith("#") or t.startswith("*"):
        return True
    return False

with open(diff_file) as f:
    for line in f:
        if line.startswith("diff --git"):
            m = re.search(r" b/([^ \t]+)$", line.rstrip())
            fname = m.group(1) if m else ""
            md = fname.endswith(".md") or fname.endswith(".MD")
            in_block = in_doc = False
            continue
        if line.startswith(("---", "+++", "index ", "@@")):
            continue
        if not line.startswith(("+", "-")):
            continue
        sign, body = line[0], line[1:]
        if md:
            if sign == "-": MD_REM += 1
            else: MD_ADD += 1
        elif removable(body):
            if sign == "-": REMOVED += 1
            else: ADDED += 1
        else:
            FUNC += 1

net = (REMOVED - ADDED) + (MD_REM - MD_ADD)
print(f"removable lines removed={REMOVED} added={ADDED} (net {REMOVED - ADDED})")
print(f"doc lines removed={MD_REM} added={MD_ADD} (net {MD_REM - MD_ADD})")
print(f"net comment/doc compression = {net} (vs {base})")
if FUNC:
    print(f"FATAL: {FUNC} functional line(s) changed — not a docs-only edit")
    sys.exit(1)
if net <= 0:
    print("WARN: no net compression — nothing verified")
    sys.exit(2)
print("no functional changes")
sys.exit(0)
PYEOF