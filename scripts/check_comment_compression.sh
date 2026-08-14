#!/usr/bin/env bash
# check_comment_compression.sh — verify a docs/comments-only change is a NET
# compression with zero functional-code modification.
#
# Compares `git diff BASE...HEAD` (default BASE = HEAD~1, override with argv[1]),
# classifying every added/removed line as:
#   - removable : comment / docstring / blank line (per a full-file scan of the
#                 old file for "-" lines and the new file for "+" lines), a .md
#                 line, or a code line whose only difference is a trailing
#                 in-line comment (matched code prefix)
#   - functional: anything else  -> fail
# Net compression = removables removed - added.
#
# Exit codes: 0 = verified; 1 = functional change; 2 = no net compression;
# 3 = script/diff error.

set -u

BASE="${1:-HEAD~1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 3

DIFF_FILE="$(mktemp)"
trap 'rm -f "$DIFF_FILE"' EXIT

if ! git diff --quiet "$BASE" -- . \
	':(exclude)forward/build' \
	':(exclude)scripts/check_comment_compression.sh'; then
	if ! git diff --unified=2 "$BASE" -M --no-color -- . \
		':(exclude)forward/build' \
		':(exclude)scripts/check_comment_compression.sh' >"$DIFF_FILE"; then
		echo "check: cannot diff $BASE" >&2
		exit 3
	fi
else
	echo "check: no diff vs $BASE — nothing to verify" >&2
	exit 3
fi

python3 - "$DIFF_FILE" "$BASE" <<'PYEOF'
import re
import subprocess
import sys

diff_file, base = sys.argv[1], sys.argv[2]
REMOVED = 0            # removable lines deleted
ADDED = 0              # removable lines added
FUNC = 0               # functional line changes (must stay 0)
MD_REM = MD_ADD = 0

def mark_comment_lines(lines):
    """Boolean per line: is it part of a comment / docstring / blank? Full-file
    state machine — reliable because it scans real source, not diff hunks."""
    n = len(lines)
    flags = [False] * n
    in_block = in_doc = False
    for i, raw in enumerate(lines):
        t = raw.rstrip("\n").strip()
        if in_block:
            flags[i] = True
            if "*/" in t or "#=" in t:
                in_block = False
            continue
        if in_doc:
            flags[i] = True
            if t.count('"""') >= 1:
                in_doc = t.count('"""') % 2 == 0
            continue
        if t == "":
            flags[i] = True
            continue
        if t.startswith(("/*", "#=")):
            flags[i] = True
            if not t.rstrip().endswith(("*/", "#=")):
                in_block = True
            continue
        if t.startswith('"""'):
            nq = t.count('"""')
            if nq == 1:
                in_doc = True
            flags[i] = True
            continue
        if t.startswith(("//", "#", "*")):
            flags[i] = True
            continue
    return flags

def strip_inline(s):
    """Drop a trailing in-line comment (whitespace + // or #) from a code line."""
    m = re.search(r'\s+(//|#)', s)
    return s[:m.start()].rstrip() if m else s.rstrip()

# resolve per-file comment maps on demand
old_cache, new_cache = {}, {}

def comment_set(path, version="new"):
    """Line numbers (1-based) that are comment/docstring/blank."""
    if path in (old_cache if version == "old" else new_cache):
        return (old_cache if version == "old" else new_cache)[path]
    try:
        if version == "old":
            src = subprocess.run(
                ["git", "show", f"{base}:{path}"], capture_output=True, check=True,
                text=True, cwd=".").stdout
        else:
            src = open(path, encoding="utf-8", errors="replace").read()
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        return None
    flags = mark_comment_lines(src.splitlines())
    s = {i + 1 for i, f in enumerate(flags) if f}
    (old_cache if version == "old" else new_cache)[path] = s
    return s

md = False
cur = None           # (path)
old_no = new_no = None
del_pending, add_pending = [], []   # body lines awaiting pairing

def flush(del_set, add_set, path):
    """Match code lines whose only difference is a trailing in-line comment."""
    global REMOVED, ADDED, FUNC
    unmatched_add = list(add_set)
    for d in del_set:
        sig_d = strip_inline(d).strip()
        matched = False
        if sig_d:
            for i, a in enumerate(unmatched_add):
                if strip_inline(a).strip() == sig_d:
                    REMOVED += 1
                    ADDED += 1
                    unmatched_add.pop(i)
                    matched = True
                    break
        if not matched:
            FUNC += 1
    FUNC += len(unmatched_add)

with open(diff_file) as f:
    for line in f:
        if line.startswith("diff --git"):
            flush(del_pending, add_pending, cur)
            del_pending, add_pending = [], []
            m = re.search(r" b/([^ \t]+)$", line.rstrip())
            cur = m.group(1) if m else ""
            md = cur.endswith((".md", ".MD"))
            old_no = new_no = None
            continue
        h = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if h:
            old_no, new_no = int(h.group(1)), int(h.group(2))
            continue
        if line.startswith(("---", "+++", "index ")):
            continue
        if line.startswith(" "):
            # context line: counts in neither file; only advances numbering
            if old_no is not None:
                old_no += 1
            if new_no is not None:
                new_no += 1
            continue
        if not line.startswith(("+", "-")):
            continue
        sign, body = line[0], line[1:]
        if md:
            if sign == "-": MD_REM += 1
            else: MD_ADD += 1
            if sign == "-":
                old_no += 1
            else:
                new_no += 1
            continue
        version, no = ("old", old_no) if sign == "-" else ("new", new_no)
        cmtset = comment_set(cur, version)
        if cmtset is not None and no in cmtset:
            if sign == "-": REMOVED += 1
            else: ADDED += 1
        elif sign == "-":
            del_pending.append(body)
        else:
            add_pending.append(body)
        if sign == "-":
            old_no += 1
        else:
            new_no += 1
    flush(del_pending, add_pending, cur)

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
