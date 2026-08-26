#!/usr/bin/env bash
# Test driver for fmri_decomposition stages 2 and 3.
#
#   ./run_tests.sh              # unit tests + end-to-end smoke run
#   ./run_tests.sh --unit       # unit tests only (fast, no fixture)
#   ./run_tests.sh --smoke      # end-to-end fixture pipeline only
#   ./run_tests.sh --no-venv    # use the current interpreter as-is
#   ./run_tests.sh --keep       # keep the fixture directory for inspection
#
# Inside an Apptainer/Singularity container this auto-detects and skips both
# venv creation and `pip install -e .` (site-packages is read-only there);
# the repo is imported via PYTHONPATH instead. Run from the repo root.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RUN_UNIT=1
RUN_SMOKE=1
USE_VENV=1
DO_INSTALL=1
KEEP_FIXTURE=0
FIXTURE_DIR="${FIXTURE_DIR:-$HERE/.fixture}"

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; }

for arg in "$@"; do
  case "$arg" in
    --unit)    RUN_SMOKE=0 ;;
    --smoke)   RUN_UNIT=0 ;;
    --no-venv) USE_VENV=0 ;;
    --keep)    KEEP_FIXTURE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# joblib handles concurrency; BLAS threads inside workers would oversubscribe.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Keep nilearn's atlas cache out of $HOME on shared filesystems. Respects an
# existing value -- the container pre-populates /opt/nilearn_data at build.
export NILEARN_DATA="${NILEARN_DATA:-$HERE/.nilearn_data}"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\033[31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

# /usr/bin/time -v reports max RSS across the process AND its reaped children.
# Python's resource.getrusage(RUSAGE_SELF) would miss the joblib workers,
# which is where the memory actually goes.
TIME_BIN="$(command -v /usr/bin/time || true)"
timed() {
  local label="$1"; shift
  if [[ -n "$TIME_BIN" ]]; then
    "$TIME_BIN" -f "  [${label}] wall %e s | maxRSS %M KB | cpu %P" "$@"
  else
    "$@"
  fi
}

# ------------------------------------------------------------- container ---
if [[ -n "${FMRI_DECOMP_CONTAINER:-}${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]]; then
  USE_VENV=0
  DO_INSTALL=0
  export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
  echo "  (container detected: using image environment, no venv / no pip)"
fi

# --------------------------------------------------------------- python ---
if [[ "$USE_VENV" -eq 1 ]]; then
  # Test for the activate script, not the directory: a half-built .venv would
  # otherwise pass the check and fail on `source`.
  if [[ ! -f .venv/bin/activate ]]; then
    say "creating .venv"
    rm -rf .venv
    python3 -m venv .venv || fail "venv creation failed"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PY="$(command -v python3)"
say "python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

if [[ "$DO_INSTALL" -eq 1 ]]; then
  say "installing package (editable) + test extras"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -e ".[test]" || fail "pip install failed"
  "$PY" -c 'import nilearn' 2>/dev/null || "$PY" -m pip install --quiet nilearn || \
    echo "  (nilearn unavailable -- built-in atlas fetch will be skipped)"
fi

# Confirm we are testing the working tree, not something baked into the image.
say "import check"
"$PY" - <<'PYEOF'
import fmri_decomposition as m
import os, sys
print("  fmri_decomposition:", m.__file__)
if "site-packages" in (m.__file__ or ""):
    print("  WARNING: imported from site-packages, NOT your working tree", file=sys.stderr)
PYEOF

# ----------------------------------------------------------- unit tests ---
if [[ "$RUN_UNIT" -eq 1 ]]; then
  say "unit tests"
  "$PY" -m pytest tests -v --tb=short || fail "unit tests failed"
fi

# ------------------------------------------------------- end-to-end run ---
if [[ "$RUN_SMOKE" -eq 1 ]]; then
  say "building synthetic cohort at $FIXTURE_DIR"
  rm -rf "$FIXTURE_DIR"
  "$PY" -m fmri_decomposition.cli fixture "$FIXTURE_DIR" --n-subs 4 --n-tr 240

  CFG="$FIXTURE_DIR/config.yaml"
  RERUN_LOG="$FIXTURE_DIR/rerun.log"

  say "validate"
  "$PY" -m fmri_decomposition.cli validate "$CFG" || fail "validation failed"

  say "stage 2: extract"
  timed extract "$PY" -m fmri_decomposition.cli extract "$CFG" --n-jobs 2 \
    || fail "extract failed"

  say "stage 3: dfc"
  timed dfc "$PY" -m fmri_decomposition.cli dfc "$CFG" --n-jobs 2 \
    || fail "dfc failed"

  say "stage 3 again (must be a no-op)"
  "$PY" -m fmri_decomposition.cli dfc "$CFG" --n-jobs 2 | tee "$RERUN_LOG"
  grep -q "skipped=" "$RERUN_LOG" || fail "re-run did not skip existing shards"

  say "diagnostics"
  "$PY" -m fmri_decomposition.cli diagnose "$CFG" || echo "  (diagnostics reported issues)"

  say "output tree"
  find "$FIXTURE_DIR/outputs" -name '*.parquet' | head -20 || true
  echo "..."
  echo "  shards: $(find "$FIXTURE_DIR/outputs" -name '*.parquet' | wc -l)"
  echo "  stray tmp files: $(find "$FIXTURE_DIR/outputs" -name '*.tmp.*' | wc -l)"

  # 240 TR at TR=1 is shorter than a 300 s window, so that partition must be
  # empty. This is the run-shorter-than-window path: it should log and skip,
  # not crash and not emit a malformed shard.
  say "short-run guard (240 TR < 300 s window)"
  n300=$(find "$FIXTURE_DIR/outputs/dfc" -path '*window_s=300*' -name '*.parquet' 2>/dev/null | wc -l)
  echo "  window_s=300 shards: $n300 (expected 0)"
  [[ "$n300" -eq 0 ]] || fail "300 s windows emitted from a 240 TR run"

  say "inspect one DFC shard"
  "$PY" - "$FIXTURE_DIR" <<'PYEOF'
import sys, glob
import pandas as pd
import pyarrow.parquet as pq

root = sys.argv[1]
hits = sorted(glob.glob(f"{root}/outputs/dfc/**/window_s=30/**/*.parquet", recursive=True))
if not hits:
    sys.exit("no window_s=30 shard found")
path = hits[0]
t = pq.read_table(path)
meta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
df = t.to_pandas()
print(f"{path}\n  rows={len(df)} cols={len(df.columns)}")
print("  metadata:", {k: meta[k] for k in ("atlas", "n_edges", "edge_storage",
                                           "estimator", "window_s") if k in meta})
qc = ["window_id", "stimulus_start_s", "n_tr_nominal", "n_tr_effective",
      "frac_good_frames", "rank_deficient"]
print(df[qc].head(5).to_string(index=False))
edges = [c for c in df.columns if "__" in c]
print(f"  edges: {len(edges)} e.g. {edges[:3]}")
arr = df[edges].to_numpy()
print(f"  r range: [{arr.min():.3f}, {arr.max():.3f}]")
assert df["n_tr_effective"].le(df["n_tr_available"]).all()
assert arr.min() >= -1.0 and arr.max() <= 1.0
print("  OK")
PYEOF

  say "resource summary"
  echo "  fixture: 4 subs x 240 TR, SYNTHETIC -- not a basis for cluster sizing."
  echo "  Real cost is dominated by gzip decompression of the 4D nii.gz and"
  echo "  atlas resampling; profile one real subject for --mem and --time."
  echo "  n-jobs=2 -> --mem must cover 2x per-worker peak, not the total above."
  du -sh "$FIXTURE_DIR/outputs" 2>/dev/null || true
  find "$FIXTURE_DIR/outputs" -name '*.parquet' -printf '%s\n' 2>/dev/null \
    | awk '{s+=$1; n++} END {if (n) printf "  %d shards, mean %.1f KB\n", n, s/n/1024}'

  if [[ "$KEEP_FIXTURE" -eq 0 ]]; then
    rm -rf "$FIXTURE_DIR"
  else
    say "fixture kept at $FIXTURE_DIR"
  fi
fi

say "ALL CHECKS PASSED"
