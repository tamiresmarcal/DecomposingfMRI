#!/usr/bin/env bash
# Test driver for fmri_decomposition stages 2 and 3.
#
#   ./run_tests.sh              # venv + unit tests + end-to-end smoke run
#   ./run_tests.sh --unit       # unit tests only (fast, no fixture)
#   ./run_tests.sh --smoke      # end-to-end fixture pipeline only
#   ./run_tests.sh --no-venv    # use the current interpreter as-is
#   ./run_tests.sh --keep       # keep the fixture directory for inspection
#
# On a cluster: module load python first, then run with --no-venv inside your
# own environment, or let this script build .venv locally.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RUN_UNIT=1
RUN_SMOKE=1
USE_VENV=1
KEEP_FIXTURE=0
FIXTURE_DIR="${FIXTURE_DIR:-$HERE/.fixture}"

for arg in "$@"; do
  case "$arg" in
    --unit)    RUN_SMOKE=0 ;;
    --smoke)   RUN_UNIT=0 ;;
    --no-venv) USE_VENV=0 ;;
    --keep)    KEEP_FIXTURE=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# joblib handles concurrency; BLAS threads inside workers would oversubscribe.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Keep nilearn's atlas cache out of $HOME on shared filesystems.
export NILEARN_DATA="${NILEARN_DATA:-$HERE/.nilearn_data}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\033[31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------- python ---
if [[ "$USE_VENV" -eq 1 ]]; then
  if [[ ! -d .venv ]]; then
    say "creating .venv"
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
PY="$(command -v python3)"
say "python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

say "installing package (editable) + test extras"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e ".[test]" || fail "pip install failed"
# nilearn is optional: only the built-in atlas fetchers need it.
"$PY" -c 'import nilearn' 2>/dev/null || "$PY" -m pip install --quiet nilearn || \
  echo "  (nilearn unavailable -- built-in atlas fetch will be skipped)"

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

  say "validate"
  "$PY" -m fmri_decomposition.cli validate "$CFG" || fail "validation failed"

  say "stage 2: extract"
  "$PY" -m fmri_decomposition.cli extract "$CFG" --n-jobs 2 || fail "extract failed"

  say "stage 3: dfc"
  "$PY" -m fmri_decomposition.cli dfc "$CFG" --n-jobs 2 || fail "dfc failed"

  say "stage 3 again (must be a no-op)"
  "$PY" -m fmri_decomposition.cli dfc "$CFG" --n-jobs 2 | tee /tmp/rerun.log
  grep -q "skipped=" /tmp/rerun.log || fail "re-run did not skip existing shards"

  say "diagnostics"
  "$PY" -m fmri_decomposition.cli diagnose "$CFG" || echo "  (diagnostics reported issues)"

  say "output tree"
  find "$FIXTURE_DIR/outputs" -name '*.parquet' | head -20
  echo "..."
  echo "shards: $(find "$FIXTURE_DIR/outputs" -name '*.parquet' | wc -l)"
  echo "stray tmp files: $(find "$FIXTURE_DIR/outputs" -name '*.tmp.*' | wc -l)"

  say "inspect one DFC shard"
  "$PY" - "$FIXTURE_DIR" <<'PYEOF'
import sys, glob
import pandas as pd
import pyarrow.parquet as pq

root = sys.argv[1]
path = sorted(glob.glob(f"{root}/outputs/dfc/**/window_s=30/**/*.parquet", recursive=True))[0]
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
print(f"  r range: [{df[edges].to_numpy().min():.3f}, {df[edges].to_numpy().max():.3f}]")
assert df["n_tr_effective"].le(df["n_tr_available"]).all()
assert df[edges].to_numpy().min() >= -1.0 and df[edges].to_numpy().max() <= 1.0
print("  OK")
PYEOF

  if [[ "$KEEP_FIXTURE" -eq 0 ]]; then
    rm -rf "$FIXTURE_DIR"
  else
    say "fixture kept at $FIXTURE_DIR"
  fi
fi

say "ALL CHECKS PASSED"
