#!/usr/bin/env bash
# Reproduction driver for the educational event-release study.
#
#   ./reproduce.sh verify             check every frozen headline number (seconds)
#   ./reproduce.sh checksums          verify SHA256SUMS.txt against the working tree
#   ./reproduce.sh postprocess        recompute bootstrap and NC decomposition from frozen inputs
#   ./reproduce.sh eedi-unseal        unseal the frozen Eedi confirmatory payload
#   ./reproduce.sh assist-integrity   verify the sealed ASSISTments bundle
#   ./reproduce.sh data               rebuild the ASSISTments splits from the raw archive
#   ./reproduce.sh stages             run every analysis stage against the frozen checkpoints
#   ./reproduce.sh stage <name>       run one stage (see STAGES below)
#   ./reproduce.sh retrain-primary    retrain the primary chronological GRU (overwrites the frozen checkpoint)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
MODE="${1:-verify}"

# Ordered analysis stages. Later stages read score caches written by earlier ones.
STAGES=(
  eedi/support_sensitivity
  assistments/capacity_certification
  assistments/temporal_forensics
  assistments/chronological_protocol
  assistments/interval_nesting
  assistments/matched_information
  assistments/seed_robustness
  assistments/attention_consumer
  assistments/threshold_false_stable
  assistments/release_field_ablation
  assistments/support_intersection
  assistments/matched_estimand_controls
  assistments/kt_baselines
  assistments/decision_baselines
  assistments/support_predicate_robustness
  assistments/fingerprint_consistency
  assistments/certificate_reliability
  assistments/behavioral_plausibility
  assistments/bootstrap_capacity
)

echo "==> Educational event-release reproducibility"
echo "    mode: $MODE"
echo "    root: $ROOT"

"$PY" - <<'PY'
import importlib
mods = ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch", "yaml", "cryptography"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit("Missing dependencies: " + ", ".join(missing) + "\nInstall with: pip install -r requirements.txt")
print("Dependencies OK")
PY

mkdir -p outputs

need_archive() {
  if [ ! -f data/raw/assistments/assistments_2012.zip ]; then
    echo "Missing data/raw/assistments/assistments_2012.zip (see the Data section of README.md)." >&2
    exit 3
  fi
}

need_splits() {
  for split in development calibration confirmatory; do
    if [ ! -f "data/processed/assistments/${split}.parquet" ]; then
      echo "Missing data/processed/assistments/${split}.parquet. Run: ./reproduce.sh data" >&2
      exit 3
    fi
  done
}

run_stage() {
  local name="$1"
  if [ ! -f "scripts/${name}.py" ]; then
    echo "Unknown stage: ${name}" >&2
    exit 2
  fi
  echo "==> stage ${name}"
  "$PY" -u "scripts/${name}.py" 2>&1 | tee "outputs/$(basename "${name}").log"
}

case "$MODE" in
  verify)
    "$PY" scripts/verify_primary_results.py | tee outputs/verify.log
    "$PY" scripts/verify_chronological_results.py | tee -a outputs/verify.log
    ;;
  checksums)
    shasum -a 256 -c SHA256SUMS.txt | tee outputs/checksums.log
    ;;
  postprocess)
    echo "==> Recompute clustered false-stable bootstrap from frozen user aggregates"
    "$PY" scripts/wdlf_r4/false_stable_clustered_bootstrap.py
    echo "==> Recompute support-resolution decomposition from frozen validity JSON"
    "$PY" scripts/wdlf_r3/nc_decomposition.py
    echo "==> Re-verify against expected primary numbers"
    "$PY" scripts/verify_primary_results.py | tee outputs/verify.log
    ;;
  eedi-unseal)
    echo "==> Unseal frozen Eedi certificate payload (no retraining)"
    "$PY" scripts/wdlf_r3/run_r3_local_conditional.py unseal
    "$PY" scripts/verify_primary_results.py | tee outputs/verify.log
    ;;
  assist-integrity)
    echo "==> ASSISTments sealed integrity check"
    "$PY" scripts/wdlf_r7/run_release_frontier_confirmatory.py integrity
    "$PY" scripts/verify_primary_results.py | tee outputs/verify.log
    ;;
  data)
    need_archive
    run_stage assistments/source_reconstruction
    ;;
  stages)
    need_archive
    need_splits
    for name in "${STAGES[@]}"; do
      run_stage "$name"
    done
    "$PY" scripts/verify_chronological_results.py | tee outputs/verify_chronological.log
    ;;
  stage)
    need_archive
    need_splits
    run_stage "${2:?usage: ./reproduce.sh stage <name>}"
    ;;
  retrain-primary)
    need_archive
    need_splits
    run_stage assistments/chronological_replication
    ;;
  *)
    echo "Usage: ./reproduce.sh [verify|checksums|postprocess|eedi-unseal|assist-integrity|data|stages|stage <name>|retrain-primary]"
    exit 2
    ;;
esac

echo "Done. See outputs/ for logs."
