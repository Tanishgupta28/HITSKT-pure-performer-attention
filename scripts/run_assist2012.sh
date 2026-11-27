#!/usr/bin/env bash
# Run from the repository root on an available CUDA device.
set -euo pipefail
export PYTHONHASHSEED=42 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
data=${ASSIST2012_DATA:-data/processed/assist2012/full}
evidence=reports/datasets/assist2012
workers=${ASSIST2012_WORKERS:-2}
mkdir -p "$evidence"
stage() {
    printf '[%s] ASSIST2012 %s\n' "$(date -u +%FT%TZ)" "$1"
}
stage "verify existing preprocessing and temporal cache"
cp "$data/reports/summary.json" "$evidence/summary.json"
cp "$data/rkt/provenance.json" "$evidence/rkt_provenance.json"
if [[ ! -s "$evidence/historical_architecture.json" ]]; then
    python -m scripts.verify_assist2012 --architecture > "$evidence/historical_architecture.json"
fi
python -m scripts.write_assist2012_methods
if [[ ! -s "$evidence/data_verification.json" ]]; then
    python -m scripts.verify_assist2012 "$data" > "$evidence/data_verification.json"
fi
python - "$data" "$evidence/data_verification.json" <<'PY_DATA'
import hashlib
import json
import sys
from pathlib import Path

data, report = map(Path, sys.argv[1:])
audit = json.loads(report.read_text())
metadata = (data / "session_store" / "metadata.json").read_bytes()
if not audit["verified"] or audit["session_store_metadata_sha256"] != hashlib.sha256(metadata).hexdigest():
    raise RuntimeError("ASSIST2012 data verification does not match the session store")
PY_DATA
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
free, total = torch.cuda.mem_get_info()
print(f"Pytorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name()}, free/total {free}/{total} bytes")
PY
stage "CUDA smoke tests"
python -m scripts.smoke_algebra2005 "$data/session_store" "$data/rkt" --dataset assist2012 > "$evidence/cuda_smoke.jsonl"
for model in dkt dkvmn sakt hitskt rkt; do
    run="experiments/$model/assist2012/full"
    resume=()
    if [[ -s "$run/last_model.pt" ]]; then
        resume=(--resume)
    fi
    case "$model" in
        rkt) trainer=(scripts.train_rkt assist2012 "$data/session_store" "$data/rkt" "$run")
             verify=(--prepared-root "$data/rkt");;
        hitskt) trainer=(scripts.train_hitskt assist2012 "$data/session_store" "$run")
                verify=();;
        *) trainer=(scripts.train_baseline "$model" assist2012 "$data/session_store" "$run" --transport packed64)
           verify=();;
    esac
    if [[ ! -f "$run/_SUCCESS" ]]; then
        stage "$model full GPU training"
        python -u -m "${trainer[@]}" --workers "$workers" "${resume[@]}"
    fi
    if [[ ! -f "$run/independent_evaluation.json" ]]; then
        stage "$model independent best-checkpoint replay"
        python -m scripts.evaluate_checkpoint "$run" "$data/session_store" --workers "$workers" \
            "${verify[@]}" --output "$run/independent_evaluation.json"
    fi
done
stage "repository tests"
python -m pytest -q tests
stage "verify and aggregate full-run results"
python -m scripts.aggregate_assist2012
stage "render three comparison graphs"
python report_graphs/generate_comparison_graphs.py --assist2012
python report_graphs/generate_parameter_comparison.py --assist2012
