#!/usr/bin/env bash
# Run from the repository root on an available CUDA device.
set -euo pipefail
export PYTHONHASHSEED=42 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
data=${ALGEBRA2005_DATA:-data/processed/algebra2005/full}
evidence=reports/datasets/algebra2005
workers=${ALGEBRA2005_WORKERS:-2}
mkdir -p "$evidence"
stage() {
    printf '[%s] Algebra2005 %s\n' "$(date -u +%FT%TZ)" "$1"
}
stage "verify existing preprocessing and temporal cache"
cp "$data/reports/summary.json" "$evidence/summary.json"
cp "$data/rkt/provenance.json" "$evidence/rkt_provenance.json"
python -m scripts.verify_algebra2005_data "$data" > "$evidence/data_verification.json"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
free, total = torch.cuda.mem_get_info()
print(f"Pytorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name()}, free/total {free}/{total} bytes")
PY
stage "CUDA smoke tests"
python -m scripts.smoke_algebra2005 "$data/session_store" "$data/rkt" > "$evidence/cuda_smoke.jsonl"
for model in dkt dkvmn sakt rkt hitskt; do
    run="experiments/$model/algebra2005/full"
    resume=()
    if [[ -s "$run/last_model.pt" ]]; then
        resume=(--resume)
    fi
    case "$model" in
        rkt) trainer=(scripts.train_rkt algebra2005 "$data/session_store" "$data/rkt" "$run")
             verify=(--prepared-root "$data/rkt");;
        hitskt) trainer=(scripts.train_hitskt algebra2005 "$data/session_store" "$run")
                verify=();;
        *) trainer=(scripts.train_baseline "$model" algebra2005 "$data/session_store" "$run" --transport packed64)
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
python -m scripts.aggregate_algebra2005 "$@"
stage "render three comparison graphs"
python report_graphs/generate_comparison_graphs.py --algebra2005
python report_graphs/generate_parameter_comparison.py --algebra2005
