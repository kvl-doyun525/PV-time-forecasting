#!/usr/bin/env bash
# SegRNN: seq_len=168 일 때 seg_len 은 168과 pred_len 의 약수여야 함.
# (seg48 × pred48 은 168%48≠0 으로 실패하므로 제외) → seg24 만 사용, pred 24/48/72 × 3 seed = 9 runs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPTS_DIR}/lib_training_skip.sh"

COMPOSE=(docker compose -f docker/docker-compose.yml)
RUNS_GROUP="${RUNS_GROUP:-segrnn_seq_168}"
SEQ_LEN="${SEQ_LEN:-168}"
SEG_LEN="${SEG_LEN:-24}"
MART="${FEATURE_MART:-artifacts/feature_mart_per_site}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_BATCH_EVERY="${LOG_BATCH_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUT_BASE="artifacts/training_runs/${RUNS_GROUP}"
SEEDS=(42)
HORIZONS=(24 48 72)

for H in "${HORIZONS[@]}"; do
  for S in "${SEEDS[@]}"; do
    out_dir="${OUT_BASE}/seg${SEG_LEN}_h${H}_seed${S}"
    if training_is_done "${out_dir}" "${H}"; then
      echo "=== skip (이미 완료): ${out_dir} ==="
      continue
    fi
    echo "=== SegRNN seg_len=${SEG_LEN} pred_len=${H} seed=${S} ==="
    "${COMPOSE[@]}" run --rm unified \
      python src/train/train_tslib_model.py \
        --model SegRNN \
        --feature-mart "${MART}" \
        --seq-len "${SEQ_LEN}" \
        --batch-size "${BATCH_SIZE}" \
        --pred-len "${H}" \
        --seg-len "${SEG_LEN}" \
        --seed "${S}" \
        --num-workers "${NUM_WORKERS}" \
        --log-batch-every "${LOG_BATCH_EVERY}" \
        --output-dir "${out_dir}"
  done
done

echo "=== SegRNN 전체 완료, 집계 중 ==="
python3 src/report/aggregate_seeds.py \
  --model segrnn \
  --runs-dir artifacts/training_runs \
  --runs-group "${RUNS_GROUP}" \
  --horizons 24 48 72

echo "=== 리더보드 갱신 ==="
python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

echo "완료."
