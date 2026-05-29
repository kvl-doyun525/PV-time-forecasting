#!/usr/bin/env bash
# SegRNN: SEQ_LEN(기본 168)에 맞춰 seg_len·pred_len 조합을 맞춰야 함.
# (seg48 × pred48 은 168%48≠0 으로 실패하므로 제외) → seg24 만 사용, pred 24/48/72 × 3 seed = 9 runs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPTS_DIR}/lib_training_skip.sh"

COMPOSE=(docker compose -f docker/docker-compose.yml)
SEQ_LEN="${SEQ_LEN:-168}"
RUNS_GROUP="${RUNS_GROUP:-segrnn_seq_${SEQ_LEN}}"
SEG_LEN="${SEG_LEN:-24}"
MART="${FEATURE_MART:-artifacts/feature_mart_per_site}"
NUM_WORKERS="${NUM_WORKERS:-12}"
LOG_BATCH_EVERY="${LOG_BATCH_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_WINDOW_STRIDE="${TRAIN_WINDOW_STRIDE:-24}"
NO_MIDNIGHT_WINDOW_ALIGN="${NO_MIDNIGHT_WINDOW_ALIGN:-0}"
OUT_BASE="artifacts/training_runs/${RUNS_GROUP}"
SEEDS=(42)
HORIZONS=(24 48 72)

EXTRA_TRAIN_ARGS=()
if [[ "${NO_MIDNIGHT_WINDOW_ALIGN}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--no-midnight-window-align)
fi

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
        --train-window-stride "${TRAIN_WINDOW_STRIDE}" \
        "${EXTRA_TRAIN_ARGS[@]}" \
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
