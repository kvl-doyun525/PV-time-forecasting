#!/usr/bin/env bash
# DLinear: pred_len × seed 전부 학습 후 seed 집계 + 리더보드 갱신
# 사용: project/ 에서  bash scripts/run_dlinear.sh
# 환경: SEQ_LEN 기본 168 → RUNS_GROUP 기본 dlinear_seq_${SEQ_LEN} (RUNS_GROUP 직접 지정 시 그대로)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPTS_DIR}/lib_training_skip.sh"

COMPOSE=(docker compose -f docker/docker-compose.yml)
SEQ_LEN="${SEQ_LEN:-168}"
RUNS_GROUP="${RUNS_GROUP:-dlinear_seq_${SEQ_LEN}}"
MART="${FEATURE_MART:-artifacts/feature_mart_per_site}"
NUM_WORKERS="${NUM_WORKERS:-4}"
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

train_one() {
  local h="$1" seed="$2" out_rel="$3"
  local out_dir="${OUT_BASE}/${out_rel}"
  if training_is_done "${out_dir}" "${h}"; then
    echo "=== skip (이미 완료): ${out_dir} ==="
    return 0
  fi
  echo "=== DLinear pred_len=${h} seed=${seed} → ${out_dir} ==="
  "${COMPOSE[@]}" run --rm unified \
    python src/train/train_tslib_model.py \
      --model DLinear \
      --feature-mart "${MART}" \
      --seq-len "${SEQ_LEN}" \
      --batch-size "${BATCH_SIZE}" \
      --pred-len "${h}" \
      --seed "${seed}" \
      --num-workers "${NUM_WORKERS}" \
      --log-batch-every "${LOG_BATCH_EVERY}" \
      --train-window-stride "${TRAIN_WINDOW_STRIDE}" \
      "${EXTRA_TRAIN_ARGS[@]}" \
      --output-dir "${out_dir}"
}

for S in "${SEEDS[@]}"; do
  train_one 24 "${S}" "seed_${S}"
done
for H in 48 72; do
  for S in "${SEEDS[@]}"; do
    train_one "${H}" "${S}" "h${H}_seed_${S}"
  done
done

echo "=== DLinear 전체 완료, 집계 중 ==="
python3 src/report/aggregate_seeds.py \
  --model dlinear \
  --runs-dir artifacts/training_runs \
  --runs-group "${RUNS_GROUP}" \
  --horizons 24 48 72

echo "=== 리더보드 갱신 ==="
python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

echo "완료."
