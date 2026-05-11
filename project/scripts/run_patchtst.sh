#!/usr/bin/env bash
# PatchTST: §8 실험 매트릭스 (patch_len, stride) × pred_len × seed = 27 runs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPTS_DIR}/lib_training_skip.sh"

COMPOSE=(docker compose -f docker/docker-compose.yml)
RUNS_GROUP="${RUNS_GROUP:-patchtst_seq_168}"
SEQ_LEN="${SEQ_LEN:-168}"
MART="${FEATURE_MART:-artifacts/feature_mart_per_site}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_BATCH_EVERY="${LOG_BATCH_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_WINDOW_STRIDE="${TRAIN_WINDOW_STRIDE:-24}"
NO_MIDNIGHT_WINDOW_ALIGN="${NO_MIDNIGHT_WINDOW_ALIGN:-0}"
OUT_BASE="artifacts/training_runs/${RUNS_GROUP}"
SEEDS=(42)

EXTRA_TRAIN_ARGS=()
if [[ "${NO_MIDNIGHT_WINDOW_ALIGN}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--no-midnight-window-align)
fi

run_combo() {
  local pl="$1" st="$2" h="$3" seed="$4"
  local out_dir="${OUT_BASE}/pl${pl}_s${st}_h${h}_seed${seed}"
  if training_is_done "${out_dir}" "${h}"; then
    echo "=== skip (이미 완료): ${out_dir} ==="
    return 0
  fi
  echo "=== PatchTST pl=${pl} st=${st} pred_len=${h} seed=${seed} ==="
  "${COMPOSE[@]}" run --rm unified \
    python src/train/train_tslib_model.py \
      --model PatchTST \
      --feature-mart "${MART}" \
      --seq-len "${SEQ_LEN}" \
      --batch-size "${BATCH_SIZE}" \
      --pred-len "${h}" \
      --patch-len "${pl}" \
      --stride "${st}" \
      --seed "${seed}" \
      --num-workers "${NUM_WORKERS}" \
      --log-batch-every "${LOG_BATCH_EVERY}" \
      --train-window-stride "${TRAIN_WINDOW_STRIDE}" \
      "${EXTRA_TRAIN_ARGS[@]}" \
      --output-dir "${out_dir}"
}

for pair in "24,24" "48,24" "48,48"; do
  IFS=, read -r pl st <<< "${pair}"
  for h in 24 48 72; do
    for seed in "${SEEDS[@]}"; do
      run_combo "${pl}" "${st}" "${h}" "${seed}"
    done
  done
done

echo "=== PatchTST 전체 완료, 집계 중 ==="
python3 src/report/aggregate_seeds.py \
  --model patchtst \
  --runs-dir artifacts/training_runs \
  --runs-group "${RUNS_GROUP}" \
  --horizons 24 48 72

echo "=== 리더보드 갱신 ==="
python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

echo "완료."
