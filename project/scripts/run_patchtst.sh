#!/usr/bin/env bash
# PatchTST: §8 실험 매트릭스 (patch_len, stride) × pred_len × seed = 27 runs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

COMPOSE=(docker compose -f docker/docker-compose.yml)
RUNS_GROUP="${RUNS_GROUP:-patchtst_seq_168}"
SEQ_LEN="${SEQ_LEN:-168}"
MART="${FEATURE_MART:-artifacts/feature_mart_per_site}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_BATCH_EVERY="${LOG_BATCH_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUT_BASE="artifacts/training_runs/${RUNS_GROUP}"
SEEDS=(42)

run_combo() {
  local pl="$1" st="$2" h="$3" seed="$4"
  local out_dir="${OUT_BASE}/pl${pl}_s${st}_h${h}_seed${seed}"
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
