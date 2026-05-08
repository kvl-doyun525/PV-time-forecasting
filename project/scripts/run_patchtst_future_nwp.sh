#!/usr/bin/env bash
# PatchTST + 미래 NWP fan 통합 입력 (27 run, Track B mart)
# 산출: patchtst_future_nwp_seq_168
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

COMPOSE=(docker compose -f docker/docker-compose.yml)
RUNS_GROUP="${RUNS_GROUP:-patchtst_future_nwp_seq_168}"
SEQ_LEN="${SEQ_LEN:-168}"
MART="${FEATURE_MART:-artifacts/feature_mart_track_b_per_site}"
NWP_VARS="${FUTURE_NWP_VARS:-tmp,reh,wsd,vec,sky,pcp}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_BATCH_EVERY="${LOG_BATCH_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_BASE="artifacts/training_runs/${RUNS_GROUP}"
SEEDS=(42)
HORIZONS=(24 48 72)

run_combo() {
  local pl="$1" st="$2" h="$3" seed="$4"
  local out_dir="${OUT_BASE}/pl${pl}_s${st}_h${h}_seed${seed}"
  echo "=== PatchTST+future_nwp pl=${pl} st=${st} h=${h} seed=${seed} ==="
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
      --merge-future-nwp-into-encoder-input \
      --future-nwp-variable-names "${NWP_VARS}" \
      --num-workers "${NUM_WORKERS}" \
      --log-batch-every "${LOG_BATCH_EVERY}" \
      --output-dir "${out_dir}"
}

for pair in "24,24" "48,24" "48,48"; do
  IFS=, read -r pl st <<< "${pair}"
  for h in "${HORIZONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_combo "${pl}" "${st}" "${h}" "${seed}"
    done
  done
done

python3 src/report/aggregate_seeds.py \
  --model patchtst \
  --runs-dir artifacts/training_runs \
  --runs-group "${RUNS_GROUP}" \
  --horizons "${HORIZONS[@]}"

python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

echo "완료."
