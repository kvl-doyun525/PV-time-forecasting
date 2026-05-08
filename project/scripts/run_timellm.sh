#!/usr/bin/env bash
# Time-LLM (train_tslib_model + GPT2 backbone) — time-llm 이미지 사용
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

COMPOSE=(docker compose -f docker/docker-compose.yml)
RUNS_GROUP="${RUNS_GROUP:-timellm_seq_168}"
SEQ_LEN="${SEQ_LEN:-168}"
MART="${FEATURE_MART:-artifacts/feature_mart_per_site}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_BATCH_EVERY="${LOG_BATCH_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_BASE="artifacts/training_runs/${RUNS_GROUP}"
SEEDS=(42)
HORIZONS=(24 48 72)

for H in "${HORIZONS[@]}"; do
  for S in "${SEEDS[@]}"; do
    out_dir="${OUT_BASE}/timellm_gpt2_h${H}_seed${S}"
    echo "=== TimeLLM pred_len=${H} seed=${S} ==="
    "${COMPOSE[@]}" run --rm time-llm \
      python src/train/train_tslib_model.py \
        --model TimeLLM \
        --feature-mart "${MART}" \
        --seq-len "${SEQ_LEN}" \
        --batch-size "${BATCH_SIZE}" \
        --pred-len "${H}" \
        --seed "${S}" \
        --llm-model GPT2 \
        --llm-layers 6 \
        --num-workers "${NUM_WORKERS}" \
        --log-batch-every "${LOG_BATCH_EVERY}" \
        --output-dir "${out_dir}"
  done
done

echo "=== TimeLLM 전체 완료, 집계 중 ==="
python3 src/report/aggregate_seeds.py \
  --model timellm \
  --runs-dir artifacts/training_runs \
  --runs-group "${RUNS_GROUP}" \
  --horizons 24 48 72

echo "=== 리더보드 갱신 ==="
python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

echo "완료."
