#!/usr/bin/env bash
# Time-LLM + 미래 NWP fan 통합 입력 (time-llm 이미지, Track B mart)
# 산출: timellm_future_nwp_seq_168
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

COMPOSE=(docker compose -f docker/docker-compose.yml)
RUNS_GROUP="${RUNS_GROUP:-timellm_future_nwp_seq_168}"
SEQ_LEN="${SEQ_LEN:-168}"
MART="${FEATURE_MART:-artifacts/feature_mart_track_b_per_site}"
NWP_VARS="${FUTURE_NWP_VARS:-tmp,reh,wsd,vec,sky,pcp}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_BATCH_EVERY="${LOG_BATCH_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUT_BASE="artifacts/training_runs/${RUNS_GROUP}"
SEEDS=(42)
HORIZONS=(24 48 72)

for H in "${HORIZONS[@]}"; do
  for S in "${SEEDS[@]}"; do
    out_dir="${OUT_BASE}/timellm_gpt2_h${H}_seed${S}"
    echo "=== TimeLLM+future_nwp pred=${H} seed=${S} ==="
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
        --merge-future-nwp-into-encoder-input \
        --future-nwp-variable-names "${NWP_VARS}" \
        --num-workers "${NUM_WORKERS}" \
        --log-batch-every "${LOG_BATCH_EVERY}" \
        --output-dir "${out_dir}"
  done
done

python3 src/report/aggregate_seeds.py \
  --model timellm \
  --runs-dir artifacts/training_runs \
  --runs-group "${RUNS_GROUP}" \
  --horizons 24 48 72

python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

echo "완료."
