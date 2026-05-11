#!/usr/bin/env bash
# DLinear + 미래 NWP fan 통합 입력 (`--merge-future-nwp-into-encoder-input`, Track B mart)
# 산출 그룹명: dlinear_future_nwp_seq_168 (기존 artifacts와 동일)
# 사용: project/ 에서 bash scripts/run_dlinear_future_nwp.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPTS_DIR}/lib_training_skip.sh"

COMPOSE=(docker compose -f docker/docker-compose.yml)
RUNS_GROUP="${RUNS_GROUP:-dlinear_future_nwp_seq_168}"
SEQ_LEN="${SEQ_LEN:-168}"
MART="${FEATURE_MART:-artifacts/feature_mart_track_b_per_site}"
NWP_VARS="${FUTURE_NWP_VARS:-tmp,reh,wsd,vec,sky,pcp}"
NUM_WORKERS="${NUM_WORKERS:-12}"
LOG_BATCH_EVERY="${LOG_BATCH_EVERY:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
OUT_BASE="artifacts/training_runs/${RUNS_GROUP}"
SEEDS=(42)
HORIZONS=(24 48 72)

train_one() {
  local h="$1" seed="$2" out_rel="$3"
  local out_dir="${OUT_BASE}/${out_rel}"
  if training_is_done "${out_dir}" "${h}"; then
    echo "=== skip (이미 완료): ${out_dir} ==="
    return 0
  fi
  echo "=== DLinear+future_nwp pred_len=${h} seed=${seed} → ${out_dir} ==="
  "${COMPOSE[@]}" run --rm unified \
    python src/train/train_tslib_model.py \
      --model DLinear \
      --feature-mart "${MART}" \
      --seq-len "${SEQ_LEN}" \
      --batch-size "${BATCH_SIZE}" \
      --pred-len "${h}" \
      --seed "${seed}" \
      --merge-future-nwp-into-encoder-input \
      --future-nwp-variable-names "${NWP_VARS}" \
      --num-workers "${NUM_WORKERS}" \
      --log-batch-every "${LOG_BATCH_EVERY}" \
      --output-dir "${out_dir}"
}

for H in "${HORIZONS[@]}"; do
  for S in "${SEEDS[@]}"; do
    train_one "${H}" "${S}" "h${H}_seed_${S}"
  done
done

echo "=== DLinear+future_nwp 집계 ==="
python3 src/report/aggregate_seeds.py \
  --model dlinear \
  --runs-dir artifacts/training_runs \
  --runs-group "${RUNS_GROUP}" \
  --horizons "${HORIZONS[@]}"

python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

echo "완료."
