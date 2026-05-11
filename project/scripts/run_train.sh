#!/usr/bin/env bash
# Track B future_nwp: 공통 env 설정 후 학습 스크립트를 순서대로 실행.
# 사용: project/ 루트에서  bash scripts/run_train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

# --- 필요 시 여기만 수정 (하위 run_*_future_nwp.sh 에 export 전달) ---
export SEQ_LEN="${SEQ_LEN:-168}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
export LOG_BATCH_EVERY="${LOG_BATCH_EVERY:-0}"
# 선택: 모든 단계에 동일 배치 (TimeLLM은 기본 32라 VRAM 여유 없으면 32 이하 권장)
# export BATCH_SIZE=128
export FEATURE_MART="${FEATURE_MART:-artifacts/feature_mart_track_b_per_site}"
export FUTURE_NWP_VARS="${FUTURE_NWP_VARS:-tmp,reh,wsd,vec,sky,pcp}"
export TRAIN_WINDOW_STRIDE="${TRAIN_WINDOW_STRIDE:-1}"
export SKIP_IF_DONE=0
# export NO_MIDNIGHT_WINDOW_ALIGN=1   # 첫 자정 앵커 끄고 행 0·stride 그리드만 쓸 때

echo "[run_train] SEQ_LEN=${SEQ_LEN} NUM_WORKERS=${NUM_WORKERS} LOG_BATCH_EVERY=${LOG_BATCH_EVERY} BATCH_SIZE=${BATCH_SIZE:-<미설정·스크립트 기본>} TRAIN_WINDOW_STRIDE=${TRAIN_WINDOW_STRIDE} NO_MIDNIGHT_WINDOW_ALIGN=${NO_MIDNIGHT_WINDOW_ALIGN:-0}"
echo "[run_train] FEATURE_MART=${FEATURE_MART}"

# bash scripts/run_dlinear_future_nwp.sh 2>&1 | tee logs/run_dlinear_future_nwp.log
bash scripts/run_segrnn_future_nwp.sh 2>&1 | tee logs/run_segrnn_future_nwp.log
bash scripts/run_patchtst_future_nwp.sh 2>&1 | tee logs/run_patchtst_future_nwp.log
bash scripts/run_timellm_future_nwp.sh 2>&1 | tee logs/run_timellm_future_nwp.log

echo "run_train: 완료."
