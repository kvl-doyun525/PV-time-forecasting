#!/usr/bin/env bash
# train_tslib_model.py 가 학습·평가 완료 후 기록하는 파일과 동일해야 함.
# SKIP_IF_DONE=0 이면 항상 학습 실행(스킵 비활성).

training_is_done() {
  local out_dir="$1"
  local pred_len="$2"
  [[ "${SKIP_IF_DONE:-1}" != "0" ]] && [[ -f "${out_dir}/metrics_test_${pred_len}h.json" ]]
}
