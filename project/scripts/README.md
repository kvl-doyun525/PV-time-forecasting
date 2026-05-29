# `project/scripts/` — 학습·집계 스크립트

작업 디렉터리는 **`project/` 루트**이다 (`docker-compose.yml` 이 `docker/` 에 있음).

## 공통

- **GPU**: `docker compose ... run --rm unified` / `time-llm` 은 `docker-compose.yml` 의 GPU 예약을 따른다.
- **데이터**: 기본 mart는 `artifacts/feature_mart_per_site` (`FEATURE_MART` 로 변경).  
  `run_*_future_nwp.sh` 는 기본 **`artifacts/feature_mart_track_b_per_site`** (`fcst_*` fan 필요).
- **실험 그룹 폴더**: `RUNS_GROUP` 으로 명시 가능. 미설정 시 기본값은 **`{모델}_seq_${SEQ_LEN}`** 형태(예: `SEQ_LEN=168` → `dlinear_seq_168`, `SEQ_LEN=720` → `dlinear_seq_720`)라 `SEQ_LEN`만 바꿔도 이전 결과 폴더를 덮어쓰지 않는다.
- **배치**: `BATCH_SIZE` (기본: DLinear/SegRNN/PatchTST·해당 future_nwp 는 `256`, TimeLLM·`run_timellm*_future_nwp.sh` 는 `32`). `BATCH_SIZE=128 bash scripts/run_dlinear.sh` 처럼 환경변수로 덮어쓴다.
- **train 윈도 슬라이스(행)**: `TRAIN_WINDOW_STRIDE` (기본 `24` → `train_tslib_model.py --train-window-stride`). 첫 자정 앵커 끄기: `NO_MIDNIGHT_WINDOW_ALIGN=1` → `--no-midnight-window-align` 전달.
- **이미 학습된 런 건너뛰기**: 기본적으로 `--output-dir` 아래에 `metrics_test_<pred_len>h.json` 이 있으면 해당 조합은 스킵한다. 강제 재학습은 `SKIP_IF_DONE=0 bash scripts/run_dlinear_future_nwp.sh` .
- **윈도우 stride·자정 앵커 (`pv_dataset`)**: 기본은 **가능한 가장 이른 00:00에 시작하는 첫 윈도만** 자정에 맞추고, 그 다음은 **`--train-window-stride`(행)** 간격으로만 이동한다(stride=24·48이면 1시간 마트에서 이후 시작도 자정에 맞춰짐). stride=1·12이면 첫만 자정·이후는 매시간 또는 12시간 간격. 전부 행 0 기준이면 `--no-midnight-window-align` . `valid`·테스트 예측은 `pred_len` 행 간격 + 동일 자정 앵커.
- **배치 진행 로그**: `LOG_BATCH_EVERY` (기본 `0` = 비활성). 예: `LOG_BATCH_EVERY=50` 이면 50 배치마다 train/valid step 출력 (`--log-batch-every`).
- **로그**: `mkdir -p logs` 후 `2>&1 | tee logs/run_xxx.log` 권장.

### 이전 컨테이너 종료 대기 후 다음 스크립트 실행

```bash
echo "DLinear 완료 대기 중..."
while docker ps --format '{{.Names}}' | grep -q 'pv-bench'; do
  sleep 10
done
sleep 5

echo "=== DLinear 완료, SegRNN 시작 ==="
bash scripts/run_segrnn.sh 2>&1 | tee logs/run_segrnn.log

echo "=== SegRNN 완료, PatchTST 시작 ==="
bash scripts/run_patchtst.sh 2>&1 | tee logs/run_patchtst.log
```

### 한 줄로 연쇄 (이전 단계 실패 시 중단)

```bash
bash scripts/run_dlinear.sh 2>&1 | tee logs/run_dlinear.log && \
bash scripts/run_segrnn.sh 2>&1 | tee logs/run_segrnn.log && \
bash scripts/run_patchtst.sh 2>&1 | tee logs/run_patchtst.log
```

### Track B future_nwp 일괄 학습 (`run_train.sh`)

스크립트 상단에서 `SEQ_LEN`, `NUM_WORKERS`, `LOG_BATCH_EVERY`, (선택) `BATCH_SIZE` `export` 한 뒤 `bash scripts/run_*_future_nwp.sh` 를 **순차 호출**한다. 순서·포함 모델을 바꾸려면 `run_train.sh` 안의 `bash scripts/...` 줄을 직접 주석 처리하거나 순서를 바꾼다.

```bash
BATCH_SIZE=128 NUM_WORKERS=8 bash scripts/run_train.sh
```

## 스크립트 요약

| 스크립트 | 설명 |
|-----------|------|
| `setup_vendor.sh` | `vendor/TSLib` 클론 |
| `build_all.sh` | Docker 이미지 빌드 |
| `verify_gpu.sh` | `nvidia-smi` 스모크 |
| `run_dlinear.sh` | DLinear 24/48/72h × 3 seed → `aggregate_seeds` + `build_leaderboard` |
| `run_segrnn.sh` | SegRNN `seg24` × 24/48/72 × 3 seed (`SEQ_LEN` 기본 168, `RUNS_GROUP` 기본 `segrnn_seq_${SEQ_LEN}`) |
| `run_patchtst.sh` | PatchTST §8 매트릭스 27 run |
| `run_dlinear_future_nwp.sh` | Track B mart + `--merge-future-nwp-…` → 기본 `dlinear_future_nwp_seq_${SEQ_LEN}` |
| `run_segrnn_future_nwp.sh` | 동일 → 기본 `segrnn_future_nwp_seq_${SEQ_LEN}` |
| `run_patchtst_future_nwp.sh` | 동일 → 기본 `patchtst_future_nwp_seq_${SEQ_LEN}` |
| `run_timellm_future_nwp.sh` | `time-llm` 이미지 + merge → 기본 `timellm_future_nwp_seq_${SEQ_LEN}` |
| `run_train.sh` | future_nwp 스크립트 연쇄 실행 + 공통 env (파일 안 `export`·`bash` 줄 편집) |
| `run_timellm.sh` | Time-LLM (`time-llm` 이미지), merge 없음 |
| `run_baseline.sh` | Seasonal + Persistence (`baseline_seasonal_naive.py`) |
| `run_llama_lora.sh` | `src/train/train_llama_lora.py` **있을 때만** 실행 |
| `run_gemma_lora.sh` | `src/train/train_gemma_lora.py` **있을 때만** 실행 |
| `finalize_dlinear.sh` | DLinear만 재집계 + 리더보드 |
| `build_track_b_mart.sh` | Track B enrich (레포 루트 `dataset/preprocessor` 호출) |
| `batch_plot_training_runs.py` 등 | 기타 유틸은 동일 폴더 참고 |

## Python 집계·리더보드 (`src/report/`)

- `python3 src/report/aggregate_seeds.py --model segrnn --runs-dir artifacts/training_runs --runs-group <학습 시 RUNS_GROUP 과 동일> --horizons 24 48 72` (예: `segrnn_seq_168`, `dlinear_seq_720`)
- `python3 src/report/build_leaderboard.py --runs-dir artifacts/training_runs --output artifacts/leaderboard.md` — horizon별 표, `h{H}_seed{N}` vs `h{H}_seed_{N}` 폴더 쌍 비교 표, Raw 절. `--max-raw-per-horizon` 기본 80.
- 원시 metrics만 표로: `python3 src/report/build_accuracy_leaderboard.py`

## 설계 메모 (SegRNN)

`seq_len=168` 일 때 **`seg_len` 은 168의 약수**여야 하고, **`pred_len % seg_len == 0`** 이어야 한다.  
그래서 `seg48`+`pred48` 조합은 **사용하지 않는다** (과거 로그의 reshape 오류 원인).

## 참고 문서

- [`../pv_model_benchmark_execution.md`](../pv_model_benchmark_execution.md) — Docker 볼륨·단계별 절차
