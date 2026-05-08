# PV Feature Mart 전처리 (`dataset/preprocessor`)

`dataset_snapshot` → 정제·기상 조인·태양 위치·파생 feature → **train/valid/test parquet** 및 **Track B wide fan(`fcst_*`)** 까지의 파이프라인이다.

상위 절차·Docker 경로는 저장소 루트의 [`pv_model_benchmark_execution.md`](../../pv_model_benchmark_execution.md) 를, Track B 통합 텐서·채널 매핑은 [`project/docs/track_b_mart_layout_and_training_implementation.md`](../../project/docs/track_b_mart_layout_and_training_implementation.md) 를 본다.

## 디렉터리 구성

| 파일 | 역할 |
|------|------|
| `config.py` | 경로(`SNAPSHOT_DIR`, `FEATURE_MART*`), split 날짜, `FEATURE_COLS`, Track B fan 슬롄 목록 |
| `run.py` | CLI 진입점 (`build`, `quality-check`, `enrich-track-b`, `update-manifest`) |
| `pv_cleaner.py` | PV 시계열 정제 |
| `weather_joiner.py` | ASOS·ERA5·격자 매핑 lookup |
| `solar_features.py` | pvlib 기반 일사·태양 위치 |
| `derived_features.py` | 롤링·lag·캘린더 파생 |
| `feature_mart_builder.py` | `build_feature_mart`, `per_site_split()` |
| `track_b_forecast_join.py` | `fcst_*` wide fan 조인(누수 규칙·스키마) |
| `track_b_enrich_mart.py` | per-site mart 전체에 enrich + `track_b_build_report.json` |

## 실행 위치

모든 예시는 **`dataset/preprocessor/`** 에서:

```bash
cd dataset/preprocessor
```

(저장소 루트 `time_forecasting/` 기준 상대 경로.)

## `run.py` 서브커맨드

### `build` — Feature mart 빌드

```bash
# per-site split (기본 출력: ../../project/artifacts/feature_mart_per_site)
python run.py build --split-mode per_site

# 출력 경로 명시
python run.py build --split-mode per_site --output-dir ../../project/artifacts/feature_mart_per_site

# global split (기본 출력: ../../project/artifacts/feature_mart)
python run.py build --split-mode global

# 일부 site만
python run.py build --split-mode per_site --sites 1 2 3
```

- **입력**: `config.SNAPSHOT_DIR` (기본 `../../project/artifacts/dataset_snapshot`) 내 `plant_meta.parquet`, PV·기상 parquet 등.
- **출력**: `train/`, `valid/`, `test/` 하위에 `{cid_seq}.parquet`, `scaler_stats.json`, `build_report.json` 등.
- `per_site` 시 추가로 `../../project/artifacts/per_site_split_manifest.json`.

### `quality-check` — 품질 리포트

```bash
python run.py quality-check --feature-mart-dir ../../project/artifacts/feature_mart_per_site
```

- 결과: 대상 mart 루트에 `quality_report.json`.
- 기준 요약: `run.py` 의 `cmd_quality_check` 독스트링 및 코드(전체 결측 완화·daytime 결측·최소 행 수 등)가 **정본**이다. `pv_model_benchmark_execution.md` §4.1 수치와 다르면 **코드·본 README**를 우선한다.

### `enrich-track-b` — Track B wide fan 부착

```bash
# ERA5 hourly valid → fcst_* (기본 컬럼: era5_native)
python run.py enrich-track-b \
  --input-mart-dir ../../project/artifacts/feature_mart_per_site \
  --join-mode era5_hourly_valid \
  --fcst-schema era5_native \
  --horizon-max 72

# 단기 proxy 슬롄 (tmp, pcp, sky …)
python run.py enrich-track-b \
  --input-mart-dir ../../project/artifacts/feature_mart_per_site \
  --join-mode era5_hourly_valid \
  --fcst-schema shortterm_aligned \
  --horizon-max 72

# 출력 디렉터리 명시 (미지정 시 per_site 입력이면 ../../project/artifacts/feature_mart_track_b_per_site)
python run.py enrich-track-b \
  --input-mart-dir ../../project/artifacts/feature_mart_per_site \
  --output-dir ../../project/artifacts/feature_mart_track_b_per_site \
  --join-mode era5_hourly_valid \
  --fcst-schema era5_native
```

- **예보 원천**: `--forecast-parquet` 비우면 `join_mode=era5_hourly_valid` → `dataset_snapshot/era5_nwp_input_raw.parquet`, `issue_target` → `era5_fcst_long.parquet`.
- **주요 옵션**: `--join-mode`, `--fcst-schema`, `--horizon-max`, `--fcst-cols`, `--sites`.
- **산출**: `track_b_build_report.json`, 입력에서 복사한 `scaler_stats.json` 등 + `fcst_*` 열이 붙은 parquet.

### `update-manifest`

```bash
python run.py update-manifest
```

- `config.py` 의 날짜 상수로 `../../project/artifacts/split_manifest.yaml` 을 갱신한다.

## 관련 스크립트

- `../../project/scripts/build_track_b_mart.sh` — 위 `enrich-track-b` + `update-manifest` 래퍼.
- `../../project/scripts/scan_fcst_parquet_health.py` — fan 스키마 점검.

## 의존성

- Python 3, `pandas`, `pyarrow`, `pyyaml`, pvlib·기타 `feature_mart_builder` import 체인 (프로젝트 Docker 이미지 권장).

## 복구 메모

- `run.py` / `config.py` / Track B 모듈은 PhotoRec 조각 및 구현 역추적로 맞춰 두었다. 동작 변경 시 **본 README**와 [`pv_model_benchmark_execution.md`](../../pv_model_benchmark_execution.md) §3.7 을 함께 갱신할 것.
