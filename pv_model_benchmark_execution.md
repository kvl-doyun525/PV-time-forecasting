# PV 발전 예측 모델 벤치마크 — 구체적 수행 절차서

**파일**: `pv_model_benchmark_execution.md` (저장소 최상위)  
**상위 설계**: [`pv_model_benchmark_plan.md`](./pv_model_benchmark_plan.md)

> **복구·병합 (2026-05-06)**: PhotoRec `recup_dir.2/f279814712.txt`(2685줄) 본문 **§2.2.5~문서 끝**을 기준 축으로 삼고, 원본에 없던 **§단계 1·2·§2.1~2.2.4**는 `pv_model_benchmark_plan.md`, `dataset/pv_collector/README.md`, `dataset/weather_collector/README.md`, 구현 코드와 정합되도록 재작성했다. 에이전트 기록의 **「입력 설계 확장 벤치마크」**·**§3.7 Track B**를 단계 3·체크리스트에 반영했다.
>
> **추가 정합 (2026-05-06)**: 소스 복구에 맞춰 문서를 갱신했다. — **`dataset/preprocessor/config.py`**, **`run.py`**, **`track_b_forecast_join.py`**, **`track_b_enrich_mart.py`**, **`project/src/datasets/pv_dataset.py`**, **`project/src/train/train_tslib_model.py`**. Docker `docker run` 예시의 호스트 볼륨은 레포 구조에 맞게 **`project/artifacts`**, **`project/vendor/TSLib`**, **`project/src`**, **`project/conf`** 기준으로 통일했다 (`docker compose -f project/docker/docker-compose.yml` 사용 시에는 YAML의 `../artifacts` 규칙이 동일하다). 컨테이너 안에서 받는 경로는 **`/workspace/artifacts/...`**, TSLib은 **`/workspace/vendor/TSLib`** 이다. §5.10 GPU 스모크 코드는 TSLib `Model(..., None, None, None)` 호출·`task_name` 등 필수 configs를 반영했다.
>
> **「입력 설계 확장 벤치마크」 보강 (2026-05-06)**: PhotoRec 복구본에는 해당 절의 **긴 원문 덩어리가 거의 없었고**, 기억과의 차이는 `project/docs/track_b_mart_layout_and_training_implementation.md`에 있던 설명과 겹친 것으로 보인다. 실행 절차서 쪽에는 **§10 트랙과의 용어 구분**, **Track B 파이프라인·merge CLI·주의사항**을 위 문서·코드에 맞춰 요약해 넣었다.

---

## 목차 (단계 요약)

1. [단계 1: 환경·저장소 준비](#단계-1-환경저장소-준비)
2. [단계 2: 원시 데이터 수집](#단계-2-원시-데이터-수집)
3. [단계 3: 데이터 전처리 및 Feature Mart 구축](#단계-3-데이터-전처리-및-feature-mart-구축)
4. [단계 4: 데이터 검증 및 Sanity Check](#단계-4-데이터-검증-및-sanity-check)
5. [단계 5: Docker 환경 구성](#단계-5-docker-환경-구성)
6. [단계 6: Baseline 실험](#단계-6-baseline-실험)
7. [단계 7: SegRNN 실험](#단계-7-segrnn-실험)
8. [단계 8: PatchTST 실험](#단계-8-patchtst-실험)
9. [단계 9: Time-LLM 실험](#단계-9-time-llm-실험)
10. [단계 10: LLaMA 실험](#단계-10-llama-실험)
11. [단계 11: GEMMA 실험](#단계-11-gemma-실험)
12. [단계 12: CPU 추론 벤치마크](#단계-12-cpu-추론-벤치마크)
13. [단계 13: 리더보드 집계 및 의사결정](#단계-13-리더보드-집계-및-의사결정)
14. [입력 설계 확장 벤치마크](#입력-설계-확장-벤치마크)
15. [전체 체크리스트](#전체-체크리스트)

---

## 단계 1: 환경·저장소 준비

- **권장 OS**: Ubuntu + Docker (`pv_model_benchmark_plan.md` §3). Windows/WSL2는 개발용으로 허용하되, 공식 비교 결과는 Linux 컨테이너 기준으로 남긴다.
- **저장소 루트**: 본 문서는 `time_forecasting/` 최상위에 있으며, 학습·전처리 **작업 디렉터리**는 `project/`를 기본으로 한다.
- **초기 작업**:
  - `project/docker/` 이미지 정의 확인
  - `project/scripts/README.md` — `setup_vendor.sh`, `build_all.sh`, `verify_gpu.sh` 등 실행 순서
  - Git LFS·대용량 모델은 `project/docs/model_download_guide.md` 참고

---

## 단계 2: 원시 데이터 수집

### 2.1 사내 DB — PV 시간별 발전 (`tb_data_pv_hour`)

**구현**: `dataset/pv_collector/` (`README.md`, `run.py`)

| 항목 | 내용 |
|------|------|
| **핵심 테이블** | `tb_data_pv_hour` — 시간(1h) 단위 발전·전기량 |
| **단위 키** | `cid_seq` (CID = 인버터/발전 유닛). `plant_seq`는 발전소 단위로 메타와 조인 |
| **필터** | `phase_type = 'tp'` (삼상)만 수집 |
| **출력** | `project/artifacts/dataset_snapshot/plant_meta.parquet`, `pv_raw_hourly.parquet` (경로는 `pv_collector` 설정 따름) |
| **접속** | `.env`에 DB 호스트·계정 (템플릿: `dataset/pv_collector/.env.example`) |

발전소 **위경도·행정구역**은 `plant_meta` 수집 쿼리에서 `plant_seq` 기준 메타 테이블과 조인해 채운다 (상세 DDL은 저장소의 `krems_db_schema.sql` 참고).

### 2.2 기상·NWP 수집 (KMA + ERA5 + 예보)

**구현**: `dataset/weather_collector/`

#### 2.2.1 원칙

- `plant_meta.parquet`의 위경도를 기준으로 **지상관측(ASOS)·AWS·단기예보 격자**를 매핑한다.
- API 키: 공공데이터포털(`KMA_API_KEY`/`ASOS_API_KEY`), AWS 시간통계는 **기상청 API허브** `APIHUB_KEY` (선택), ERA5는 CDS (`CDS_API_KEY` 또는 `~/.cdsapirc`).

#### 2.2.2 단기·초단기 예보

- **단기예보** (`kma_forecast_collector.py`): 최대 약 3일, 1시간 간격, 일 8회 발표.
- **초단기**: 최대 6시간, 고빈도 보완용.
- **리드타임**: `target_time - issue_time`. 학습·조인 시 **반드시** `issue_time ≤ base_time`(또는 t0) 규칙을 지킨다 — [§3.3](#33-forecast-leakage-방지-join).

#### 2.2.3 ERA5 재분석 (전략 B, 학습용 NWP)

- **훈련**: ERA5 재분석(바이어스 교정)으로 과거 일관된 기상장을 제공.
- **실시간 추론**: 기상청 단기예보(일사량 변수 없음 → pvlib clearsky + ASOS `icsr` 등으로 보완).
- **수집·교정**: `dataset/weather_collector/era5_collector.py`, 출력 예: `era5_nwp_input_raw.parquet`, `era5_nwp_bias_corrected.parquet`.

#### 2.2.4 ASOS 시간 관측 이력

- **서비스**: `AsosHourlyInfoService/getWthrDataList` (공공데이터포털).
- **구현 파일**: `dataset/weather_collector/asos_collector.py` — 재시도·지점별 incremental 저장·`site_to_kma_grid.csv`의 `asos_stn_id` 사용.
- **CLI 예**: `python run.py mapping` 후 `python run.py asos --start YYYY-MM-DD --end YYYY-MM-DD`
- **산출**: `project/artifacts/dataset_snapshot/kma_obs_asos_hourly.parquet` (경로는 설정 따름).

아래 **§2.2.5**부터는 PhotoRec에서 연속 복구된 원문을 그대로 이어 붙인다.

#### 2.2.5 AWS 시간 관측 이력 수집 (신규 구현)

**목적**: ASOS 미설치 지역 보완, 공간 해상도 향상

> `weather_api/` 프로젝트는 이 API를 구현하지 않았다. **신규 작성이 필요하다.**

**⚠️ 공공데이터포털 AWS API (15057084) 사용 불가 이유**:
- `Aws1miInfoService/getAws1miList` — **1분 자료만 제공, 최근 2일 이내만 조회 가능** → 과거 이력 수집 불가
- **공공기관 전용** 서비스 (방재기상업무 수행 목적, 일반 기업 사용 불가)

**대체 수집 방법**:

| 방법 | 엔드포인트 | 특징 |
|------|-----------|------|
| 기상청 API허브 (권장) | `https://apihub.kma.go.kr/api/typ01/url/awsh.php` | 시간통계, 별도 키 필요 |
| 기상자료개방포털 파일셋 | `https://data.kma.go.kr/data/grnd/selectAwsRltmList.do` | 수동 다운로드, CSV |

**API 정보 (기상청 API허브 사용 시)**:
- 서비스명: 방재기상관측(AWS) → 3. AWS 시간통계 자료 조회
- URL: `https://apihub.kma.go.kr/api/typ01/url/awsh.php`
- 공공데이터포털 URL: https://www.data.go.kr/data/15057084/openapi.do (참고용)
- 인증키: **apihub.kma.go.kr 에서 별도 발급** (공공데이터포털 키와 다름)

```python
# src/weather/aws_collector.py
# ASOSCollector와 동일한 패턴, BASE_URL만 변경

class AWSCollector:
    # ⚠️ 공공데이터포털 AWS 방재기상관측 API 엔드포인트.
    # 실제 서비스명은 'AwsHrlyInfoService' 일 수 있음 — API 포털에서 확인 후 교체 필요.
    BASE_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    
    # 구현 패턴은 ASOSCollector와 동일
    # 주요 컬럼: tm, stnId, ta(기온), rn(강수량), ws(풍속), wd(풍향), hm(습도)
    # AWS는 일사/일조 관측이 없는 경우가 많음
```

**주의 사항**:
- AWS는 `si`(일사), `ss`(일조) 미관측 지점이 대부분 → ASOS 일사를 공간 보간으로 보완
- 지점 수가 많아 전체 수집 시간이 길 수 있음 → chunk + sleep 전략 필수
- 저장: `artifacts/dataset_snapshot/kma_obs_aws_hourly.parquet`

---

#### 2.2.6 KMA 격자 좌표 변환 및 지점 매핑 (기존 구현 재사용)

**목적**: 각 site 좌표를 ASOS/AWS 지점 및 단기예보 격자에 연결

**격자 변환** — `WeatherAPI.get_grid_coordinates()` 직접 재사용:

```python
# src/weather/kma_mapping.py

import sys
sys.path.insert(0, "weather_api")  # 또는 복사 경로
from kma_weather_api import WeatherAPI   # get_grid_coordinates 재사용

import pandas as pd
import math

_weather_api = WeatherAPI("")   # 격자 변환만 쓸 때는 키 불필요

def latlon_to_kma_grid(latitude: float, longitude: float) -> tuple[int, int]:
    """
    위경도 → 기상청 5km 격자 (nx, ny) 변환.
    WeatherAPI.get_grid_coordinates() 재사용.
    """
    return _weather_api.get_grid_coordinates(latitude, longitude)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """두 위경도 간 거리 (km)"""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_site_to_kma_mapping(
    plant_meta: pd.DataFrame,       # cid_seq, latitude, longitude  ← plant_meta.parquet 컬럼명
    asos_stations: pd.DataFrame,    # stnId, lat, lon  (asos_stations.csv에서 로드)
    aws_stations: pd.DataFrame,
) -> pd.DataFrame:
    """
    각 site(cid_seq)에 대해 가장 가까운 ASOS 지점, AWS 지점, 예보 격자(nx, ny) 연결.

    plant_meta.parquet의 식별자는 site_id가 아닌 cid_seq 이다.
    asos_stations.csv 는 weather_collector/asos_stations.csv 에 번들로 포함된다.
    
    반환 컬럼:
        cid_seq, fcst_nx, fcst_ny,
        asos_stn_id, dist_to_asos_km,
        aws_stn_id,  dist_to_aws_km
    """
    rows = []
    for _, site in plant_meta.iterrows():
        nx, ny = latlon_to_kma_grid(site["latitude"], site["longitude"])
        
        # 최근접 ASOS 지점
        asos_dist = asos_stations.apply(
            lambda r: haversine_km(site["latitude"], site["longitude"], r["lat"], r["lon"]),
            axis=1
        )
        nearest_asos = asos_stations.iloc[asos_dist.idxmin()]
        
        # 최근접 AWS 지점
        aws_dist = aws_stations.apply(
            lambda r: haversine_km(site["latitude"], site["longitude"], r["lat"], r["lon"]),
            axis=1
        )
        nearest_aws = aws_stations.iloc[aws_dist.idxmin()]
        
        rows.append({
            "cid_seq":          site["cid_seq"],          # ← site_id → cid_seq
            "fcst_nx":          nx,
            "fcst_ny":          ny,
            "asos_stn_id":      nearest_asos["stnId"],
            "dist_to_asos_km":  asos_dist.min(),
            "aws_stn_id":       nearest_aws["stnId"],
            "dist_to_aws_km":   aws_dist.min(),
        })
    
    return pd.DataFrame(rows)
```

**실행**:
```bash
python src/weather/kma_mapping.py \
  --plant-meta artifacts/dataset_snapshot/plant_meta.parquet \
  --output project/artifacts/dataset_snapshot/site_to_kma_grid.csv
```

저장: `artifacts/dataset_snapshot/site_to_kma_grid.csv`

---

#### 2.2.7 일출/일몰 정보 (기존 `SunMoonAPI` 재사용)

`weather_api/weather_api.py`의 `SunMoonAPI.calculate_sunrise_sunset()` 은 pvlib 없이도 사용 가능한 천문 계산 구현이다. pvlib의 `get_solarposition()`이 주 구현이지만, API 환경 없이 빠르게 검증할 때 대체로 활용한다.

```python
# pvlib이 설치된 환경에서는 pvlib 우선 사용
# pvlib 미설치 환경 또는 빠른 검증 시:
from kma_weather_api import SunMoonAPI   # weather_api.py 복사본

sun_api = SunMoonAPI("")   # 계산 방식은 키 불필요
sunrise_sunset = sun_api.calculate_sunrise_sunset(latitude=37.5, longitude=126.9)
# 반환: {"sunrise": "0607", "sunset": "1842", ...}  (HHMM 형식)
```

---

#### 2.2.8 수집 절차 실행 순서

```
1. 환경변수 설정 (.env 파일)
2. site_to_kma_grid.csv 생성      ← kma_mapping.py
3. ASOS 이력 수집                 ← asos_collector.py  (수 시간 ~ 수일 소요 가능)
4. AWS 이력 수집                  ← aws_collector.py
5. ERA5 학습용 기상 이력 수집      ← era5_collector.py  (CDS API, 전략 B 참고)
6. 단기예보 수집 시작 (실시간 누적) ← kma_forecast_collector.py (스케줄러에 등록)
7. 수집 완료 후 parquet 검증      ← notebooks/01_data_eda.ipynb
```

**API 호출 제한 주의**:
- 공공데이터포털 API는 일일 호출 제한이 있다 (서비스별 상이, 통상 1,000~10,000 건/일)
- 장기 이력 수집은 **여러 날에 걸쳐 분할 수집** 또는 **기상자료개방포털 bulk 다운로드** 병행 고려
- 실패 기록: `logs/weather_collection_failures.log`에 날짜·지점·에러 메시지 기록

---

### 2.4 ERA5 재분석 데이터 수집 (학습용 미래 기상 입력 — 전략 B)

#### 의사결정 배경 (전략 B 채택)

PV 발전량 예측 모델의 훈련·추론에 사용할 **미래 기상 입력(NWP 데이터)** 전략을 다음과 같이 결정하였다.

| 구분 | 데이터 소스 | 사유 |
|------|------------|------|
| **훈련** | ERA5 재분석 (바이어스 교정) | 2022~현재까지 일관된 과거 기상 이력 제공, CDS API로 자동 수집 가능 |
| **추론 (실시간)** | 기상청 단기예보 API | 현재 시점 이후 최대 3일 예보 제공, 이미 수집 파이프라인 구현됨 |
| **테스트** | ERA5 / 단기예보 **모두** | 두 소스 간 성능 차이를 정량 비교 |

**일사량 변수 제외 결정**:
- 기상청 단기예보에는 일사량 변수가 없다
- ERA5의 `ssrd`(일사량)는 관측 대비 **+23~40% 고편향**으로 학습-추론 분포 불일치가 심함
- 따라서 훈련·추론 모두에서 일사량을 NWP 입력으로 사용하지 않는다
- 일사량 정보는 **pvlib clear-sky + ASOS `icsr` 실측** 으로 대체한다

**바이어스 교정 필요성**:
- ERA5 기온: 관측 대비 ±2~4°C 편향 존재
- ERA5 풍속: 평균 -23% 저편향
- 학습 전 ASOS 관측값 기준 **quantile mapping** 또는 **선형 회귀** 교정 적용

#### 수집 대상 변수

| ERA5 변수명 | 설명 | 단위 | 추론 시 대응 (단기예보) | 단위 변환 |
|------------|------|------|----------------------|---------|
| `2m_temperature` (`t2m`) | 2m 기온 | K | `tmp` (℃) | − 273.15 |
| `10m_u_component_of_wind` (`u10`) | 동서 풍속 | m/s | `wsd` + `vec` | 벡터→크기/방향 |
| `10m_v_component_of_wind` (`v10`) | 남북 풍속 | m/s | — | 동일 |
| `2m_dewpoint_temperature` (`d2m`) | 이슬점 기온 | K | `reh` (%) | Magnus 공식 |
| `total_precipitation` (`tp`) | 강수량 | m | `pcp` (mm) | × 1000 |
| `total_cloud_cover` (`tcc`) | 전운량 | 0~1 | `sky` (1/3/4) | 임계값 매핑 |
| ~~`surface_solar_radiation_downwards`~~ | ~~일사량~~ | — | ~~없음~~ | **제외** |

#### 수집 범위

```
기간   : 2022-01-01 ~ 현재 (연 단위 분할 수집)
영역   : 한반도 (N=38.5°, W=126.0°, S=33.0°, E=130.0°)
해상도 : 0.25° × 0.25° (~28km)
시간   : 매 정시 (0~23h)
포맷   : NetCDF → 사후 parquet 변환
출력   : artifacts/dataset_snapshot/era5_nwp_input.parquet
```

#### 구현 위치

```
dataset/weather_collector/era5_collector.py   ← 수집 + 바이어스 교정
python run.py era5 --year 2022 2023 2024      ← CLI
```

#### 바이어스 교정 전략

```
1단계: ERA5 격자 → PV site 위경도로 bilinear interpolation (nearest-neighbor 가능)
2단계: 각 변수별 ASOS 관측(kma_obs_asos_hourly.parquet)과 비교
3단계: quantile mapping (일사량 제외 변수 모두)
4단계: 교정된 ERA5를 era5_nwp_bias_corrected.parquet 으로 저장
```

#### 테스트 시 비교 방법

```python
# 테스트 셋에서 두 가지 NWP 입력으로 각각 추론
results_era5  = model.predict(test_features_era5)   # ERA5 기반 입력
results_fcst  = model.predict(test_features_fcst)   # 기상청 단기예보 기반 입력
results_true  = test_labels

metrics = {
    "ERA5_input":  compute_metrics(results_era5, results_true),
    "KMA_fcst_input": compute_metrics(results_fcst, results_true),
}
# 두 결과를 벤치마크 리포트에 모두 기록
```

---

### 2.5 태양 위치 계산 (pvlib + SunMoonAPI 보조)

**목적**: 시각/위치 기반 파생 feature 생성 (leakage 없는 미래 feature로도 활용 가능)

**구현 전략**:
- **주 구현**: `pvlib` — solar zenith/elevation/azimuth + clear-sky irradiance (Ineichen 모델)
- **보조/검증용**: `weather_api/` 의 `SunMoonAPI.calculate_sunrise_sunset()` — pvlib 없이 빠른 일출/일몰 확인

```python
# src/features/solar_position.py

import pvlib
import pandas as pd
import sys

def compute_solar_features(
    latitude: float,
    longitude: float,
    timestamps: pd.DatetimeIndex,
    altitude_m: float = 0.0
) -> pd.DataFrame:
    """
    pvlib 기반 태양 위치 및 맑은하늘 복사량 계산.
    
    Returns:
        DataFrame with columns:
            solar_zenith      - 천정각 (°)
            solar_elevation   - 고도각 (°)
            solar_azimuth     - 방위각 (°)
            clearsky_ghi      - 맑은하늘 수평면 전일사 (W/m²)  [Ineichen 모델]
            clearsky_dni      - 맑은하늘 직달일사 (W/m²)
            clearsky_dhi      - 맑은하늘 산란일사 (W/m²)
            is_daytime        - solar_elevation > 5° 여부 (bool)
            sunrise_kst       - 일출 시각 (KST datetime)
            sunset_kst        - 일몰 시각 (KST datetime)
    """
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz="Asia/Seoul",
        altitude=altitude_m
    )
    solar_pos = location.get_solarposition(timestamps)
    clearsky = location.get_clearsky(timestamps, model="ineichen")
    
    result = pd.DataFrame(index=timestamps)
    result["solar_zenith"]    = solar_pos["apparent_zenith"]
    result["solar_elevation"] = solar_pos["apparent_elevation"]
    result["solar_azimuth"]   = solar_pos["azimuth"]
    result["clearsky_ghi"]    = clearsky["ghi"]
    result["clearsky_dni"]    = clearsky["dni"]
    result["clearsky_dhi"]    = clearsky["dhi"]
    result["is_daytime"]      = result["solar_elevation"] > 5.0
    
    # 일별 일출/일몰 계산
    dates = pd.DatetimeIndex(timestamps.normalize().unique())
    sunrise_sunset = location.get_sun_rise_set_transit(dates, method="spa")
    # timestamps에 일출/일몰 시각 매핑
    date_map = sunrise_sunset.reindex(timestamps, method="ffill")
    result["sunrise_kst"] = date_map["sunrise"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    result["sunset_kst"]  = date_map["sunset"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    
    return result


def compute_solar_features_fallback(
    latitude: float,
    longitude: float,
    date_str: str   # "YYYYMMDD"
) -> dict:
    """
    pvlib 미설치 환경 또는 빠른 검증용.
    weather_api/weather_api.py 의 SunMoonAPI.calculate_sunrise_sunset() 활용.
    
    Returns:
        {"sunrise": "0607", "sunset": "1842", ...}  # HHMM 형식
    """
    sys.path.insert(0, "weather_api")
    from kma_weather_api import SunMoonAPI
    
    api = SunMoonAPI("")
    return api.calculate_sunrise_sunset(latitude=latitude, longitude=longitude)
```

**처리 방식**:
- site별로 연산 후 `site_id`를 키로 feature mart에 merge
- 저장: `artifacts/dataset_snapshot/solar_features_{site_id}.parquet`

**pvlib 설치 확인**:
```bash
pip install pvlib
python -c "import pvlib; print(pvlib.__version__)"
```

---

## 단계 3: 데이터 전처리 및 Feature Mart 구축

### 3.1 데이터 정제 파이프라인

```
원천 데이터
  ├── pv_raw_hourly
  ├── kma_obs_asos_hourly
  ├── kma_obs_aws_hourly
  └── kma_fcst_shortterm
       ↓
[3.1] 정제 (결측/이상치 처리)
       ↓
[3.2] 기상 join (ASOS/AWS → site 매핑 기반)
       ↓
[3.3] Forecast join (leakage 검증 포함)
       ↓
[3.4] 파생 feature 계산 (solar, rolling, calendar)
       ↓
[3.5] 정규화 (train 통계 기반)
       ↓
Feature Mart (site_id × base_time × horizon)
```

### 3.2 결측값 처리

```python
# src/features/preprocessing.py

def impute_pv(df: pd.DataFrame, max_linear_gap_hours: int = 1) -> pd.DataFrame:
    """
    PV 발전량 결측 처리.
    - 1시간 이하 연속 결측: 선형 보간
    - 초과 구간: NaN 유지 (학습 시 마스킹)
    """
    df["is_imputed"] = df["pv_power_kw"].isna()
    df["pv_power_kw"] = df["pv_power_kw"].interpolate(
        method="linear", limit=max_linear_gap_hours, limit_direction="both"
    )
    return df

def clip_pv(df: pd.DataFrame) -> pd.DataFrame:
    """
    물리적 허용 범위 클리핑.
    - capacity_kw 초과 값 → capacity_kw로 클리핑
    - 야간(is_daytime=False) 음수 값 → 0 클리핑
    """
    df["pv_power_kw"] = df["pv_power_kw"].clip(lower=0, upper=df["capacity_kw"])
    nighttime_mask = ~df["is_daytime"]
    df.loc[nighttime_mask, "pv_power_kw"] = df.loc[nighttime_mask, "pv_power_kw"].clip(lower=0)
    return df
```

### 3.3 Forecast Leakage 방지 Join

```python
# src/features/forecast_join.py

def join_forecast_no_leakage(
    base_times: pd.Series,         # 예측 기준 시각 t0
    forecast_df: pd.DataFrame,      # kma_fcst_shortterm (issue_time, target_time, features)
    horizon_hours: list = [24, 48, 72]
) -> pd.DataFrame:
    """
    각 base_time t0에 대해:
      1. issue_time <= t0 를 만족하는 최신 forecast run 선택
      2. 해당 run에서 target_time in [t0+1, ..., t0+H] 를 join
    
    leakage 검증:
      - join 후 target_time > t0 확인
      - issue_time <= t0 확인
      - 위반 row가 있으면 AssertionError 발생
    """
    results = []
    for t0 in base_times:
        # 최신 issue run 선택
        valid_runs = forecast_df[forecast_df["issue_time"] <= t0]
        latest_run = valid_runs["issue_time"].max()
        
        run_data = forecast_df[forecast_df["issue_time"] == latest_run]
        
        for h in horizon_hours:
            target = t0 + pd.Timedelta(hours=h)
            row = run_data[run_data["target_time"] == target]
            ...
        
        # leakage 검증 assertion
        assert (result["target_time"] > result["base_time"]).all(), "Leakage detected"
        assert (result["issue_time"] <= result["base_time"]).all(), "Future forecast used"
        
        results.append(result)
    return pd.concat(results)
```

### 3.4 파생 Feature 계산

```python
# src/features/derived_features.py

CALENDAR_FEATURES = ["hour", "dayofweek", "month", "dayofyear", "is_holiday"]
ROLLING_WINDOWS = [24, 72, 168]   # hours

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    # 한국 공휴일: holidays 라이브러리 또는 사전 정의 목록 사용
    ...
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """결측 마스크 적용 후 rolling 통계 계산"""
    for w in ROLLING_WINDOWS:
        df[f"pv_roll_mean_{w}h"] = (
            df["pv_power_kw"].rolling(w, min_periods=int(w * 0.8)).mean()
        )
        df[f"pv_roll_std_{w}h"] = (
            df["pv_power_kw"].rolling(w, min_periods=int(w * 0.8)).std()
        )
    df["pv_lag_24h"] = df["pv_power_kw"].shift(24)
    df["pv_lag_168h"] = df["pv_power_kw"].shift(168)
    return df
```

### 3.5 정규화 및 Feature Mart 저장

```python
# src/features/feature_mart_builder.py

def build_feature_mart(
    split_manifest: dict,
    output_dir: str = "project/artifacts/feature_mart/"
):
    """
    Feature Mart 생성 절차:
    1. train 구간으로 scaler 학습 (z-score: mean/std)
    2. train/valid/test 구간 각각 변환
    3. 결과를 site별 parquet으로 저장
    
    저장 구조:
        feature_mart/
          train/
            {site_id}.parquet
          valid/
            {site_id}.parquet
          test/
            {site_id}.parquet
          scaler_stats.json   # train 구간 mean/std (test 적용용)
    
    키 구조: (site_id, base_time, horizon)
    """
    ...
```

**scaler_stats.json 형식**:
```json
{
  "version": "1.0",
  "fit_end": "YYYY-MM-DD",
  "features": {
    "ta": {"mean": 12.5, "std": 9.2},
    "ws": {"mean": 2.1, "std": 1.8}
  }
}
```

### 3.6 데이터 분할 전략 — per-site adaptive split (★ 채택)

> 구현: `dataset/preprocessor/feature_mart_builder.py` — `per_site_split()` 함수

PV 발전 예측의 핵심 신호는 **태양 위치 (solar_elevation)**로, `month × dayofyear × hour`에 의해 결정된다.  
특정 연도의 태양 위치 분포는 다른 연도와 물리적으로 동일하므로, 연도 간 경계보다 **계절 패턴 학습**이 중요하다.

**global split의 문제점**:

```
전체 100 site 중 60 site가 2025년 이후 설치
  → global train_end=2024-12-31 기준: 이 60개 site의 train 데이터 = 0행
  → 60% site는 학습에 참여 불가 (검증·테스트에만 사용)
```

**per-site split 방식**:

```
[site A: 2022-01 ~ 2026-04]
  active_start                             active_end
       |── train (70%) ──|── valid (15%) ──|── test (15%) ──|

[site B: 2025-01 ~ 2026-04]
  active_start              active_end
       |── train (70%) ──|── valid ──|── test ──|
```

- 활성 기간: `first_valid(normalized_power) ~ last_valid(normalized_power)`
- 비율 70/15/15 적용 → 모든 site가 train/valid/test 데이터 보유
- 최소 split 기준: 500h/split 미달 시 해당 site 제외

```bash
# 빌드
python dataset/preprocessor/run.py build --split-mode per_site

# 품질 검사
python dataset/preprocessor/run.py quality-check \
  --feature-mart-dir project/artifacts/feature_mart_per_site
```

출력:
- `project/artifacts/feature_mart_per_site/` — 분할된 feature mart
- `project/artifacts/per_site_split_manifest.json` — site별 분할 경계 기록

**평가 주의사항**: site마다 test 기간이 다르므로, 모델 비교 시 **site별 지표를 각각 산출한 뒤 평균**해야 한다 (집계 편향 방지).

### 3.7 Track B Feature Mart (미래 공변량 wide fan)

> 단계 3 본문은 PhotoRec 복구 시점 이전에 작성되었으며, 이 절은 **트랙 B(입력 설계 확장)** 대응으로 추가되었다.

- **목적**: `feature_mart_per_site`에 **t0에서 허용되는 미래 horizon 공변량**을 `fcst_{변수}_{h:03d}` 형태로 붙인 **`feature_mart_track_b_per_site`**를 만든다.
- **구현 모듈 (복구됨)**: `dataset/preprocessor/track_b_forecast_join.py`, `track_b_enrich_mart.py`, CLI는 `dataset/preprocessor/run.py enrich-track-b`.
- **CLI 예시** (저장소 루트 `time_forecasting/` 기준, `dataset/preprocessor` 에서 실행):

```bash
cd dataset/preprocessor

# ERA5 hourly → fcst_* (기본: era5_native 컬럼명)
python run.py enrich-track-b \
  --input-mart-dir ../../project/artifacts/feature_mart_per_site \
  --join-mode era5_hourly_valid \
  --fcst-schema era5_native \
  --horizon-max 72

# 단기 슬롄 proxy(tmp, pcp, sky 등)로 fan을 채울 때
python run.py enrich-track-b \
  --input-mart-dir ../../project/artifacts/feature_mart_per_site \
  --join-mode era5_hourly_valid \
  --fcst-schema shortterm_aligned \
  --horizon-max 72

# split_manifest.yaml 날짜·경계 재기록
python run.py update-manifest
```

- **fan 스키마 점검** (저장소 루트에서): `python project/scripts/scan_fcst_parquet_health.py --mart project/artifacts/feature_mart_track_b_per_site`
- **학습 시 통합 입력(L+H)**: `project/src/train/train_tslib_model.py` 의 `--merge-future-nwp-into-encoder-input`, `--future-nwp-variable-names` — Dataset은 `project/src/datasets/pv_dataset.py`.
- **빌드 스크립트**(선택): `project/scripts/build_track_b_mart.sh` 가 위 `enrich-track-b`를 호출한다면 동일 효과.
- **누수 방지**: `dataset/preprocessor/track_b_forecast_join.py` — hourly 모드는 `valid_time = t0 + h`, issue/target 모드는 §3.3과 동일한 `issue_time`/`target_time` 규칙.
- **EDA 노트북**: `project/notebooks/02_data_eda_track_b.ipynb`
- **레이아웃·학습 상세**: [`project/docs/track_b_mart_layout_and_training_implementation.md`](project/docs/track_b_mart_layout_and_training_implementation.md)

---

## 단계 4: 데이터 검증 및 Sanity Check

이 단계를 통과하지 못하면 모델 학습을 시작하지 않는다.

### 4.1 데이터 품질 리포트

**운영 CLI**: `python dataset/preprocessor/run.py quality-check --feature-mart-dir project/artifacts/feature_mart_per_site` → `quality_report.json` (기준은 `run.py` 내장 로직, §4.1 통과 기준과 정합시키려면 스크립트·문서 중 한쪽을 기준으로 고정).

```python
# scripts/run_data_quality_check.py

def run_quality_check(feature_mart_dir: str) -> dict:
    report = {
        "n_sites": ...,
        "date_range": ...,
        "split": {"train": ..., "valid": ..., "test": ...},
        "missing_rate_by_site": ...,     # site별 결측률
        "missing_rate_by_feature": ...,  # feature별 결측률
        "pv_stats": {
            "capacity_factor_mean": ...,
            "daytime_zeros_ratio": ...,   # 낮 시간대 0 비율 (이상 여부 확인)
        },
        "n_samples_per_split": ...,
    }
    # report를 artifacts/data_quality_report.json에 저장
    return report
```

**통과 기준**:
- 전체 결측률 < 10%
- site별 PV 결측률 < 15%
- 낮 시간대(is_daytime=True) 0값 비율 < 30% (설비 정지 과다 여부 점검)
- train/valid/test 각 구간이 최소 8760시간(1년) 이상

### 4.2 Forecast Leakage 검증

```bash
# 독립 검증 스크립트로 실행
python scripts/verify_no_leakage.py \
  --feature-mart project/artifacts/feature_mart/ \
  --manifest project/artifacts/split_manifest.yaml
```

**검증 내용**:
1. 모든 row에서 `issue_time <= base_time` 확인
2. 모든 row에서 `target_time > base_time` 확인
3. test 구간의 PV target이 train 구간과 겹치지 않는지 확인
4. 결과를 `project/artifacts/leakage_check_result.json`에 저장 (pass/fail + 위반 건수)

### 4.3 시각화 검증 (Notebook)

`notebooks/01_data_eda.ipynb`에서 아래 항목 확인:

- [ ] site별 발전 프로파일 (일별 peak 패턴 정상 여부)
- [ ] 계절별 clear-sky vs 실측 발전량 산점도
- [ ] 기상 feature 분포 (이상 outlier 없는지)
- [ ] train/valid/test 경계에서 급격한 분포 변화 없는지

---

## 단계 5: Docker 환경 구성

> NVIDIA Driver, Docker, NVIDIA Container Toolkit은 이미 설치된 상태.  
> 이 단계에서는 **이미지 빌드 → 패키지 설치 → GPU 동작 검증** 까지 수행한다.

### 5.0 파일 관리 원칙: 호스트 마운트 방식

모든 소스 코드, 설정 파일, 데이터, 학습 결과는 **호스트 디렉토리에 저장**한다.  
Docker는 GPU/패키지 실행 환경만 제공하며, 파일 시스템은 호스트가 소유한다.

```
project/                       ← 호스트 (항상 접근 가능)
  src/          → /workspace/src               (볼륨 마운트)
  conf/         → /workspace/conf              (볼륨 마운트)
  artifacts/    → /workspace/artifacts         (볼륨 마운트)
  artifacts/models/ → /models                 (볼륨 마운트)
  vendor/TSLib/ → /workspace/vendor/TSLib      (볼륨 마운트, 호스트에서 코드 확인 가능)
  docker/       → 이미지 정의만 (Dockerfile, requirements)
```

**이미지 내부**에는 Python 패키지(`pip install`)만 포함한다.  
TSLib은 `scripts/setup_vendor.sh`로 호스트의 `vendor/TSLib/`에 클론하여 볼륨 마운트로 제공한다.  
`src/`, `conf/`, `artifacts/`, `vendor/`는 Dockerfile에서 `COPY`하지 않는다.

| 위치 | 호스트 경로 | 컨테이너 경로 | 비고 |
|---|---|---|---|
| 소스 코드 | `project/src/` | `/workspace/src/` | 코드 수정 즉시 반영 |
| 설정 파일 | `project/conf/` | `/workspace/conf/` | yaml 수정 즉시 반영 |
| 데이터/결과 | `project/artifacts/` | `/workspace/artifacts/` | 학습 결과 호스트에 저장 |
| HF 모델 | `project/artifacts/models/` | `/models/` | 재다운로드 불필요 |
| TSLib 코드 | `project/vendor/TSLib/` | `/workspace/vendor/TSLib/` | 호스트에서 직접 열람 가능 |

**`docker run` 예시 경로**: 문서의 `-v $(pwd)/project/artifacts:...` 는 **저장소 루트(`time_forecasting/`)에서 실행**할 때를 가정한다. `docker compose -f project/docker/docker-compose.yml` 은 `project/` 를 context로 하므로 YAML 안의 `../artifacts` 가 동일한 호스트 경로를 가리킨다.

---

### 5.1 Docker 전략: 2개 이미지

모델별 의존성 충돌 분석 결과, **최소 2개 이미지**로 운영한다.

| 이미지 | 포함 모델 | 분리 이유 |
|---|---|---|
| `pv-benchmark/unified` | SegRNN, PatchTST, DLinear, LLaMA 3.2 1B, Gemma 4 E2B | `torch 2.4.1` + `transformers 5.x` 공통 사용 가능 |
| `pv-benchmark/time-llm` | Time-LLM (GPT-2 backbone) | `transformers==4.31.0` 강제 고정으로 분리 필수 |

**`llama`, `gemma`, `tslib`를 `unified`로 통합할 수 있는 이유**:

| 기존 이미지 | transformers 요구 | unified에서 |
|---|---|---|
| tslib | 미사용 | 충돌 없음 |
| llama | `>=4.45.0` | `transformers 5.x`로 만족 |
| gemma 4 | `transformers 5.x` 전용 | `transformers>=5.0.0`으로 만족 |
| time-llm | **`==4.31.0` (강제 고정)** | **분리 필수** |

`bitsandbytes` (LLaMA/Gemma용) 컴파일을 위해 `-devel` 베이스 이미지를 공통 사용한다.

### 5.2 디렉토리 구조 및 공통 파일 준비

```
docker/
  unified/
    Dockerfile
    requirements-unified.txt
  time_llm/
    Dockerfile
    requirements-time-llm.txt
  docker-compose.yml
vendor/
  TSLib/                       # scripts/setup_vendor.sh 로 클론 (git 미포함)
scripts/
  setup_vendor.sh              # TSLib 호스트 클론 스크립트
  build_all.sh
  verify_gpu.sh
docs/
  model_download_guide.md      # LLM 모델 다운로드 절차
.env                           # API 키, HF 토큰 (git 미포함)
```

**vendor 초기화** (이미지 빌드 전 최초 1회 실행):

```bash
bash scripts/setup_vendor.sh
```

**공통 `.env` 파일 생성** (프로젝트 루트):

```bash
# .env  —  git에 절대 커밋하지 않는다
cat > .env << 'EOF'
# 공공데이터포털 API 키 (디코딩된 값으로 입력)
WEATHER_API_KEY=여기에_입력

# Hugging Face 토큰 (Llama gate 통과 후 발급)
HF_TOKEN=hf_xxxx

# 학습 실험 추적 (선택)
WANDB_API_KEY=
MLFLOW_TRACKING_URI=
EOF
```

```bash
# .gitignore에 추가
echo ".env" >> .gitignore
echo "artifacts/models/" >> .gitignore
echo "vendor/" >> .gitignore
```

---

### 5.3 env-unified (SegRNN, PatchTST, DLinear, LLaMA 3.2 1B, Gemma 4 E2B)

> `bitsandbytes`(LLaMA/Gemma용)가 CUDA 헤더를 필요로 하므로 `-devel` 베이스 이미지를 사용한다.  
> `transformers>=4.50.0`은 LLaMA(`>=4.45.0`)와 Gemma(`>=4.50.0`) 요구사항을 모두 만족한다.

**`docker/unified/requirements-unified.txt`**:

```text
# 공통 기반
pandas>=2.1.0
pyarrow>=14.0.0
scikit-learn>=1.3.0
pvlib>=0.10.0
holidays>=0.46
pyyaml>=6.0
tqdm>=4.66.0
requests>=2.31.0
numpy>=1.26.0

# TSLib용 (SegRNN, PatchTST, DLinear)
einops>=0.7.0
reformer-pytorch>=1.4.4
matplotlib>=3.8.0

# LLaMA + Gemma용 (transformers>=4.50.0이 양쪽 모두 만족)
transformers>=4.50.0
peft>=0.12.0
bitsandbytes>=0.43.0
accelerate>=0.31.0
trl>=0.9.0
datasets>=2.20.0
```

**`docker/unified/Dockerfile`**:

```dockerfile
# SegRNN, PatchTST, DLinear, LLaMA 3.2 1B, Gemma 4 E2B 통합 이미지
# bitsandbytes CUDA 컴파일을 위해 -devel 베이스 사용
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl wget build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/unified/requirements-unified.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-unified.txt

# transformers 5.x(Gemma 4 지원)이 torch>=2.4 를 요구하므로
# 호스트 CUDA 드라이버(12.2)와 호환되는 torch 2.4.1+cu121 로 재고정
RUN pip install --no-cache-dir --force-reinstall \
    "torch==2.4.1+cu121" \
    "torchvision==0.19.1+cu121" \
    "torchaudio==2.4.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# bitsandbytes CUDA 컴파일 확인
RUN python -c "import bitsandbytes; print('bitsandbytes OK:', bitsandbytes.__version__)"

# src/, conf/, artifacts/, vendor/TSLib 는 런타임에 호스트 볼륨으로 마운트
# TSLib은 호스트에서 scripts/setup_vendor.sh 로 클론 후 볼륨 마운트로 제공
# HF 토큰도 런타임에 환경변수로 주입 (-e HF_TOKEN=xxx)

ENV PYTHONPATH=/workspace:/workspace/vendor/TSLib
```

> **TSLib을 이미지에 포함하지 않는 이유**: 호스트에서 직접 코드를 열람·수정하고  
> 컨테이너에 즉시 반영하기 위해 `vendor/TSLib/`를 볼륨 마운트로 제공한다.  
> TSLib의 `requirements.txt`에는 Python 3.11+ 전용 패키지가 포함되어 있어  
> Python 3.10 베이스에서 `pip install`이 불가능하다. 필요한 패키지(einops 등)는  
> `requirements-unified.txt`에서 이미 설치한다.
>
> **torch 버전 노트**: `reformer-pytorch` 의존성 체인이 `torch>=2.5`를 요구하여  
> 초기 `pip install` 중 `torch 2.11.0+cu130`이 설치된다.  
> 이후 force-reinstall 단계에서 `torch 2.4.1+cu121`로 덮어쓴다.  
> `hyper-connections`, `torch-einops-utils` 가 `torch>=2.5` 경고를 출력하나  
> 벤치마크 모델 동작에는 영향 없다.

**빌드**:

```bash
docker build -t pv-benchmark/unified:latest \
  -f docker/unified/Dockerfile .
```

**GPU 동작 검증**:

```bash
# CUDA 및 기본 패키지 확인
docker run --rm --gpus all pv-benchmark/unified:latest \
  python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device name  :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('PyTorch ver  :', torch.__version__)
import bitsandbytes; print('bitsandbytes :', bitsandbytes.__version__)
import transformers; print('transformers :', transformers.__version__)
"

# TSLib + peft 임포트 확인
docker run --rm --gpus all pv-benchmark/unified:latest \
  python -c "
import sys; sys.path.insert(0, '/workspace/vendor/TSLib')
from models.SegRNN import Model as SegRNN
from models.PatchTST import Model as PatchTST
from peft import LoraConfig
print('TSLib models + peft loaded OK')
"
```

---

### 5.4 env-time-llm (Time-LLM)

> Time-LLM은 `transformers==4.31.0` 고정이 필요하므로 `unified`와 분리된 독립 이미지를 사용한다.  
> torch 버전도 2.1.x 계열로 고정한다.

**`docker/time_llm/requirements-time-llm.txt`**:

```text
transformers==4.31.0
accelerate>=0.21.0
einops>=0.7.0
matplotlib>=3.8.0
pandas>=2.1.0
pyarrow>=14.0.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.66.0
requests>=2.31.0
pvlib>=0.10.0
holidays>=0.46
```

**`docker/time_llm/Dockerfile`**:

```dockerfile
# Time-LLM 전용: torch 2.1.x + transformers 4.31.0 고정
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/time_llm/requirements-time-llm.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-time-llm.txt

# Time-LLM 공식 저장소 설치
RUN git clone https://github.com/KimMeen/Time-LLM.git /workspace/TimeLLM \
    && cd /workspace/TimeLLM \
    && pip install --no-cache-dir -r requirements.txt || true

# GPT-2 backbone 사전 캐시 (빌드 시 다운로드 → 런타임 인터넷 불필요)
RUN python -c "
from transformers import GPT2Model, GPT2Tokenizer
GPT2Model.from_pretrained('gpt2')
GPT2Tokenizer.from_pretrained('gpt2')
print('GPT-2 backbone cached')
"

# src/, conf/, artifacts/ 는 런타임에 호스트 볼륨으로 마운트
# (-v $(pwd)/project/src:/workspace/src 등) — 이미지에 포함하지 않는다

ENV PYTHONPATH=/workspace:/workspace/TimeLLM
```

**빌드**:

```bash
docker build -t pv-benchmark/time-llm:latest \
  -f docker/time_llm/Dockerfile .
```

**동작 검증**:

```bash
docker run --rm --gpus all pv-benchmark/time-llm:latest \
  python -c "
import torch
from transformers import GPT2Model
print('CUDA available:', torch.cuda.is_available())
print('transformers ver:', __import__('transformers').__version__)
model = GPT2Model.from_pretrained('gpt2')
print('GPT-2 loaded OK')
"
```

---

### 5.5 LLaMA + Gemma 모델 사전 다운로드

> **상세 절차**: `docs/model_download_guide.md` 참조  
> (HF 토큰 설정, gate 승인, 트러블슈팅 포함)

`unified` 이미지에서 HF 모델을 host 볼륨에 다운로드한다.  
모델은 호스트의 `artifacts/models/` 에 저장되므로 도커 없이도 직접 확인 가능하다.

**전제 조건**:
1. `.env` 파일의 `HF_TOKEN` 에 실제 토큰 입력 완료
2. [Llama 3.2 gate](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 및 [Gemma 4 E2B gate](https://huggingface.co/google/gemma-4-e2b-it) 승인 완료

**LLaMA 3.2 1B 다운로드**:

```bash
docker run --rm --gpus all \
  --env-file .env \
  -v $(pwd)/project/artifacts/models:/models \
  pv-benchmark/unified:latest \
  python -c "
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
token    = os.environ['HF_TOKEN']
model_id = 'meta-llama/Llama-3.2-1B-Instruct'
AutoTokenizer.from_pretrained(model_id, token=token, cache_dir='/models')
AutoModelForCausalLM.from_pretrained(model_id, token=token, cache_dir='/models', torch_dtype='auto')
print('Llama 3.2 1B download complete')
"
```

**Gemma 4 E2B 다운로드**:

```bash
docker run --rm --gpus all \
  --env-file .env \
  -v $(pwd)/project/artifacts/models:/models \
  pv-benchmark/unified:latest \
  python -c "
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
token    = os.environ['HF_TOKEN']
model_id = 'google/gemma-4-e2b-it'
AutoTokenizer.from_pretrained(model_id, token=token, cache_dir='/models')
AutoModelForCausalLM.from_pretrained(model_id, token=token, cache_dir='/models', torch_dtype='auto')
print('Gemma 4 E2B download complete')
"
```

---

### 5.6 docker-compose.yml (전체 컨테이너 관리)

학습 실행 시 일일이 `docker run` 명령을 타이핑하는 대신 `docker compose run <service>` 로 실행한다.

**`docker/docker-compose.yml`**:

```yaml
# docker/docker-compose.yml
# 사용법:
#   docker compose -f docker/docker-compose.yml run --rm unified python ...
#   docker compose -f docker/docker-compose.yml run --rm time-llm python ...

x-common-volumes: &common-volumes
  - ../src:/workspace/src
  - ../conf:/workspace/conf
  - ../artifacts:/workspace/artifacts
  - ../vendor/TSLib:/workspace/vendor/TSLib    # 호스트에서 코드 수정 가능

x-gpu-deploy: &gpu-deploy
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]

services:

  # ─── 통합 이미지 (SegRNN, PatchTST, DLinear, LLaMA, Gemma) ────
  unified:
    image: pv-benchmark/unified:latest
    build:
      context: ..
      dockerfile: docker/unified/Dockerfile
    volumes:
      - ../src:/workspace/src
      - ../conf:/workspace/conf
      - ../artifacts:/workspace/artifacts
      - ../artifacts/models:/models             # HF 모델 캐시
      - ../vendor/TSLib:/workspace/vendor/TSLib  # 호스트에서 코드 수정 가능
    <<: *gpu-deploy
    env_file: ../.env
    working_dir: /workspace
    shm_size: "16gb"

  # ─── Time-LLM ──────────────────────────────────────────────
  time-llm:
    image: pv-benchmark/time-llm:latest
    build:
      context: ..
      dockerfile: docker/time_llm/Dockerfile
    volumes: *common-volumes
    <<: *gpu-deploy
    env_file: ../.env
    working_dir: /workspace
    shm_size: "8gb"

  # ─── CPU 벤치마크 전용 (GPU 비활성화) ─────────────────────
  unified-cpu:
    image: pv-benchmark/unified:latest
    volumes:
      - ../src:/workspace/src
      - ../conf:/workspace/conf
      - ../artifacts:/workspace/artifacts
      - ../artifacts/models:/models
      - ../vendor/TSLib:/workspace/vendor/TSLib
    environment:
      - CUDA_VISIBLE_DEVICES=""
      - OMP_NUM_THREADS=16
      - MKL_NUM_THREADS=16
    env_file: ../.env
    working_dir: /workspace
    cpus: "16"

  time-llm-cpu:
    image: pv-benchmark/time-llm:latest
    volumes: *common-volumes
    environment:
      - CUDA_VISIBLE_DEVICES=""
      - OMP_NUM_THREADS=16
      - MKL_NUM_THREADS=16
    working_dir: /workspace
    cpus: "16"
```

---

### 5.7 전체 이미지 빌드 순서

```bash
# 프로젝트 루트에서 실행
cd /path/to/project

# 1. unified (SegRNN, PatchTST, DLinear, LLaMA, Gemma 통합)
docker build -t pv-benchmark/unified:latest \
  -f docker/unified/Dockerfile .

# 2. time-llm (transformers==4.31.0 고정 — 분리 필수)
docker build -t pv-benchmark/time-llm:latest \
  -f docker/time_llm/Dockerfile .

# 전체 이미지 목록 확인
docker images | grep pv-benchmark
```

**`scripts/build_all.sh`**:

```bash
#!/usr/bin/env bash
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

echo "=== [1/2] unified (SegRNN + PatchTST + DLinear + LLaMA + Gemma) ==="
docker build -t pv-benchmark/unified:latest -f docker/unified/Dockerfile .

echo "=== [2/2] time-llm ==="
docker build -t pv-benchmark/time-llm:latest -f docker/time_llm/Dockerfile .

echo "=== Build complete ==="
docker images | grep pv-benchmark
```

```bash
chmod +x scripts/build_all.sh
./scripts/build_all.sh
```

---

### 5.8 전체 이미지 GPU 동작 검증

```bash
# scripts/verify_gpu.sh
#!/usr/bin/env bash
set -e

IMAGES=(
  "pv-benchmark/unified:latest"
  "pv-benchmark/time-llm:latest"
)

for IMG in "${IMAGES[@]}"; do
  echo -n "[$IMG] GPU check ... "
  RESULT=$(docker run --rm --gpus all "$IMG" \
    python -c "import torch; print('OK' if torch.cuda.is_available() else 'FAIL')" 2>&1)
  echo "$RESULT"
done
```

```bash
chmod +x scripts/verify_gpu.sh
./scripts/verify_gpu.sh
```

**기대 출력**:
```
[pv-benchmark/unified:latest] GPU check ... OK
[pv-benchmark/time-llm:latest] GPU check ... OK
```

---

### 5.9 CPU 벤치마크 실행 방식

GPU 학습 완료 후 CPU 추론 벤치마크는 `CUDA_VISIBLE_DEVICES=""` 오버라이드로 실행한다.

```bash
# compose 방식 (GPU 없이 실행)
docker compose -f docker/docker-compose.yml run --rm unified-cpu \
  python src/benchmark/run_cpu_benchmark.py --model segrnn

# 또는 docker run 방식 (직접 오버라이드)
docker run --rm \
  -e CUDA_VISIBLE_DEVICES="" \
  -e OMP_NUM_THREADS=16 \
  -e MKL_NUM_THREADS=16 \
  --cpus="16" \
  -v $(pwd)/project/src:/workspace/src \
  -v $(pwd)/project/conf:/workspace/conf \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/artifacts/models:/models \
  pv-benchmark/unified:latest \
  python src/benchmark/run_cpu_benchmark.py --model segrnn

# CPU 환경에서 GPU가 완전히 비활성화됐는지 확인
docker run --rm \
  -e CUDA_VISIBLE_DEVICES="" \
  pv-benchmark/unified:latest \
  python -c "import torch; assert not torch.cuda.is_available(), 'GPU still visible!'; print('CPU-only mode: OK')"
```

---

### 5.10 더미 데이터 스모크 테스트 (환경 최종 검증)

> **목적**: 실제 데이터 없이도 전 모델의 forward pass / 추론 속도를 검증한다.  
> 환경 설정 완료 선언 전 마지막 게이트. **GPU/CPU 양쪽 모두 통과해야 한다.**

#### 더미 데이터 스펙

실제 Feature Mart와 동일한 형태로 생성한다.

| 항목 | 값 | 근거 |
|------|----|----|
| 총 rows | 26,280 | 3년 × 365일 × 24h |
| train / valid / test | 17,520 / 4,392 / 4,368 | 70% / 17% / 17% 시간 기반 분할 |
| Feature 수 | 26 | normalized_power + 기상 9 + 태양 3 + rolling 6 + lag 2 + calendar 5 |
| seq_len (lookback) | 8,760 | 1년 |
| pred_len (horizon) | 24 | 1일 |
| 저장 위치 | `artifacts/dummy_feature_mart/` | |

#### 5.10.1 더미 데이터 생성

모든 명령은 `project/` 루트에서 실행한다.

```bash
cd /disk1/krems/time_forecasting/project

docker run --rm \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/scripts:/workspace/scripts \
  pv-benchmark/unified:latest \
  python scripts/gen_dummy_data.py
```

**기대 출력**:
```
  [train]  17520 rows × 26 cols → artifacts/dummy_feature_mart/train/site_dummy.parquet
  [valid]   4392 rows × 26 cols → artifacts/dummy_feature_mart/valid/site_dummy.parquet
  [test]   4368 rows × 26 cols → artifacts/dummy_feature_mart/test/site_dummy.parquet
✓ 더미 Feature Mart 생성 완료
```

---

#### 5.10.2 스모크 테스트 스크립트 공통 구조

아래 모든 테스트는 동일한 패턴으로 구성된다:

1. `artifacts/dummy_feature_mart/train/site_dummy.parquet` 로드
2. `DataLoader` 구성 (seq_len=8760, pred_len=24, batch_size=32)
3. 모델 1 epoch 학습 → 시간 측정
4. 단일 배치 추론 latency 측정 (10회 평균)
5. GPU 메모리 peak 기록 (GPU 모드 한정)

---

#### 5.10.3 DLinear — GPU 스모크 테스트

```bash
docker run --rm --gpus all \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python -c "
import time, torch
import pandas as pd
import numpy as np

# ── 데이터 로드 ──────────────────────────────
df = pd.read_parquet('/workspace/artifacts/dummy_feature_mart/train/site_dummy.parquet')
data = torch.tensor(df.values, dtype=torch.float32)

SEQ_LEN, PRED_LEN = 8760, 24
N_FEAT = data.shape[1]

def make_batches(data, seq_len, pred_len, batch_size=32):
    xs, ys = [], []
    for i in range(0, len(data) - seq_len - pred_len, batch_size):
        end = min(i + batch_size, len(data) - seq_len - pred_len)
        for j in range(i, end):
            xs.append(data[j:j+seq_len])
            ys.append(data[j+seq_len:j+seq_len+pred_len, 0:1])
        if len(xs) >= batch_size:
            yield torch.stack(xs), torch.stack(ys)
            xs, ys = [], []

# ── DLinear 모델 (TSLib) ──────────────────────
import sys; sys.path.insert(0, '/workspace/vendor/TSLib')
from models.DLinear import Model

class Args:
    task_name = "long_term_forecast"
    seq_len = SEQ_LEN
    pred_len = PRED_LEN
    enc_in = N_FEAT
    dec_in = N_FEAT
    c_out = N_FEAT
    individual = False
    moving_avg = 25

model = Model(Args()).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# ── 1 epoch 학습 ──────────────────────────────
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for x, y in make_batches(data, SEQ_LEN, PRED_LEN):
    x, y = x.cuda(), y.cuda()
    pred = model(x, None, None, None)[:, -PRED_LEN:, 0:1]
    loss = criterion(pred, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
epoch_sec = time.time() - t0

# ── 추론 latency (10회 평균) ──────────────────
model.eval()
x_single = data[:SEQ_LEN].unsqueeze(0).cuda()
times = []
with torch.no_grad():
    for _ in range(10):
        t = time.time(); model(x_single, None, None, None); torch.cuda.synchronize()
        times.append((time.time()-t)*1000)

peak_mem = torch.cuda.max_memory_allocated() / 1024**3

print(f'[DLinear GPU]')
print(f'  1 epoch time  : {epoch_sec:.1f}s')
print(f'  infer latency : {np.mean(times):.1f} ms (avg 10회)')
print(f'  peak GPU mem  : {peak_mem:.2f} GB')
print('  PASS')
"
```

---

#### 5.10.4 SegRNN — GPU 스모크 테스트

```bash
docker run --rm --gpus all \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python -c "
import time, torch, sys, numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/vendor/TSLib')
from models.SegRNN import Model

df = pd.read_parquet('/workspace/artifacts/dummy_feature_mart/train/site_dummy.parquet')
data = torch.tensor(df.values, dtype=torch.float32)
SEQ_LEN, PRED_LEN, N_FEAT = 8760, 24, data.shape[1]

class Args:
    task_name = "long_term_forecast"
    seq_len = SEQ_LEN
    pred_len = PRED_LEN
    enc_in = N_FEAT
    d_model = 512
    dropout = 0.1
    seg_len = 48
    num_class = 0

model = Model(Args()).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
step = 0
for i in range(0, len(data) - SEQ_LEN - PRED_LEN - 32, 32):
    x = torch.stack([data[j:j+SEQ_LEN] for j in range(i, i+32)]).cuda()
    y = torch.stack([data[j+SEQ_LEN:j+SEQ_LEN+PRED_LEN, 0:1] for j in range(i, i+32)]).cuda()
    pred = model(x, None, None, None)[:, -PRED_LEN:, 0:1]
    loss = criterion(pred, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    step += 1
    if step >= 10: break   # 속도 측정용 10 step
epoch_sec = time.time() - t0

model.eval()
x_s = data[:SEQ_LEN].unsqueeze(0).cuda()
times = []
with torch.no_grad():
    for _ in range(10):
        t = time.time(); model(x_s, None, None, None); torch.cuda.synchronize()
        times.append((time.time()-t)*1000)

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f'[SegRNN GPU]')
print(f'  10 steps time : {epoch_sec:.1f}s')
print(f'  infer latency : {np.mean(times):.1f} ms (avg 10회)')
print(f'  peak GPU mem  : {peak_mem:.2f} GB')
print('  PASS')
"
```

---

#### 5.10.5 PatchTST — GPU 스모크 테스트

```bash
docker run --rm --gpus all \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python -c "
import time, torch, sys, numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/vendor/TSLib')
from models.PatchTST import Model

df = pd.read_parquet('/workspace/artifacts/dummy_feature_mart/train/site_dummy.parquet')
data = torch.tensor(df.values, dtype=torch.float32)
SEQ_LEN, PRED_LEN, N_FEAT = 8760, 24, data.shape[1]

class Args:
    task_name = "long_term_forecast"
    seq_len = SEQ_LEN
    pred_len = PRED_LEN
    enc_in = N_FEAT
    d_model = 128
    n_heads = 4
    e_layers = 2
    d_ff = 256
    dropout = 0.1
    factor = 1
    activation = "gelu"
    num_class = 0

model = Model(Args(), patch_len=16, stride=8).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for step, i in enumerate(range(0, len(data) - SEQ_LEN - PRED_LEN - 32, 32)):
    x = torch.stack([data[j:j+SEQ_LEN] for j in range(i, i+32)]).cuda()
    y = torch.stack([data[j+SEQ_LEN:j+SEQ_LEN+PRED_LEN, 0:1] for j in range(i, i+32)]).cuda()
    pred = model(x, None, None, None)[:, -PRED_LEN:, 0:1]
    loss = criterion(pred, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if step >= 5: break
epoch_sec = time.time() - t0

model.eval()
x_s = data[:SEQ_LEN].unsqueeze(0).cuda()
times = []
with torch.no_grad():
    for _ in range(10):
        t = time.time(); model(x_s, None, None, None); torch.cuda.synchronize()
        times.append((time.time()-t)*1000)

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f'[PatchTST GPU]')
print(f'  6 steps time  : {epoch_sec:.1f}s')
print(f'  infer latency : {np.mean(times):.1f} ms (avg 10회)')
print(f'  peak GPU mem  : {peak_mem:.2f} GB')
print('  PASS')
"
```

---

#### 5.10.6 Time-LLM — GPU 스모크 테스트

```bash
docker run --rm --gpus all \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/time-llm:latest \
  python -c "
import time, torch, sys, numpy as np
import pandas as pd
from transformers import GPT2Model

df = pd.read_parquet('/workspace/artifacts/dummy_feature_mart/train/site_dummy.parquet')
data = torch.tensor(df.values[:, 0:1], dtype=torch.float32)  # target only
SEQ_LEN, PRED_LEN = 512, 24   # Time-LLM 권장 입력길이

# GPT-2 backbone 로드 확인
t0 = time.time()
backbone = GPT2Model.from_pretrained('gpt2').cuda()
load_sec = time.time() - t0

# 단순 forward pass (임베딩 레이어 입력 형태 확인)
backbone.eval()
x_s = data[:SEQ_LEN].unsqueeze(0).cuda()
times = []
with torch.no_grad():
    dummy_ids = torch.zeros(1, SEQ_LEN//10, dtype=torch.long).cuda()
    for _ in range(5):
        t = time.time()
        out = backbone(inputs_embeds=torch.randn(1, 64, 768).cuda())
        torch.cuda.synchronize()
        times.append((time.time()-t)*1000)

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f'[Time-LLM GPU]')
print(f'  backbone load : {load_sec:.1f}s')
print(f'  forward latency: {np.mean(times):.1f} ms (avg 5회, seq=64)')
print(f'  peak GPU mem  : {peak_mem:.2f} GB')
print('  PASS')
"
```

---

#### 5.10.7 LLaMA 3.2 1B — GPU 스모크 테스트

```bash
docker run --rm --gpus all \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/artifacts/models:/models \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python -c "
import time, torch, glob, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 로컬 모델 로드
snapshot = sorted(glob.glob('/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*'))[-1]
tokenizer = AutoTokenizer.from_pretrained(snapshot)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    snapshot, dtype=torch.float16, device_map='auto'
)
load_sec = time.time() - t0

# LoRA 설정 (학습 파라미터 수 확인)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
    target_modules=['q_proj', 'v_proj'], lora_dropout=0.05, bias='none'
)
model = get_peft_model(base_model, lora_config)
trainable, total = model.get_nb_trainable_parameters()

# 더미 시계열 → 텍스트 프롬프트 변환 후 forward pass
dummy_prompt = (
    'The following is a time-series of solar power generation: '
    + ', '.join([f'{v:.2f}' for v in np.random.rand(24).tolist()])
    + '. Predict the next 24 hours:'
)
inputs = tokenizer(dummy_prompt, return_tensors='pt', truncation=True, max_length=512)
inputs = {k: v.to('cuda') for k, v in inputs.items()}

times = []
model.eval()
with torch.no_grad():
    for _ in range(3):
        t = time.time()
        out = model(**inputs, labels=inputs['input_ids'])
        torch.cuda.synchronize()
        times.append((time.time()-t)*1000)

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f'[LLaMA 3.2 1B GPU]')
print(f'  model load    : {load_sec:.1f}s')
print(f'  LoRA params   : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
print(f'  forward latency: {np.mean(times):.0f} ms (avg 3회)')
print(f'  peak GPU mem  : {peak_mem:.2f} GB')
print('  PASS')
"
```

---

#### 5.10.8 Gemma 4 E2B — GPU 스모크 테스트

```bash
docker run --rm --gpus all \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/artifacts/models:/models \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python -c "
import time, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

gemma_path = '/models/gemma-4-e2b-it'
tokenizer = AutoTokenizer.from_pretrained(gemma_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    gemma_path, dtype=torch.float16, device_map='auto'
)
load_sec = time.time() - t0

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
    target_modules=['q_proj', 'v_proj'], lora_dropout=0.05, bias='none'
)
model = get_peft_model(base_model, lora_config)
trainable, total = model.get_nb_trainable_parameters()

dummy_prompt = (
    'Time series solar power: '
    + ', '.join([f'{v:.2f}' for v in np.random.rand(24).tolist()])
    + '. Forecast next 24h:'
)
inputs = tokenizer(dummy_prompt, return_tensors='pt', truncation=True, max_length=512)
inputs = {k: v.to('cuda') for k, v in inputs.items()}

times = []
model.eval()
with torch.no_grad():
    for _ in range(3):
        t = time.time()
        out = model(**inputs, labels=inputs['input_ids'])
        torch.cuda.synchronize()
        times.append((time.time()-t)*1000)

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f'[Gemma 4 E2B GPU]')
print(f'  model load    : {load_sec:.1f}s')
print(f'  LoRA params   : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
print(f'  forward latency: {np.mean(times):.0f} ms (avg 3회)')
print(f'  peak GPU mem  : {peak_mem:.2f} GB')
print('  PASS')
"
```

---

#### 5.10.9 CPU 모드 추론 latency 테스트

GPU 학습 후 CPU 추론 속도 사전 검증. `CUDA_VISIBLE_DEVICES=""`로 강제 CPU 전환.

```bash
# DLinear + SegRNN + PatchTST CPU 추론 (unified-cpu 서비스)
docker run --rm \
  -e CUDA_VISIBLE_DEVICES="" \
  -e OMP_NUM_THREADS=16 \
  -e MKL_NUM_THREADS=16 \
  --cpus="16" \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python -c "
import time, torch, sys, numpy as np
import pandas as pd

assert not torch.cuda.is_available(), 'GPU should be disabled'
sys.path.insert(0, '/workspace/vendor/TSLib')

df = pd.read_parquet('/workspace/artifacts/dummy_feature_mart/test/site_dummy.parquet')
data = torch.tensor(df.values, dtype=torch.float32)
SEQ_LEN, PRED_LEN, N_FEAT = 8760, 24, data.shape[1]
x = data[:SEQ_LEN].unsqueeze(0)   # batch=1

results = {}

# DLinear
from models.DLinear import Model as DL
class DArgs:
    seq_len=SEQ_LEN; pred_len=PRED_LEN; enc_in=N_FEAT
    individual=False; moving_avg=25
m = DL(DArgs()).eval()
times = []
with torch.no_grad():
    for _ in range(20): t=time.time(); m(x); times.append((time.time()-t)*1000)
results['DLinear'] = {'p50': float(np.percentile(times,50)), 'p95': float(np.percentile(times,95))}

# SegRNN
from models.SegRNN import Model as SR
class SArgs:
    seq_len=SEQ_LEN; pred_len=PRED_LEN; enc_in=N_FEAT; d_model=512
    dropout=0.1; seg_len=48; rnn_type='gru'; dec_way='pmf'
    channel_id=False; revin=False
m = SR(SArgs()).eval()
times = []
with torch.no_grad():
    for _ in range(5): t=time.time(); m(x); times.append((time.time()-t)*1000)
results['SegRNN'] = {'p50': float(np.percentile(times,50)), 'p95': float(np.percentile(times,95))}

# PatchTST
from models.PatchTST import Model as PT
class PArgs:
    seq_len=SEQ_LEN; pred_len=PRED_LEN; enc_in=N_FEAT; c_out=1
    d_model=128; n_heads=4; e_layers=2; d_ff=256; dropout=0.1
    fc_dropout=0.1; head_dropout=0.0; patch_len=16; stride=8
    padding_patch='end'; revin=True; affine=False; subtract_last=False
    decomposition=False; kernel_size=25; individual=False
m = PT(PArgs()).eval()
times = []
with torch.no_grad():
    for _ in range(5): t=time.time(); m(x); times.append((time.time()-t)*1000)
results['PatchTST'] = {'p50': float(np.percentile(times,50)), 'p95': float(np.percentile(times,95))}

print('[CPU 추론 latency (batch=1, seq_len=8760, pred_len=24)]')
for model_name, r in results.items():
    print(f'  {model_name:<12} p50={r[\"p50\"]:7.1f}ms  p95={r[\"p95\"]:7.1f}ms')
print('  CPU mode: PASS')
"
```

---

#### 5.10.10 LLM CPU 추론 latency 테스트

```bash
# LLaMA CPU 추론
docker run --rm \
  -e CUDA_VISIBLE_DEVICES="" \
  -e OMP_NUM_THREADS=16 \
  --cpus="16" \
  -v $(pwd)/project/artifacts/models:/models \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python -c "
import time, torch, glob, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

assert not torch.cuda.is_available()

snapshot = sorted(glob.glob('/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*'))[-1]
tokenizer = AutoTokenizer.from_pretrained(snapshot)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32)
model.eval()
load_sec = time.time() - t0

prompt = 'Solar power forecast: ' + ', '.join([f'{v:.2f}' for v in np.random.rand(24)])
inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)

times = []
with torch.no_grad():
    for _ in range(3):
        t = time.time()
        out = model(**inputs, labels=inputs['input_ids'])
        times.append((time.time()-t)*1000)

print(f'[LLaMA 3.2 1B CPU]')
print(f'  load time     : {load_sec:.1f}s')
print(f'  forward p50   : {np.percentile(times,50):.0f} ms')
print(f'  forward p95   : {np.percentile(times,95):.0f} ms')
print('  CPU mode: PASS')
"
```

---

#### 5.10.11 스모크 테스트 결과 요약표

> **수행일**: 2026-04-22 / **전체 13개 테스트 PASS** ✓  
> **측정 조건**: batch=1, warm-up 1회 제외. TS 모델 seq_len=8,760 / LLM seq=64 (GPU), 상세 조건은 `project/docs/smoke_test_report.md` 참조.

##### GPU 추론 결과

| 모델 | 이미지 | GPU avg (ms) | GPU p95 (ms) | Peak GPU mem | GPU util avg/peak | 상태 |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| DLinear | unified | **0.10** | 0.12 | 0.16 GB | 48% / 50% | ✅ PASS |
| SegRNN | unified | **4.09** | 4.11 | 10.55 GB | *(측정 예정)* | ✅ PASS |
| PatchTST | unified | **11.33** | 11.97 | 6.39 GB | 99% / 100% | ✅ PASS |
| Time-LLM (GPT-2) | time-llm | **2.5** | 2.6 | 0.50 GB | *(측정 예정)* | ✅ PASS |
| LLaMA 3.2 1B | unified | **7** | 8 | 2.73 GB | 73% / 88% | ✅ PASS |
| Gemma 4 E2B | unified | **22** | 23 | 10.82 GB | 66% / 78% | ✅ PASS |

##### CPU 추론 결과

| 모델 | 파라미터 | CPU p50 (ms) | CPU p95 (ms) | CPU util avg/peak | 운영 투입 판단 |
|------|:---:|:---:|:---:|:---:|:---:|
| DLinear | — | **0.4** | 0.4 | 56% / 63% | ✅ CPU 운영 적합 |
| SegRNN | — | **67.8** | 72.6 | 31% / 62% | ✅ CPU 운영 가능 |
| PatchTST | — | **369.9** | 376.5 | 40% / 51% | ⚠️ 최적화 검토 |
| Time-LLM (GPT-2, seq=64) | 117M | **27.8** | 29.7 | *(측정 예정)* | ✅ CPU 운영 가능 |
| Time-LLM (GPT-2, seq=512) | 117M | **166.2** | 179.8 | *(측정 예정)* | ⚠️ seq 길이 주의 |
| LLaMA 3.2 1B | 1.24B | **590** | 611 | *(측정 예정)* | ❌ GPU 전용 권장 |
| Gemma 4 E2B | 5.11B | **130,310** | 130,345 | *(측정 예정)* | ❌ CPU 절대 불가 |

##### Time-LLM 백본 확장 비교 (GPU, seq=64)

| 백본 | 파라미터 | GPU avg (ms) | GPU p95 (ms) | Peak GPU mem | 비고 |
|------|:---:|:---:|:---:|:---:|:---|
| gpt2 | 124M | **2.36** | 2.43 | 0.50 GB | 논문 기본값 |
| gpt2-medium | 355M | **4.75** | 4.79 | 1.38 GB | GPU/속도 균형 권장 |
| gpt2-large | 774M | **8.62** | 8.68 | 3.04 GB | |
| gpt2-xl | 1,558M | **13.95** | 13.98 | 6.05 GB | |
| **LLaMA 3.2 1B** | **1,238M** | **4.4** | 4.5 | 2.32 GB | gpt2-xl 대비 3배 빠름 (GQA) |

##### 주요 이슈 요약

| # | 모델 | 이슈 | 해결 |
|---|------|------|------|
| 1 | DLinear / SegRNN / PatchTST | `Args`에 `task_name` 속성 누락 | `task_name = 'long_term_forecast'` 추가 |
| 2 | PatchTST | `activation`, `factor` 속성 누락 | `activation='gelu'`, `factor=1` 추가 |
| 3 | SegRNN | `seg_len=48`이 `seq_len=8760` 나누지 못함 | `seg_len=24`로 변경 (8760/24=365) |
| 4 | PatchTST | batch=32, seq=8760 → attention OOM (14.87 GB 요구) | 학습 batch=2로 축소, 추론 batch=1 정상 |
| 5 | Gemma 4 E2B | vision encoder LoRA 타겟 불가 | `target_modules` 정규식으로 language_model만 지정 |
| 6 | CPU 테스트 | test parquet 4,368행 < seq_len 8,760 | train parquet (17,520행)으로 교체 |

> CPU 추론이 지나치게 느린 모델(Gemma 130s/batch, LLaMA 600ms/batch)은 단계 12 CPU 벤치마크에서 제외하거나  
> 별도 최적화(INT8 quantization, ONNX export) 적용을 검토한다.  
> 상세 결과: `project/docs/smoke_test_report.md`

---

## 단계 6: Baseline 실험

모든 baseline은 `env-unified` 컨테이너 내에서 실행한다.

### 6.1 Seasonal Naive

**방법**: 예측 시각과 동일한 시각의 **직전 주** (7일 전 동일 시각) 값을 예측값으로 사용

```python
# src/train/baseline_seasonal_naive.py

def seasonal_naive_predict(history: np.ndarray, horizon: int) -> np.ndarray:
    """
    history: [T] shape, 과거 시계열
    horizon: 예측할 시간 수
    Returns: [horizon] shape 예측값
    """
    lag = 7 * 24  # 168시간
    return history[-lag: -lag + horizon] if len(history) >= lag + horizon else history[-horizon:]
```

**실행**:
```bash
docker run --rm \
  -v $(pwd)/project/src:/workspace/src \
  -v $(pwd)/project/conf:/workspace/conf \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  pv-benchmark/unified:latest \
  python src/train/baseline_seasonal_naive.py \
    --manifest /workspace/artifacts/split_manifest.yaml \
    --feature-mart /workspace/artifacts/feature_mart/ \
    --output /workspace/artifacts/training_runs/seasonal_naive/
```

**저장 산출물**:
```
project/artifacts/training_runs/seasonal_naive/
  predictions_test_{horizon}h.parquet   # 예측값
  metrics_test_{horizon}h.json          # MAE, RMSE, nRMSE, daytime_MAE 등
```

### 6.2 Persistence

**방법**: 예측 시각과 동일한 시각의 **직전 24시간** 값을 그대로 반복

```python
def persistence_predict(history: np.ndarray, horizon: int) -> np.ndarray:
    lag = 24
    tile_count = (horizon // lag) + 1
    return np.tile(history[-lag:], tile_count)[:horizon]
```

### 6.3 DLinear (TSLib)

**역할**: 딥러닝 기반 최소 baseline

```bash
docker run --rm --gpus all \
  -v $(pwd)/project/src:/workspace/src \
  -v $(pwd)/project/conf:/workspace/conf \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python /workspace/vendor/TSLib/run.py \
    --model DLinear \
    --data custom \
    --root_path /workspace/artifacts/feature_mart/train/ \
    --seq_len 8760 \
    --pred_len 24 \
    --enc_in [feature_count] \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --seed 42
```

**3 seed 반복**:
```bash
for SEED in 42 123 2024; do
  docker run --rm --gpus all \
    -v $(pwd)/project/src:/workspace/src \
    -v $(pwd)/project/conf:/workspace/conf \
    -v $(pwd)/project/artifacts:/workspace/artifacts \
    -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
    pv-benchmark/unified:latest \
    python /workspace/vendor/TSLib/run.py --model DLinear \
      --seed $SEED \
      --checkpoints /workspace/artifacts/training_runs/dlinear/seed_$SEED/
done
```

---

## 단계 7: SegRNN 실험

### 7.1 학습

```bash
# segment_length 후보: 24, 48
# horizon 후보: 24, 48, 72

docker run --rm --gpus all \
  -v $(pwd)/project/src:/workspace/src \
  -v $(pwd)/project/conf:/workspace/conf \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python /workspace/vendor/TSLib/run.py \
    --model SegRNN \
    --data custom \
    --root_path /workspace/artifacts/feature_mart/train/ \
    --seq_len 8760 \
    --pred_len 24 \
    --seg_len 48 \
    --enc_in [feature_count] \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_epochs 30 \
    --patience 5 \
    --seed 42 \
    --checkpoints /workspace/artifacts/training_runs/segrnn/seg48_h24_seed42/
```

**실험 조합**:

| seg_len | pred_len | seed |
|:---:|:---:|:---:|
| 24 | 24 | 42, 123, 2024 |
| 48 | 24 | 42, 123, 2024 |
| 24 | 48 | 42, 123, 2024 |
| 48 | 48 | 42, 123, 2024 |
| 24 | 72 | 42, 123, 2024 |
| 48 | 72 | 42, 123, 2024 |

총 **18개** 실험 run

### 7.2 평가

```bash
python src/benchmark/evaluate_model.py \
  --model-type segrnn \
  --checkpoint-dir project/artifacts/training_runs/segrnn/ \
  --feature-mart project/artifacts/feature_mart/test/ \
  --manifest project/artifacts/split_manifest.yaml \
  --output project/artifacts/training_runs/segrnn/eval_results.json
```

**평가 지표 계산**:
- MAE, RMSE, nRMSE (capacity 기준)
- daytime_MAE, daytime_nRMSE
- 일 누적 발전량 오차 (kWh)
- site별 평균 오차 분포

---

## 단계 8: PatchTST 실험

### 8.1 학습

```bash
# patch_len 후보: 24, 48
# stride 후보: 24, 48 (patch_len보다 클 수 없음 → 실질 조합 제한)

docker run --rm --gpus all \
  -v $(pwd)/project/src:/workspace/src \
  -v $(pwd)/project/conf:/workspace/conf \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python /workspace/vendor/TSLib/run.py \
    --model PatchTST \
    --data custom \
    --root_path /workspace/artifacts/feature_mart/train/ \
    --seq_len 8760 \
    --pred_len 24 \
    --patch_len 48 \
    --stride 24 \
    --n_heads 8 \
    --d_model 128 \
    --enc_in [feature_count] \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 30 \
    --patience 5 \
    --seed 42 \
    --checkpoints /workspace/artifacts/training_runs/patchtst/pl48_s24_h24_seed42/
```

**실험 조합**:

| patch_len | stride | pred_len | seed |
|:---:|:---:|:---:|:---:|
| 24 | 24 | 24, 48, 72 | 42, 123, 2024 |
| 48 | 24 | 24, 48, 72 | 42, 123, 2024 |
| 48 | 48 | 24, 48, 72 | 42, 123, 2024 |

총 **27개** 실험 run

### 8.2 다변량 feature 수 변화 실험 (선택)

feature 수를 줄여가며 정확도/속도 변화 관찰:
- `full`: 전체 feature
- `core`: PV + 기상 핵심 (기온, 풍속, 하늘상태, solar position)
- `minimal`: PV + solar position만

---

## 단계 9: Time-LLM 실험

### 9.1 환경 및 사전 주의 사항

- `env-time-llm` 컨테이너 사용 (`unified`와 분리 — `transformers==4.31.0` 고정 버전 충돌)
- GPT-2 backbone은 자동 다운로드되므로 인터넷 연결 필요
- `transformers` 버전을 Time-LLM repo requirements에 정확히 맞출 것

```bash
# backbone 모델 사전 다운로드 (빌드 시 이미 캐시됨 — 재다운로드 필요 시만 실행)
docker run --rm \
  -v $(pwd)/project/artifacts/models:/models \
  pv-benchmark/time-llm:latest \
  python -c "
from transformers import GPT2Model, GPT2Tokenizer
GPT2Model.from_pretrained('gpt2', cache_dir='/models')
GPT2Tokenizer.from_pretrained('gpt2', cache_dir='/models')
"
```

### 9.2 학습 — GPT-2 backbone

```bash
docker run --rm --gpus all \
  -v $(pwd)/project/src:/workspace/src \
  -v $(pwd)/project/conf:/workspace/conf \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/artifacts/models:/models \
  pv-benchmark/time-llm:latest \
  python /workspace/TimeLLM/run_main.py \
    --model TimeLLM \
    --backbone gpt2 \
    --gpt_layers 6 \
    --d_model 768 \
    --seq_len 8760 \
    --pred_len 24 \
    --patch_len 16 \
    --stride 8 \
    --data custom \
    --root_path /workspace/artifacts/feature_mart/train/ \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --train_epochs 20 \
    --llm_model_path /models \
    --checkpoints /workspace/artifacts/training_runs/time_llm_gpt2/h24_seed42/
```

### 9.3 평가 포인트

- 동일 horizon에서 PatchTST 대비 정확도/latency trade-off 표 작성
- CPU warm latency가 허용 범위(< 10초/site) 내인지 검증
- backbone 전체 forward pass 비용이 병목인지 프로파일링

---

## 단계 10: LLaMA 실험

### 10.1 사전 준비

```bash
# HF 토큰 환경변수 설정 (셸 세션에서 한번만)
export HF_TOKEN="hf_xxxx"

# 모델 다운로드 (이미 단계 5.4에서 완료된 경우 skip)
docker run --rm \
  -e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
  -v $(pwd)/project/artifacts/models:/models \
  pv-benchmark/unified:latest \
  python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct', cache_dir='/models', torch_dtype='auto')
AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct', cache_dir='/models')
"
```

### 10.2 Prompt 설계

```python
# src/datasets/llm_prompt_builder.py

SYSTEM_PROMPT = """당신은 태양광 발전 예측 전문 AI입니다.
주어진 과거 발전 데이터와 기상 예보를 바탕으로 
향후 {horizon}시간의 시간별 정규화 발전량(0~1)을 예측하세요."""

def build_user_prompt(
    site_id: str,
    capacity_kw: float,
    recent_168h_pv: list,       # 최근 7일 시간별 normalized_power
    monthly_stats: dict,         # 과거 12개월 월별 통계
    weather_forecast: list,      # 미래 horizon 시간의 예보
    calendar_info: dict,         # 예측 기간 달력 정보
    horizon: int
) -> str:
    return f"""
## 설비 정보
- site_id: {site_id}
- 정격용량: {capacity_kw:.1f} kW

## 최근 168시간 발전량 (normalized, 0~1)
{recent_168h_pv}

## 월별 평균 발전 통계 (최근 12개월)
{monthly_stats}

## 향후 {horizon}시간 기상 예보
{weather_forecast}

## 달력 정보
{calendar_info}

## 요청
향후 {horizon}시간의 시간별 정규화 발전량을 JSON 배열로만 출력하세요.
예시: [0.00, 0.00, 0.01, 0.05, ...]
반드시 {horizon}개의 값을 포함해야 합니다. 0 이상 1 이하의 값만 허용합니다.
"""
```

### 10.3 LoRA 학습

```python
# src/train/train_llama_lora.py 핵심 구조

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

# 학습 인자
training_args = TrainingArguments(
    output_dir="project/artifacts/training_runs/llama_lora/",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    seed=42,
)
```

```bash
docker run --rm --gpus all \
  -e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
  -v $(pwd)/project/src:/workspace/src \
  -v $(pwd)/project/conf:/workspace/conf \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/artifacts/models:/models \
  pv-benchmark/unified:latest \
  python src/train/train_llama_lora.py \
    --model-name-or-path /models/meta-llama/Llama-3.2-1B-Instruct \
    --feature-mart /workspace/artifacts/feature_mart/train/ \
    --horizon 24 \
    --seed 42 \
    --output-dir /workspace/artifacts/training_runs/llama_lora/h24_seed42/
```

### 10.4 출력 파싱 및 fallback 처리

```python
# src/infer/llm_output_parser.py

import json, re

def parse_llm_output(raw_text: str, expected_len: int) -> tuple[list, str]:
    """
    Returns:
        (values, status): status in ["ok", "partial", "failed"]
    """
    # 1차 시도: JSON 배열 직접 파싱
    match = re.search(r'\[[\d\s.,\-]+\]', raw_text)
    if match:
        try:
            values = json.loads(match.group())
            if len(values) == expected_len:
                valid = [max(0.0, min(1.0, v)) for v in values]
                return valid, "ok"
        except json.JSONDecodeError:
            pass
    
    # 2차 시도: 숫자 추출
    nums = re.findall(r'\d+\.?\d*', raw_text)
    if len(nums) >= expected_len:
        values = [float(n) for n in nums[:expected_len]]
        valid = [max(0.0, min(1.0, v)) for v in values]
        return valid, "partial"
    
    return [float("nan")] * expected_len, "failed"
```

**재시도 로직**:
```python
def infer_with_retry(model, tokenizer, prompt, expected_len, max_retries=2):
    for attempt in range(max_retries + 1):
        output = model.generate(...)
        values, status = parse_llm_output(output, expected_len)
        if status == "ok":
            return values, status, attempt
    return values, "failed", max_retries
```

---

## 단계 11: GEMMA 실험

### 11.1 사전 준비

```bash
# 모델 다운로드 (이미 단계 5.4에서 완료된 경우 skip)
docker run --rm \
  -e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
  -v $(pwd)/project/artifacts/models:/models \
  pv-benchmark/unified:latest \
  python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('google/gemma-4-e2b-it', cache_dir='/models', torch_dtype='auto')
AutoTokenizer.from_pretrained('google/gemma-4-e2b-it', cache_dir='/models')
"
```

### 11.2 LoRA/QLoRA 학습

LLaMA와 동일한 prompt 구조 및 학습 파이프라인을 사용한다.

```bash
docker run --rm --gpus all \
  -e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
  -v $(pwd)/project/src:/workspace/src \
  -v $(pwd)/project/conf:/workspace/conf \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/artifacts/models:/models \
  pv-benchmark/unified:latest \
  python src/train/train_gemma_lora.py \
    --model-name-or-path /models/google/gemma-4-e2b-it \
    --feature-mart /workspace/artifacts/feature_mart/train/ \
    --horizon 24 \
    --use-qlora true \
    --seed 42 \
    --output-dir /workspace/artifacts/training_runs/gemma_lora/h24_seed42/
```

**QLoRA 설정** (메모리 부족 시):
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### 11.3 비교 포인트

- 동일 프롬프트 구조에서 LLaMA 1B 대비 정확도/속도 비교
- QLoRA 적용 전후 정확도 차이 기록

---

## 단계 12: CPU 추론 벤치마크

모든 학습 완료 후 진행한다. 학습된 가중치를 CPU 전용 컨테이너 또는 동일 컨테이너의 GPU 비활성화 환경에서 실행한다.

### 12.1 환경 고정

```python
# src/benchmark/cpu_setup.py

import torch, os

def setup_cpu_benchmark():
    """CPU 벤치마크 재현성 보장을 위한 환경 고정"""
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["MKL_NUM_THREADS"] = "16"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU 완전 비활성화
    
    # 환경 검증
    assert not torch.cuda.is_available(), "GPU must be disabled for CPU benchmark"
    assert torch.get_num_threads() == 16
```

**컨테이너 실행 시 GPU 비활성화**:
```bash
# --gpus 플래그 없이 실행 (GPU 비활성화). WORKDIR=/workspace, PYTHONPATH에 TSLib 포함(이미지 기본).
docker run --rm \
  -e CUDA_VISIBLE_DEVICES="" \
  -w /workspace \
  -v $(pwd)/project/src:/workspace/src \
  -v $(pwd)/project/artifacts:/workspace/artifacts \
  -v $(pwd)/project/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python src/benchmark/run_cpu_benchmark.py ...
```

### 12.2 측정 스크립트

```python
# src/benchmark/run_cpu_benchmark.py

import time, psutil, os
import numpy as np

def measure_latency(
    model,
    sample_input,
    warmup_runs: int = 3,
    measure_runs: int = 20
) -> dict:
    setup_cpu_benchmark()
    
    # Warm-up
    for _ in range(warmup_runs):
        _ = model(sample_input)
    
    # 측정
    latencies_ms = []
    for _ in range(measure_runs):
        start = time.perf_counter()
        _ = model(sample_input)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000)
    
    return {
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
    }

def measure_cold_start(model_loader_fn, sample_input) -> dict:
    """모델 로드 + 첫 추론 시간 측정"""
    start = time.perf_counter()
    model = model_loader_fn()
    _ = model(sample_input)
    elapsed = (time.perf_counter() - start) * 1000
    return {"cold_start_ms": elapsed}

def measure_memory(model) -> dict:
    """피크 RSS RAM 측정"""
    proc = psutil.Process(os.getpid())
    baseline_mb = proc.memory_info().rss / 1024 / 1024
    # 모델 로드 후 측정
    peak_mb = proc.memory_info().rss / 1024 / 1024
    return {
        "peak_ram_mb": peak_mb,
        "model_overhead_mb": peak_mb - baseline_mb,
    }
```

### 12.3 배치 처리량 측정

```python
def measure_throughput(model, input_batch, batch_sizes=[1, 8, 16, 32]) -> dict:
    results = {}
    for bs in batch_sizes:
        batch = input_batch[:bs]
        start = time.perf_counter()
        for _ in range(10):
            _ = model(batch)
        elapsed = time.perf_counter() - start
        samples_per_sec = (bs * 10) / elapsed
        results[f"batch_{bs}_samples_per_sec"] = samples_per_sec
    return results
```

### 12.4 LLM 추가 측정 항목

```python
def measure_llm_reliability(model, tokenizer, prompts, horizon) -> dict:
    """LLM 출력 품질 및 신뢰성 측정"""
    ok, partial, failed = 0, 0, 0
    range_violations = 0
    
    for prompt in prompts:
        values, status, _ = infer_with_retry(model, tokenizer, prompt, horizon)
        if status == "ok": ok += 1
        elif status == "partial": partial += 1
        else: failed += 1
        
        violations = sum(1 for v in values if v < 0 or v > 1)
        range_violations += violations
    
    n = len(prompts)
    return {
        "parse_success_rate": ok / n,
        "partial_rate": partial / n,
        "failure_rate": failed / n,
        "range_violation_rate": range_violations / (n * horizon),
    }
```

### 12.5 결과 저장

```bash
# 모든 모델 CPU 벤치마크 실행 (unified 이미지 — SegRNN, PatchTST, DLinear, LLaMA, Gemma)
docker compose -f docker/docker-compose.yml run --rm unified-cpu \
  python src/benchmark/run_cpu_benchmark.py --model all

# Time-LLM CPU 벤치마크
docker compose -f docker/docker-compose.yml run --rm time-llm-cpu \
  python src/benchmark/run_cpu_benchmark.py --model time_llm

# 모든 모델 벤치마크 실행 후 결과 합산
python src/benchmark/aggregate_results.py \
  --input-dir project/artifacts/training_runs/ \
  --output project/artifacts/cpu_benchmark_report.json
```

```json
// artifacts/cpu_benchmark_report.json 형식
{
  "benchmark_date": "YYYY-MM-DD",
  "hardware": {
    "cpu": "...",
    "cores": 16,
    "ram_gb": ...
  },
  "results": {
    "segrnn": {
      "horizon_24h": {
        "cold_start_ms": ...,
        "warm_p50_ms": ...,
        "warm_p95_ms": ...,
        "batch_1_samples_per_sec": ...,
        "peak_ram_mb": ...
      }
    },
    "patchtst": { ... },
    "time_llm_gpt2": { ... },
    "llama_1b": { ... },
    "gemma_e2b": { ... }
  }
}
```

---

## 단계 13: 리더보드 집계 및 의사결정

### 13.1 정확도 리더보드 생성

```bash
python src/report/build_accuracy_leaderboard.py \
  --training-runs project/artifacts/training_runs/ \
  --manifest project/artifacts/split_manifest.yaml \
  --output project/artifacts/leaderboard.md
```

생성되는 테이블 (3 seed 평균 ± 표준편차):

| Model | Horizon | MAE | RMSE | nRMSE | Daytime MAE | Daily Energy Error |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal Naive | 24h | | | | | |
| Persistence | 24h | | | | | |
| DLinear | 24h | | | | | |
| SegRNN | 24h | | | | | |
| PatchTST | 24h | | | | | |
| Time-LLM (GPT-2) | 24h | | | | | |
| LLaMA 3.2 1B | 24h | | | | | |
| Gemma 4 E2B | 24h | | | | | |

### 13.2 CPU 벤치마크 리더보드

| Model | Horizon | Batch | Cold Start(s) | Warm p50(ms) | Warm p95(ms) | Throughput(samples/s) | Peak RAM(GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegRNN | 24h | 1 | | | | | |
| PatchTST | 24h | 1 | | | | | |
| Time-LLM (GPT-2) | 24h | 1 | | | | | |
| LLaMA 3.2 1B | 24h | 1 | | | | | |
| Gemma 4 E2B | 24h | 1 | | | | | |

### 13.3 종합 의사결정 점수

```python
# src/report/decision_score.py

WEIGHTS = {
    "accuracy":    0.50,   # daytime_nRMSE 역수 기반
    "cpu_latency": 0.20,   # warm p95 역수 기반
    "memory":      0.10,   # peak RAM 역수 기반
    "complexity":  0.20,   # 운영 복잡도 수동 점수 (0~10)
}

def compute_decision_score(metrics: dict, weights: dict = WEIGHTS) -> float:
    ...
```

| Model | Accuracy | CPU Latency | RAM | Complexity | Final Score | Decision |
|---|---:|---:|---:|---:|---:|---|
| SegRNN | | | | | | |
| PatchTST | | | | | | |
| Time-LLM | | | | | | |
| LLaMA | | | | | | |
| GEMMA | | | | | | |

### 13.4 최종 의사결정 기준

**예측 본체 선정**:
- 정확도 리더보드 상위 + CPU p95 latency < 운영 허용 임계값 동시 만족
- 운영 허용 임계값: warm p95 ≤ **5,000ms / site** (기본 기준, 실측 후 조정)

**설명/리포트 레이어 선정**:
- CPU latency가 허용 범위를 초과하는 LLM 모델 → 예측 본체 제외
- 출력 파싱 성공률 ≥ 95% 만족하는 LLM 후보만 리포트 레이어로 검토

---


## 입력 설계 확장 벤치마크

> **추가(2026-04-27)**: 동일 모델·평가 체계에서 **입력 구간·해상도만 바꾼** 비교 실험을 정의한다. 예보·미래 공변량이 포함되면 [§3.3 Forecast Leakage 방지 Join](#33-forecast-leakage-방지-join) 및 [§3.7 Track B Feature Mart](#37-track-b-feature-mart-미래-공변량-wide-fan)를 반드시 따른다.

### 범위·목표

- **무엇을 바꾸나**: 동일 `pred_len`(예: 24/48/72)과 동일 평가 파이프라인을 유지한 채, **입력 마트·lookback·시간 해상도**만 바꿔 정확도·운영 적합성을 비교한다.
- **무엇을 바꾸지 않나**: 지표 정의(`daytime_MAE` 등), per-site 집계 방식, seed 정책은 트랙 간 **고정**하는 것이 리더보드 해석을 단순화한다.
- **상세 설계**: wide fan·통합 텐서·채널 매핑·주의사항은 [`project/docs/track_b_mart_layout_and_training_implementation.md`](project/docs/track_b_mart_layout_and_training_implementation.md)가 정본이다. 본 절은 **운영 순서·CLI** 위주로 요약한다.

### `pv_model_benchmark_plan.md` §10 “트랙”과 혼동 금지

| 구분 | `pv_model_benchmark_plan.md` §10 | 본 문서 “트랙 A/B/C” |
|------|----------------------------------|----------------------|
| 목적 | LLM vs 시계열 **공정 비교**(입력 정보량 맞추기) | **마트·해상도** 실험축 (시간 hourly vs 일별, NWP 유무) |
| 트랙 B 의미 | 모든 모델에 동일한 요약·예보 텍스트/수치 묶음 제공 | **per-site mart + `fcst_*` wide fan** (`feature_mart_track_b_per_site`) |
| 운영 권고 | 운영은 §10 트랙 A 우선 | 벤치마크 후 **운영 배포용**은 별도 합의(대개 본 문서 트랙 A 마트) |

두 축을 **동시에** 쓸 경우 리더보드에 `plan_track`(A/B)와 `input_mode`(A/B/C)를 **둘 다** 남긴다.

### 트랙 B — 데이터·학습 연결 (요약)

| 단계 | 산출 / 진입점 | 비고 |
|------|----------------|------|
| 마트 | `project/artifacts/feature_mart_per_site/` | `fcst_*` 없음 → 기본 TSLib 학습 경로 |
| enrich | `dataset/preprocessor/run.py enrich-track-b` | `--join-mode`·`--fcst-schema`·`--horizon-max` — §3.7 예시 |
| 검증 | `project/scripts/scan_fcst_parquet_health.py` | fan 열·dtype·윈도 시뮬 |
| 리포트 | `track_b_build_report.json` | `fcst_value_cols`, `join_mode`, `forecast_path` |
| 학습(기본) | `project/src/train/train_tslib_model.py` | `fcst_*` 없이도 동작; mart에 fan이 있어도 **기본은 소비하지 않음** |
| 학습(merge) | 동일 + `--merge-future-nwp-into-encoder-input` | `seq_len`이 **L+H**로 잡히고 `enc_in=len(FEATURE)+1`; `--future-nwp-variable-names`로 fan 슬롄 제한 |

**조인 모드 요약** (`track_b_forecast_join.py`):

- **`era5_hourly_valid`**: `era5_nwp_input_raw` 시계열에서 **valid 시각 t0+h** 값을 채움(재분석 궤적; 실운영 단기와 다를 수 있음).
- **`issue_target`**: long 테이블에서 **`issue_time ≤ t0`** 규칙으로 후보를 줄인 뒤 `target_time`에 매칭 — §3.3과 동일한 누수 방지 계열.

**wide fan을 모델에 넣는 패턴** (TSLib 단일 입력 유지 관점):

| 패턴 | 텐서 | 비고 |
|------|------|------|
| 기본 | `x:[B,L,C]`, `y:[B,H,C]` | `fcst_*`는 디스크에만 존재해도 됨 |
| **A-1d (통합)** | `x:[B,L+H, len(FEATURE_COLS)+1]` | `--merge-future-nwp-into-encoder-input` — 채널·마스크 계약은 [아래 A-1d 절](#a-1d-merge-unified-tensor-and-channel-contract) 및 [정본: `track_b_mart_layout` §2.6](project/docs/track_b_mart_layout_and_training_implementation.md) |
| (향후) 이중 입력 | `x_hist` + `x_fcst` | `forward`·`collate_fn` 수정 필요 |

### A-1d merge: unified tensor and channel contract

`train_tslib_model.py` + `pv_dataset.SingleSiteDataset(merge_future_nwp_into_encoder_input=True)` 가 만드는 **통합 인코더 입력** 계약을 한 곳에 정리한다. (PhotoRec 본문에는 동일 수준이 없었고, 구현·[`track_b_mart_layout…`](project/docs/track_b_mart_layout_and_training_implementation.md) §2.6을 기준으로 복원·요약했다.)

1. **Shape**  
   - `L = --seq-len`, `H = --pred-len`, 통합 시간 길이 `T = L + H`.  
   - `x_unified`: **`[B, T, N]`** where `N = len(FEATURE_COLS) + 1`. 마지막 1채널은 **구간 플래그**(과거=1, 미래 NWP 슬롯=0). TSLib 쪽에서는 `configs.seq_len := T`, `enc_in := N`으로 맞춘다.

2. **행 `0 … L-1` (과거 실측 구간)**  
   - `FEATURE_COLS` 순서대로 **mart 실측·파생값**을 그대로 복사한다(기본 학습과 동일).  
   - **플래그 채널**은 전부 **1** — “이 스텝은 관측 기반 구간”임을 나타낸다(구간 마스크 역할).

3. **행 `L … T-1` (미래 H스텝, wide fan에서 전개)**  
   - 각 `h = 1…H`에 대해 `t_end`(윈도 마지막 실측 시각) 행의 **`fcst_{var}_{h:03d}`** 를 읽는다.  
   - **`var` → `FEATURE_COLS` 안의 동일 물리량 슬롄**에 쓴다. 매핑은 `pv_dataset.FUTURE_NWP_TO_FEATURE_COL_INDEX` (`tmp→t2m_c`, `reh→reh`, `sky→tcc` 등). 즉 **“실측에 있던 기상·격자 채널과 같은 인덱스”**에 미래 예보값을 넣는 방식이다.  
   - **`--future-nwp-variable-names`에 없는** fan 변수는 아예 읽지 않는다. 기본값에서 **`pty` / `pop` / `sno`는 제외**된다(실측 mart에 대응 열이 없거나 품질 계약이 별도인 슬롄 — 정본 §2.6.2 표 (2)와 동일).

4. **“실측에는 있는데 미래 fan에는 없는” 채널** (`pa`, `si`, `ss`, 캘린더·파생 등)  
   - 미래 `H`행의 해당 칸은 **0으로 둔다**. 동시에 **플래그=0**이므로, 모델은 “미래 구간 행”임을 알 수 있다.  
   - 정본 §2.6.3에서 말하듯 **0만으로는 “진짜 0℃”와 구분이 안 될 수 있어**, 엄밀히는 **채널별 유효 마스크**를 추가 채널로 넣는 설계가 권장된다. **현재 구현은 구간 플래그 1채널만** 넣는다(`pv_dataset` 참고).

5. **손실**  
   - `y`는 여전히 미래 `H`스텝의 타깃(예: `normalized_power` 포함 `FEATURE_COLS`); 손실은 기존과 같이 **미래 구간**에만 건다.

**주의 (양방향 attention·운영 시맨틱)** — 상세 논의는 [정본 §2.6.5](project/docs/track_b_mart_layout_and_training_implementation.md) :

- PatchTST 등 **시간축 양방향 attention**이면, 통합 입력에서 **미래 NWP 토큰이 과거 토큰과 양방향으로 보일 수 있다**. “예보를 이미 알고 전력을 맞추는” 설정인지, **순수 인과** 실험인지 팀에서 고정할 것.
- `shortterm_aligned`의 `sky`·`pop` 등은 **proxy**이며 단기 실운영 값과 동치가 아님.
- 기본 `--future-nwp-variable-names`는 `pty`/`pop`/`sno`를 제외한다(`pv_dataset.FUTURE_NWP_TO_FEATURE_COL_INDEX`와 동일 계약).

### 트랙 B 학습 CLI 예시 (merge 경로)

저장소 루트에서, Docker 없이 호스트 예시:

```bash
cd project
PYTHONPATH=src python src/train/train_tslib_model.py \
  --model DLinear \
  --feature-mart artifacts/feature_mart_track_b_per_site \
  --seq-len 168 \
  --pred-len 24 \
  --epochs 20 \
  --output-dir artifacts/training_runs/dlinear_track_b_merge_h24_seed42 \
  --merge-future-nwp-into-encoder-input \
  --future-nwp-variable-names tmp,reh,wsd,vec,sky,pcp
```

컨테이너에서는 [§5.0](#50-파일-관리-원칙-호스트-마운트-방식)에 따라 `/workspace/artifacts/...` 경로로 바꾼다.

### 트랙 A — 기준(현행)

| 항목 | 값 |
|------|-----|
| lookback | 과거 **168시간**(1주) |
| 예측 horizon | **24 / 48 / 72시간** |
| 입력 | 과거 구간의 관측·파생 feature (`normalized_power` 포함) |
| 산출물 | `project/artifacts/feature_mart_per_site/`, `project/artifacts/training_runs/` 등 기존 경로 |

### 트랙 B — 과거 168h + 미래 H시간 공변량

| 항목 | 값 |
|------|-----|
| lookback | 과거 **168시간** |
| 추가 입력 | t0 시점에 **이미 발표·관측 가능한** 미래 H시간의 공변량(ERA5 기반 `fcst_*` 등) |
| PV 타깃 | `normalized_power` **미래 구간**은 디코더 타깃으로만 사용하고, 인코더 입력에는 넣지 않거나 마스크 |
| 산출물 | `project/artifacts/feature_mart_track_b_per_site/` (빌드 리포트 `track_b_build_report.json` 등) |

**데이터를 새로 만들어야 하나?** 기존 per-site mart만으로는 `fcst_*`가 없는 경우가 많다 → **`enrich-track-b`** 로 기존 mart에 열을 붙이거나 동등 파이프라인으로 **Track B 전용 산출물**을 만든다. 원시 PV부터 전부 재수집할 필요는 없다.

### 트랙 C — 일 단위 lookback

| 항목 | 값 |
|------|-----|
| lookback | **365일**(일 단위 집계 시계열) |
| 예측 horizon | **1 / 2 / 3일** (일 에너지·평균출력 등 집계 극약을 실험 계약서에 명시) |
| 평가 | (옵션1) 일 예측을 시간해상도로 분해 후 기존 지표 / (옵션2) 일 단위 지표만 별도 정의 |
| 산출물 | 예: `project/artifacts/feature_mart_track_c_daily/` (구현 시 경로·스키마 확정) |

### 비교·리더보드 메타

리더보드·실험 로그에 `input_mode`(A/B/C), `resolution`(hourly/daily), `lookback`, `horizon_h` 등을 **필수 메타**로 남긴다.

### 권장 구현 순서

1. 트랙 A 고정·리더보드 스키마 확정  
2. 트랙 B: §3.7 파이프라인 검증 → 소수 모델 드라이런  
3. 트랙 C: 일별 집계·split 정의 후 베이스라인

---
## 전체 체크리스트

> **최종 갱신**: 2026-05-06 (PhotoRec 본문 + 상단·입력설계·§3.7 복원)

### 사전 준비

- [x] GPU 서버 접속 및 `nvidia-smi` 확인
- [x] Docker + NVIDIA Container Toolkit 정상 동작 확인
- [x] 사내 DB 접속 계정 확보
- [x] KMA API key 발급 (ASOS_API_KEY, 공공데이터포털)
- [x] Hugging Face 계정 + Llama gate 승인
- [ ] 운영 예측 시각 확정 (미정)

### 단계 1 완료 조건

- [x] `split_manifest.yaml` 작성 완료 — train: ~2024-12-31 / valid: ~2025-09-30 / test: ~2026-04-23
- [ ] `cpu_benchmark.yaml` 작성 완료 (단계 12 진입 시 작성 예정)

### 단계 2 완료 조건

- [x] 태양광 발전 데이터 추출 완료 (`pv_raw_hourly.parquet`) — 150 site, 2022-01-01 ~ 2026-04-23
- [x] ASOS 시간 관측 수집 완료 (`kma_obs_asos_hourly.parquet`) — 41 station, 2022-01-01 ~ 2026-04-22
- [x] site ↔ KMA grid/station 매핑 완료 (`site_to_kma_grid.csv`) — 36 ASOS station 매핑
- [x] ERA5 재분석 수집 완료 (`era5_nwp_input_raw.parquet`) — 4919 site, 2022-01-01 ~ 2026-04-19 (전략 B)
- [x] ERA5 바이어스 교정 완료 (`era5_nwp_bias_corrected.parquet`)
- [ ] AWS 시간 관측 수집 (`kma_obs_aws_hourly.parquet`) — APIHUB_KEY 미설정으로 보류
- [ ] 단기예보 누적 수집 (`kma_fcst_shortterm.parquet`) — 파이프라인 구현 완료, 데이터 누적 중
- [x] pvlib 태양 위치 계산 — Feature Mart 빌드 시 site별 자동 계산 (인라인 처리)

> **구현 위치**: `dataset/weather_collector/` (ASOS·ERA5·단기예보), `dataset/preprocessor/` (pvlib·Feature Mart)

### 단계 3 완료 조건

- [x] PV 결측값 처리 완료 (`is_imputed` flag 포함, 1h 이하 선형 보간)
- [x] 이상치 클리핑 완료 (pv_power_kw: 0 ~ capacity_kw)
- [x] ASOS + ERA5 기상 join 완료 (site_to_kma_grid.csv 기반 nearest station 매핑)
- [x] pvlib 태양 위치 feature 계산 완료 (`solar_elevation`, `solar_azimuth`, `clearsky_ghi`)
- [x] 일사량 처리: ASOS `icsr` (W/m²) — 센서 없는 8개 관측소는 NaN 유지 / `clearsky_ghi`와 분리
- [x] 파생 feature 계산 완료 (rolling 24/72/168h, lag 24/168h, 캘린더, 한국 공휴일)
- [x] z-score 정규화 완료 (train 구간 기준, `scaler_stats.json` 저장)
- [x] Feature Mart 저장 완료 — 100 site, train/valid/test 분리 (`feature_mart_per_site/{split}/{cid_seq}.parquet`)
  - **per-site adaptive split** 적용 (70/15/15 비율, 각 site 활성 기간 기준)
  - 전체 site 100개 포함 (global split 시 60% site가 train 데이터 없던 문제 해결)
  - split 경계: `project/artifacts/per_site_split_manifest.json` 저장
  - 품질 검사 PASS: train 낮결측 4.6%, valid 3.1%, test 1.0%
- [ ] Forecast leakage 없는 KMA 단기예보 join (단기예보 데이터 충분히 누적된 후 진행)

> **구현 위치**: `dataset/preprocessor/` — `pv_cleaner.py`, `weather_joiner.py`, `solar_features.py`, `derived_features.py`, `feature_mart_builder.py`

### 단계 4 완료 조건

- [x] `quality_report.json` 품질 기준 통과 (`feature_mart_per_site`, per-site split 기준)
  - train: 낮 시간대 결측률 4.6%, 0값 비율 9.5%, 평균 8,511h/site → PASS
  - valid: 낮 시간대 결측률 3.1%, 0값 비율 10.4%, 평균 1,824h/site → PASS
  - test:  낮 시간대 결측률 1.0%, 0값 비율 7.9%, 평균 1,824h/site → PASS
- [ ] `leakage_check_result.json` pass 확인 (단기예보 join 완료 후)
- [x] EDA notebook 시각화 검토 완료 (`project/notebooks/01_data_eda.ipynb`)
  - site별 발전 히트맵, 계절별 daily profile, 7일 연속 시계열 확인
  - clearsky_ghi vs normalized_power 계절별 산점도: 양의 상관 확인
  - 기상 feature 히스토그램, feature×site 결측률 히트맵, 태양위치 물리적 유효성 확인
  - split 경계 ±14일 시계열 비교, boxplot 분포 비교, KS 검정 (ta, clearsky_ghi 통과)
  - si 100% 결측 site 시각화 — icsr 센서 미보유 관측소 매핑 site 확인
  - **per-site split 반영 재실행 완료** (`feature_mart_per_site` 경로 업데이트)

### 단계 5 완료 조건

- [x] `.env` 파일 생성 및 `.gitignore` 등록 확인 (`.env`, `artifacts/models/`, `vendor/` 포함)
- [x] `docker run --rm --gpus all nvidia/cuda:12.1... nvidia-smi` GPU 패스스루 정상 확인
- [x] `bash scripts/setup_vendor.sh` 실행 → `vendor/TSLib/` 클론 확인
- [x] `pv-benchmark/unified:latest` 빌드 완료 (torch 2.4.1+cu121, transformers 5.5.4)
- [x] `pv-benchmark/unified:latest` GPU 동작 확인 (`torch.cuda.is_available() == True`)
- [x] `pv-benchmark/unified:latest` TSLib 임포트 확인 (vendor/TSLib 마운트 후 SegRNN, PatchTST) — 2026-04-27 확인
- [x] `pv-benchmark/unified:latest` peft + bitsandbytes 동작 확인
- [x] `pv-benchmark/time-llm:latest` 빌드 + GPT-2 캐시 + GPU 동작 확인
- [x] HF 토큰 설정 및 gate 승인 확인 (`docs/model_download_guide.md` 참조)
- [x] Llama 3.2 1B 모델 다운로드 완료 (`artifacts/models/`) — 2.4 GB
- [x] Gemma 4 E2B 모델 다운로드 완료 (`artifacts/models/`) — 9.6 GB
- [x] `scripts/verify_gpu.sh` 실행 — 2개 이미지 모두 `OK` 출력 확인
- [x] CPU-only 모드 검증 (`CUDA_VISIBLE_DEVICES=""` 시 GPU 비활성화 확인) — 2026-04-27 확인
- [x] `docker/docker-compose.yml` 동작 확인 (`docker compose run --rm unified python -c "print('OK')"`) — 2026-04-27 확인
- [x] `scripts/build_all.sh` 저장 및 실행 권한 확인
- [x] **더미 데이터 스모크 테스트 통과** (단계 5.10) — 전 모델 GPU/CPU forward pass 확인 (2026-04-22 수행)
  - [x] DLinear GPU/CPU forward PASS
  - [x] SegRNN GPU/CPU forward PASS
  - [x] PatchTST GPU/CPU forward PASS
  - [x] Time-LLM GPU/CPU forward PASS
  - [x] LLaMA 3.2 1B GPU/CPU forward PASS
  - [x] Gemma 4 E2B GPU/CPU forward PASS
  - [x] 결과 요약표(5.10.11) 작성 완료

### 단계 6~11 완료 조건 (각 모델)

- [ ] Seasonal Naive 평가 완료
- [ ] Persistence 평가 완료
- [ ] DLinear 학습/평가 완료 (3 seed)
- [ ] SegRNN 학습/평가 완료 (전체 실험 조합)
- [ ] PatchTST 학습/평가 완료 (전체 실험 조합)
- [ ] Time-LLM (GPT-2) 학습/평가 완료
- [ ] LLaMA 3.2 1B LoRA 학습/평가 완료
- [ ] Gemma 4 E2B LoRA 학습/평가 완료

### 단계 12 완료 조건

- [ ] 모든 모델 CPU cold start 측정 완료
- [ ] 모든 모델 CPU warm latency (p50/p95) 측정 완료
- [ ] 모든 모델 throughput (batch=1,8,16,32) 측정 완료
- [ ] 모든 모델 peak RAM 측정 완료
- [ ] LLM 계열 파싱 성공률/범위 위반율 측정 완료
- [ ] `cpu_benchmark_report.json` 저장 완료

### 단계 13 완료 조건

- [ ] 정확도 리더보드 (`leaderboard.md`) 완성
- [ ] CPU 벤치마크 리더보드 완성
- [ ] 종합 의사결정 점수 산출 완료
- [ ] 최종 모델 선정 결정 및 기록 완료

### 입력 설계 확장 벤치마크 완료 조건

- [ ] 트랙 B용 `feature_mart_track_b_per_site` 생성 및 `track_b_build_report.json` 검토
- [ ] 트랙 B 입력으로 **최소 1개 모델** 학습·평가 재현
- [ ] (선택) 트랙 C 일별 mart·평가 파이프라인 초안
- [ ] 리더보드에 `input_mode` / `lookback` / `resolution` 열 반영

---

*본 문서는 실험 진행 중 발견된 이슈에 따라 수정될 수 있다. 수정 시 버전 및 날짜를 상단에 기록한다.*
