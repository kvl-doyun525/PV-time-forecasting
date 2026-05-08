# 기상 데이터 수집기 (weather_collector)

기상청 공공데이터 API를 사용하여 PV 발전 site 에 매칭되는 기상 관측·예보 데이터를 수집합니다.  
`pv_collector` 로 수집한 `plant_meta.parquet` 을 입력으로 사용합니다.

## 디렉토리 구조

```
weather_collector/
  .env.example              ← API 키 템플릿
  .env                      ← 실제 키 (직접 생성, git 제외)
  config.py                 ← 설정 로더
  asos_stations.csv         ← ASOS 지점 목록 (번들, ~80개 주요 지점)
  kma_mapping.py            ← site → KMA 격자·ASOS 지점 매핑
  asos_collector.py         ← ASOS 시간 관측 이력
  aws_collector.py          ← AWS 방재기상관측 이력
  kma_forecast_collector.py ← 단기/초단기예보 수집
  run.py                    ← CLI 진입점
  requirements.txt
```

## 사전 조건

1. `pv_collector` 로 `plant_meta.parquet` 수집 완료
2. 공공데이터포털 API 키 발급

## 설치

```bash
cd dataset/weather_collector
pip install -r requirements.txt
```

## 설정

```bash
cp .env.example .env
```

`.env` 키별 용도:

| 키 | 발급처 | 필수 여부 | 용도 |
|----|--------|----------|------|
| `KMA_API_KEY` | data.go.kr | 단기예보 사용 시 필수 | 단기예보·초단기예보·일출몰 |
| `ASOS_API_KEY` | data.go.kr | ASOS 수집 시 필수 (KMA_API_KEY와 동일 키 가능) | ASOS 시간 관측 |
| `APIHUB_KEY` | apihub.kma.go.kr | **선택** — 미입력 시 AWS 수집 건너뜀 | AWS 방재기상관측 시간통계 |
| `CDS_API_KEY` | cds.climate.copernicus.eu | **선택** — ERA5 수집 시 필수 | ERA5 재분석 학습용 NWP 입력 |

> `APIHUB_KEY` 를 입력하지 않으면 `python run.py aws` 및 `python run.py all` 실행 시  
> AWS 단계를 자동으로 건너뛰고 나머지 수집은 정상 진행됩니다.  
> `CDS_API_KEY` 대신 `~/.cdsapirc` 파일을 직접 설정해도 됩니다.

## 실행 순서 (권장)

```bash
# 1. site → KMA 격자 / ASOS 지점 매핑 CSV 생성
python run.py mapping

# 2. ASOS 관측 이력 수집 (수 시간 소요 가능)
python run.py asos --start 2022-01-01 --end 2024-12-31

# 3. AWS 관측 이력 수집 (APIHUB_KEY 설정 시)
python run.py aws --start 2022-01-01 --end 2024-12-31

# 4. ERA5 재분석 수집 (학습용 NWP 입력 — 전략 B, 연도별 수 GB 소요)
python run.py era5 --years 2022 2023 2024

# 5-a. 단기예보 수집 — 현재 발표 시점 1회 (크론잡용, 3시간마다 반복)
python run.py forecast --type short

# 4-b. 단기예보 수집 — 과거 날짜 범위 이력 수집 (훈련 데이터용)
# 실제 API 지원 범위: 현재 시점 기준 약 48시간(2일) 이내
# --end를 오늘로 지정하면 미발표 issue_time은 자동 제외됨
python run.py forecast --type short --start 2026-04-23 --end 2026-04-24
```

또는 일괄 실행:
```bash
python run.py all --start 2022-01-01 --end 2024-12-31
```

## 세부 사용법

### ASOS 관측

```bash
# 특정 지점만 수집
python run.py asos --stations 108 112 156 --start 2023-01-01 --end 2023-12-31

# 중단 후 이어받기 (지점별 마지막 저장 날짜 이후부터 자동 재개)
python run.py asos --start 2022-01-01 --end 2024-12-31 --incremental

# 상세 로그 출력
python run.py -v asos --start 2023-01-01 --end 2023-03-31
```

**내결함성**
- 5xx / 네트워크 오류: 지수 백오프(2→4→8초) 최대 3회 자동 재시도
- 429 API 호출 제한: 60초 대기 후 자동 재시도
- **지점 완료마다 즉시 parquet 저장** — 중단 시 완료된 지점 데이터는 보존
- `--incremental`: 기존 parquet에서 **지점별 마지막 날짜**를 읽어 해당 지점은 그 이후부터 이어받기

### 단기/초단기 예보

#### 단기예보 vs 초단기예보 차이

| 항목 | 단기예보 (`--type short`) | 초단기예보 (`--type ultra`) |
|------|--------------------------|----------------------------|
| **예보 범위** | 발표 시점부터 **최대 3일(글피)** 후까지 | 발표 시점부터 **6시간** 후까지 |
| **시간 해상도** | **1시간 단위** 예보 | **1시간 단위** 예보 |
| **발표 횟수** | 1일 **8회** (02·05·08·11·14·17·20·23시) | 1일 **24회** (매 정시, 30분 후 생성) |
| **API 제공 시작** | 발표시각 + 10분 (예: 02:10) | 발표시각 + 45분 (예: 00:45) |
| **주요 변수** | 기온·습도·풍향풍속·하늘상태·강수형태·강수확률·강수량·적설 | 기온·습도·풍향풍속·하늘상태·강수형태·강수량 (적설 없음) |
| **PV 예측 활용** | **훈련/추론 입력** — 발표시각 기준 리드타임 레이블링 후 사용 | 단기예보 보완 (최근 6시간 고해상도) |

> **리드타임(lead time)이란?** `target_time - issue_time` 으로 계산한 예보 발표 후 경과 시간.  
> 예: `issue_time=2024-01-01 08:00`, `target_time=2024-01-01 14:00` → lead_time=6h  
> 모델 훈련 시 leakage 방지를 위해 반드시 `issue_time` 기준으로 분리해야 한다.

```bash
# 현재 발표 시점 수집 (크론잡으로 단기는 3시간, 초단기는 1시간마다 실행 권장)
python run.py forecast --type short   # 단기예보 (3일치)
python run.py forecast --type ultra   # 초단기예보 (6시간치)

# 과거 날짜 범위 이력 수집 (기상청 API는 최근 ~3일 이내만 지원)
python run.py forecast --type short --start 2026-04-18 --end 2026-04-21

# 중단 후 이어받기
#   현재 시점: 이번 issue_time에 이미 수집된 cid_seq 건너뜀
#   날짜 범위: 모든 cid_seq가 완료된 issue_time 건너뜀
python run.py forecast --type short --start 2026-04-18 --end 2026-04-21 --incremental
```

**내결함성**
- 5xx / 429 오류: ASOS와 동일한 재시도 로직 적용
- **20개 site마다 즉시 parquet 저장** — 중단 시 손실 최소화
- `--incremental`: 완료된 `issue_time` 또는 `cid_seq` 단위로 건너뜀

> ⚠️ **과거 예보 이력 API 한계**: 기상청 단기예보 API는 현재 시점 기준 약 **48시간(2일) 이전** 발표 자료를 지원하지 않습니다.  
> 문서상 "약 3일"이라고 되어 있으나 실측 기준 **50시간 이상 이전** issue_time은 `NO_DATA`를 반환합니다.  
> `v1.1`부터 50시간 초과 issue_time은 자동으로 건너뜁니다.

---

#### 과거 단기예보 이력이 필요한 경우 (학습 데이터 구축)

학습용으로 과거 수년치 예보 이력이 필요하다면 아래 3가지 방법을 고려한다.

| 방법 | 데이터 종류 | 기간 | 자동화 | 비고 |
|------|------------|------|:------:|------|
| **① ERA5 재분석 데이터** (권장) | NWP 재분석 (관측 동화) | 1940~현재 | ✓ Python API | 기상청 예보가 아닌 재분석이지만 학습용으로 표준적으로 사용 |
| **② 기상자료개방포털 파일셋** | 실제 동네예보 발표 이력 | 2012~현재 | △ 수동 신청 | `data.kma.go.kr → 기상예보 → 동네예보` CSV 파일셋, 대용량 신청 필요 |
| **③ 지금부터 실시간 수집** | 실제 단기예보 | 수집 시작일~ | ✓ 크론잡 | 과거는 불가, 향후 학습 데이터 누적용 |

##### ① ERA5 재분석 데이터 (가장 현실적인 대안)

[ECMWF Copernicus CDS](https://cds.climate.copernicus.eu/)에서 무료로 제공하는 전지구 시간별 재분석 데이터.  
기상청 단기예보와 동일한 물리량을 제공하며 PV 발전량 예측 연구에서 가장 널리 사용된다.

| 항목 | ERA5 사양 |
|------|-----------|
| 시간 해상도 | 1시간 |
| 공간 해상도 | 0.25° × 0.25° (~28km) |
| 제공 기간 | 1940년 ~ 현재 (약 5일 지연) |
| PV 관련 변수 | `ssrd`(일사량), `t2m`(기온), `u10`/`v10`(풍향풍속), `tp`(강수), `tcc`(전운량) |
| 비용 | 무료 (CDS 계정 필요) |

```python
# pip install cdsapi
import cdsapi

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "surface_solar_radiation_downwards",  # ssrd — 일사량
            "2m_temperature",                      # t2m — 기온
            "10m_u_component_of_wind",             # u10
            "10m_v_component_of_wind",             # v10
            "total_precipitation",                 # tp
            "total_cloud_cover",                   # tcc
        ],
        "year":  [str(y) for y in range(2022, 2025)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day":   [f"{d:02d}" for d in range(1, 32)],
        "time":  [f"{h:02d}:00" for h in range(24)],
        "area":  [38.5, 126.0, 34.0, 130.0],   # 한국 영역 (N, W, S, E)
        "format": "netcdf",
    },
    "era5_korea_2022_2024.nc",
)
```

> **⚠️ ERA5는 실시간 추론에 사용 불가** — ERA5는 현재 시점 기준 약 **5일 지연**이 있는 재분석(reanalysis) 데이터다.  
> 미래를 예측하는 값이 아니므로 실시간 서비스의 추론 입력으로 사용할 수 없다.  
> 학습 전용 데이터로만 사용하고, 실시간 추론에는 아래 [실시간 추론용 NWP 데이터](#실시간-추론용-nwp-데이터) 를 참고한다.

> **훈련-추론 일관성 주의**: 학습은 ERA5로, 실시간 추론은 기상청 단기예보로 진행하면  
> 변수 단위·스케일이 다르므로 **정규화 레이어 또는 매핑 테이블**이 필요하다.
>
> | ERA5 변수 | 기상청 단기예보 변수 | 단위 변환 |
> |-----------|--------------------|---------:|
> | `ssrd` (J/m²) | 없음 (예보에는 일사 없음) | ÷ 3600 → W/m² |
> | `t2m` (K) | `tmp` (℃) | − 273.15 |
> | `u10`/`v10` (m/s) | `wsd` (m/s), `vec` (°) | 벡터 → 속도/방향 변환 |
> | `tp` (m) | `pcp` (mm) | × 1000 |
> | `tcc` (0~1) | `sky` (1/3/4 코드) | 임계값 기반 매핑 |

##### ② 기상자료개방포털 동네예보 파일셋

[data.kma.go.kr](https://data.kma.go.kr) → 데이터 → 기상예보 → 동네예보 메뉴에서  
실제 발표된 동네예보 이력을 CSV 파일셋으로 신청·다운로드할 수 있다.  
ERA5보다 기상청 예보와 일관성이 높지만 **대용량 신청 → 처리 대기** 과정이 필요하다.

**신청 시 지정해야 할 격자 범위** (`site_to_kma_grid.csv` 기준)

`site_to_kma_grid.csv` 에는 총 **345개 고유 격자(nx, ny)** 가 있으며, 전국에 분산되어 있다.  
동네예보 파일셋은 격자 단위 또는 시도 단위로 신청하며, 아래 지역을 커버해야 한다.

| 지역 | PV site 수 | 격자 수 | nx 범위 | ny 범위 | 위도 범위 | 경도 범위 |
|------|----------:|:------:|:-------:|:-------:|:---------:|:---------:|
| 충청 (충남·충북·대전·세종) | 1,346 | 55 | 56~69 | 104~118 | 36.5°~37.2° | 126.7°~127.5° |
| 강원 | 1,252 | 115 | 61~94 | 119~129 | 37.2°~37.7° | 127.0°~129.0° |
| 전북 | 510 | 29 | 55~68 | 89~103 | 35.8°~36.5° | 126.7°~127.4° |
| 경기북부/강원북부 | 338 | 35 | 55~92 | 130~136 | 37.7°~38.0° | 126.7°~128.9° |
| 서울/경기 | 319 | 37 | 54~60 | 119~129 | 37.2°~37.7° | 126.6°~127.0° |
| 제주 | 270 | 20 | 46~51 | 33~38 | 33.3°~33.5° | 126.2°~126.4° |
| 경남/울산/부산 | 250 | 10 | 89~104 | 83~94 | 35.5°~36.0° | 128.6°~129.4° |
| 경북남부/대구 | 241 | 30 | 78~87 | 88~98 | 35.8°~36.2° | 128.0°~128.5° |
| 경북북부 | 157 | 11 | 69~93 | 110~119 | 36.8°~37.2° | 127.5°~128.9° |
| 경남남부 | 76 | 3 | 97~98 | 75~76 | 35.1°~35.2° | 129.0°~129.1° |
| **합계** | **4,759** | **345** | **46~104** | **33~136** | **33.3°~38.0°** | **126.2°~129.4°** |

> **신청 권장**: 격자 345개가 전국에 흩어져 있으므로 **전국(한반도 전체)** 데이터로 신청하는 것이 효율적이다.  
> 격자를 개별 지정하면 너무 많아 신청이 복잡해진다. 전국 데이터를 받은 뒤  
> `site_to_kma_grid.csv` 의 `(fcst_nx, fcst_ny)` 컬럼으로 필터링하면 된다.

**필터링에 필요한 ASOS 지점 목록** (연관 36개)

| 지역 | 지점번호 | 지점명 |
|------|---------|--------|
| 서울/경기/인천 | 98, 99, 108, 112, 119, 202, 203 | 동두천, 파주, 서울, 인천, 수원, 양평, 이천 |
| 강원 | 93, 95, 100, 101, 104, 106, 114, 121, 212, 216, 217 | 북춘천, 철원, 대관령, 춘천, 북강릉, 동해, 원주, 영월, 홍천, 태백, 정선군 |
| 충청 | 131, 133, 135, 177, 232, 235, 236 | 청주, 대전, 추풍령, 홍성, 천안, 보령, 부여 |
| 전라 | 146 | 전주 |
| 경북 | 138, 273, 279, 281 | 포항, 문경, 구미, 영천 |
| 경남/대구/부산/울산 | 143, 152, 159, 284 | 대구, 울산, 부산, 거창 |
| 제주 | 184, 185 | 제주, 고산 |

##### ③ 지금부터 실시간 수집 (가장 단순)

크론잡으로 3시간마다 `python run.py forecast --type short`을 실행하면  
수집 시작일 이후 모든 발표 이력이 `kma_fcst_shortterm.parquet`에 누적된다.  
초기에는 과거 데이터가 없으므로 ①②와 병행하거나, 충분한 누적 후 학습에 활용한다.

```bash
# crontab -e 예시 (매 3시간 단기예보 수집)
10 2,5,8,11,14,17,20,23 * * * cd /path/to/weather_collector && python run.py forecast --type short --incremental
```

### AWS 관측

```bash
# APIHUB_KEY 설정 후 수집
python run.py aws --start 2022-01-01 --end 2024-12-31

# 중단 후 이어받기
python run.py aws --start 2022-01-01 --end 2024-12-31 --incremental
```

> APIHUB_KEY 미설정 시 `aws` 명령 및 `all` 일괄 실행에서 AWS 단계가 자동으로 건너뜁니다.

### ERA5 재분석 (학습용 NWP 입력 — 전략 B)

> **전략 B 의사결정**: 훈련에는 ERA5(바이어스 교정, 일사량 제외), 추론에는 기상청 단기예보를 사용한다.  
> 일사량(`ssrd`)은 ERA5 고편향(+23~40%) 및 단기예보 변수 부재로 **의도적으로 제외**한다.

**사전 준비**

```bash
pip install cdsapi xarray netCDF4 scipy

# CDS 계정 생성: https://cds.climate.copernicus.eu
# 방법 1: .env 에 키 입력
echo "CDS_API_KEY=<UID>:<API-KEY>" >> .env

# 방법 2: ~/.cdsapirc 파일 생성 (cdsapi 공식 방법)
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api
key: <UID>:<API-KEY>
EOF
```

**수집**

```bash
# 2022~2024년 수집 + 바이어스 교정 (권장)
python run.py era5 --years 2022 2023 2024

# 바이어스 교정 없이 원본만 저장
python run.py era5 --years 2022 2023 2024 --no-bias-correct

# 특정 연도만
python run.py era5 --years 2024
```

**수집 변수 (일사량 제외)**

| ERA5 변수 | 출력 컬럼 | 변환 후 컬럼 | 추론 대응 |
|-----------|----------|------------|---------|
| `2m_temperature` | `t2m` (K) | `t2m_c` (℃) | `tmp` |
| `10m_u/v_wind` | `u10`, `v10` (m/s) | `wsd` (m/s), `vec` (°) | `wsd`, `vec` |
| `2m_dewpoint_temperature` | `d2m` (K) | `reh` (%) | `reh` |
| `total_precipitation` | `tp` (m) | `tp_mm` (mm) | `pcp` |
| `total_cloud_cover` | `tcc` (0~1) | — | `sky` 코드 매핑 |

**출력 파일**

| 파일 | 설명 |
|------|------|
| `era5_nwp_input_raw.parquet` | CDS 수집 원본 + 단위 변환 |
| `era5_nwp_bias_corrected.parquet` | ASOS 기준 quantile mapping 교정본 (학습 입력) |

**테스트 시 비교**

학습은 `era5_nwp_bias_corrected.parquet`으로, 테스트 시에는 ERA5와 기상청 단기예보 두 가지 입력으로 각각 추론하여 성능을 비교한다.

## 출력 파일

기본 저장 경로: `../../project/artifacts/dataset_snapshot/`

| 파일 | 설명 |
|------|------|
| `site_to_kma_grid.csv` | cid_seq ↔ KMA 격자(nx,ny) + ASOS/AWS 지점 매핑 |
| `kma_obs_asos_hourly.parquet` | ASOS 시간 관측 이력 |
| `kma_obs_aws_hourly.parquet` | AWS 방재기상관측 이력 |
| `kma_fcst_shortterm.parquet` | 단기예보 누적 (issue_time + target_time) |
| `kma_fcst_ultrashort.parquet` | 초단기예보 누적 |

### `site_to_kma_grid.csv` 컬럼

| 컬럼 | 설명 |
|------|------|
| `cid_seq` | PV site 식별자 (plant_meta 의 cid_seq) |
| `plant_seq` | 발전소 식별자 |
| `fcst_nx` | 기상청 단기예보 격자 X |
| `fcst_ny` | 기상청 단기예보 격자 Y |
| `asos_stn_id` | 가장 가까운 ASOS 지점 번호 |
| `asos_stn_name` | ASOS 지점명 |
| `dist_to_asos_km` | site → ASOS 지점 거리 (km) |
| `aws_stn_id` | 가장 가까운 AWS 지점 번호 |
| `dist_to_aws_km` | site → AWS 지점 거리 (km) |

### `kma_obs_asos_hourly.parquet` 주요 컬럼

| 컬럼 | 설명 |
|------|------|
| `tm` | 관측 시각 (KST) |
| `stnId` | ASOS 지점 번호 |
| `ta` | 기온 (℃) |
| `rn` | 강수량 (mm) |
| `ws` | 풍속 (m/s) |
| `wd` | 풍향 (°) |
| `hm` | 습도 (%) |
| `icsr` | 일사량 (MJ/m²) ⚠️ 결측률 높음 — API 컬럼명은 `icsr` (`si` 아님) |
| `ss` | 일조 (hr) ⚠️ 결측률 높음 |
| `dc10Tca` | 전운량 (0~10) |

### `kma_fcst_shortterm.parquet` / `kma_fcst_ultrashort.parquet` 주요 컬럼

두 파일의 스키마는 동일하며, `pop`(강수확률)·`sno`(적설)는 단기예보에만 존재한다.

| 컬럼 | 설명 | 단기 | 초단기 |
|------|------|:----:|:------:|
| `cid_seq` | PV site 식별자 | ✓ | ✓ |
| `issue_time` | 예보 발표 시각 — leakage 방지 기준 | ✓ | ✓ |
| `target_time` | 예보 대상 시각 | ✓ | ✓ |
| `tmp` | 기온 (℃) | ✓ | ✓ |
| `reh` | 습도 (%) | ✓ | ✓ |
| `wsd` | 풍속 (m/s) | ✓ | ✓ |
| `vec` | 풍향 (°) | ✓ | ✓ |
| `sky` | 하늘 상태 코드 (1=맑음, 3=구름많음, 4=흐림) | ✓ | ✓ |
| `pty` | 강수 형태 코드 (0=없음, 1=비, 2=비/눈, 3=눈, 4=소나기) | ✓ | ✓ |
| `pcp` | 강수량 (mm) | ✓ | ✓ |
| `pop` | 강수 확률 (%) | ✓ | — |
| `sno` | 적설 (cm) | ✓ | — |

## 데이터 흐름

```
plant_meta.parquet  →  kma_mapping.py  →  site_to_kma_grid.csv
                                                    │
              ┌─────────────────────────────────────┤
              │                                     │
    asos_stn_id  →  asos_collector.py           fcst_nx/ny  →  kma_forecast_collector.py
    aws_stn_id   →  aws_collector.py
```

## API 키 발급

| API | 발급처 | 서비스명 |
|-----|--------|----------|
| 단기예보 | [data.go.kr](https://www.data.go.kr/data/15084084/openapi.do) | 기상청_단기예보 ((구)동네예보) 조회서비스 |
| ASOS 관측 | [data.go.kr](https://www.data.go.kr/data/15057210/openapi.do) | 기상청_지상(종관, ASOS) 시간자료 조회서비스 |
| AWS 관측 | [apihub.kma.go.kr](https://apihub.kma.go.kr) | 기상청 API허브 → 지상관측 → 방재기상관측(AWS) → 시간통계 |

> ⚠️ **AWS 관측 API 제약사항** (공공데이터포털 15057084):  
> - 1분 자료만 제공, 최근 2일 이내만 조회 가능 → **과거 이력 수집 불가**  
> - 공공기관 전용 서비스 (일반 기업/개인 사용 불가)  
>
> 따라서 `aws_collector.py` 는 **기상청 API허브** (`apihub.kma.go.kr`) 의 `awsh.php` 를 사용한다.  
> API허브 인증키는 공공데이터포털 키와 **별도 발급** 필요.
>
> **과거 이력 대안**: [기상자료개방포털](https://data.kma.go.kr/data/grnd/selectAwsRltmList.do?pgmNo=56) 에서 CSV 파일셋 직접 다운로드 가능 (자동화 불필요 시 권장)

## 실시간 추론용 NWP 데이터

ERA5는 **학습 전용**이다. 실시간 PV 발전량 예측(서비스 추론) 시에는 현재 시점 이후의 미래 날씨를 예측하는 **NWP(수치예보모델) 데이터**가 필요하다.

### ERA5 vs 실시간 NWP 비교

| 항목 | ERA5 (재분석) | 실시간 NWP 예보 |
|------|:------------:|:--------------:|
| **사용 목적** | 학습 데이터 | 추론 입력 |
| **데이터 성격** | 과거 관측 동화 ("정답에 가까운 과거") | 미래 예측 |
| **지연** | 현재 시점 기준 **~5일 과거** | 발표 후 수 시간 내 제공 |
| **미래 예측 가능** | ❌ 불가 | ✓ 최대 10~16일 |

### 실시간 추론에 사용 가능한 무료 NWP 소스

| 소스 | 해상도 | 예보 기간 | 업데이트 | 일사량 | Python API |
|------|:------:|:--------:|:--------:|:------:|-----------|
| **ECMWF Open Data (IFS)** | 0.25° (~25km) | 15일 | 6시간마다 | ✓ `ssrd` | `ecmwf-opendata` |
| **NOAA GFS** | 0.25° (~28km) | 16일 | 6시간마다 | ✓ `DSWRF` | `getgfs` |
| **Open-Meteo** | 9km (IFS 원본) | 15일 | 6시간마다 | ✓ `shortwave_radiation` | HTTP REST (무료) |
| **기상청 단기예보** | ~5km (격자) | 3일 | 3시간마다 | ❌ 없음 | 현재 구현됨 |

> **기상청 단기예보에는 일사량 변수가 없다.** PV 발전량 예측에 일사량이 핵심 입력인 경우  
> ECMWF Open Data 또는 GFS를 추론 입력으로 사용해야 한다.

### 권장 훈련-추론 전략

```
전략 A — 일관성 최우선 (권장)
  훈련: ECMWF Open Data 과거 아카이브 (또는 ERA5)
  추론: ECMWF Open Data 실시간 예보
  → 동일 NWP 모델 출력 → 변수/스케일 불일치 최소화

전략 B — 기상청 단기예보 활용
  훈련: ERA5 + ASOS icsr (실측 일사량으로 ssrd 대체)
  추론: 기상청 단기예보 + ASOS 최근 실측 icsr
  → 일사량을 예보 대신 최근 관측값으로 대체
  → 일사량 예측 불확실성이 크면 오히려 더 나을 수 있음

전략 C — 멀티소스 앙상블
  훈련: ERA5
  추론: ECMWF IFS + 기상청 단기예보 동시 입력 → 앙상블 평균
```

### ECMWF Open Data 실시간 예보 수집 예시

```python
# pip install ecmwf-opendata
from ecmwf.opendata import Client

client = Client()  # 계정 불필요, 완전 무료

# 최신 예보 다운로드 (발표 후 ~7~9시간 후 제공)
client.retrieve(
    date=0,          # 0=최신, -1=이전 런
    time=0,          # 00UTC 런
    step=[i for i in range(0, 91, 1)],   # 0~90h, 1시간 단위
    param=["ssrd", "2t", "10u", "10v", "tp", "tcc"],
    target="ecmwf_latest.grib2",
)
```

### NOAA GFS 실시간 예보 수집 예시

```python
# pip install getgfs
import getgfs

f = getgfs.Forecast("0p25_1hr")   # 0.25°, 1시간 단위
res = f.get(
    ["dswrfsfc",   # 일사량 W/m²
     "tmp2m",      # 2m 기온
     "ugrd10m", "vgrd10m",   # 10m 풍향풍속
     "tprate"],    # 강수율
    "now",
    lat=37.5,   # 수집할 위경도
    lon=127.0,
)
```

---

## API 호출 제한 대응

- 공공데이터포털 API 일일 호출 제한: 서비스별 1,000 ~ 10,000건
- 장기 이력 수집 시 **여러 날에 걸쳐 분할 수집** 권장
- `--incremental` 옵션으로 이미 수집된 구간 재수집 방지
- 429 수신 시 60초 자동 대기 후 재시도 (코드 내장)
- 수집 도중 중단 → 재실행 시 `--incremental` 으로 이어받기
