# PV 데이터 수집기 (pv_collector)

사내 MySQL DB에서 태양광(삼상) 발전 데이터를 수집하여 parquet 파일로 저장하는 독립 프로그램입니다.

## 디렉토리 구조

```
pv_collector/
  .env.example    ← DB 접속 정보 템플릿
  .env            ← 실제 접속 정보 (직접 생성, git 제외)
  config.py       ← 설정 로더 (.env / 환경변수)
  db.py           ← DB 연결 관리
  queries.py      ← SQL 쿼리 정의
  collector.py    ← 수집 로직
  run.py          ← CLI 진입점
  requirements.txt
```

## 설치

```bash
cd dataset/pv_collector
pip install -r requirements.txt
```

## 설정

```bash
# 1. .env 파일 생성
cp .env.example .env

# 2. .env 파일에 DB 접속 정보 입력
#    DB_HOST=192.168.x.x
#    DB_PORT=3306
#    DB_USER=your_user
#    DB_PASSWORD=your_password
#    DB_NAME=your_database
```

## 사용법

```bash
# DB 연결 테스트
python run.py test

# DB 데이터 범위 요약 (수집 가능한 기간, CID 수 확인)
python run.py summary

# 발전소·CID 메타데이터 수집 → plant_meta.parquet
python run.py meta

# 시계열 전체 수집 (config 기본 기간) → pv_raw_hourly.parquet
python run.py hourly

# 날짜 범위 지정
python run.py hourly --start 2023-01-01 --end 2023-12-31

# 배치 단위 일수 조정 (기본 30일, 서버 부하에 따라 조정)
python run.py hourly --start 2022-01-01 --end 2023-12-31 --batch-days 7

# 특정 CID 만 수집
python run.py hourly --cid 101 102 103

# 증분 수집 (마지막 저장 timestamp 이후부터)
python run.py hourly --incremental

# 메타 + 시계열 일괄 수집
python run.py all --start 2023-01-01 --end 2023-12-31

# 출력 경로 지정
python run.py all --output /data/pv_snapshot

# 상세 로그 출력
python run.py -v hourly --start 2023-01-01 --end 2023-03-31
```

## 출력 파일

기본 저장 경로: `../../project/artifacts/dataset_snapshot/`

| 파일 | 설명 |
|------|------|
| `plant_meta.parquet` | 발전소·CID 메타 (위경도, 용량, 방위각 등) |
| `pv_raw_hourly.parquet` | 시간 집계 시계열 (삼상, kWh·kW 변환 완료) |

### `plant_meta.parquet` 주요 컬럼

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `plant_seq` | int | 발전소 식별자 |
| `cid_seq` | int | CID(인버터) 식별자 |
| `plant_nm` | str | 발전소 명 |
| `latitude` | float | 위도 (WGS84) |
| `longitude` | float | 경도 (WGS84) |
| `area_1` | str | 광역시도 |
| `area_2` | str | 시군구 |
| `module_capacity_kw` | float | 모듈 총 용량 (kW) |
| `ivt_capacity_kw` | float | 인버터 용량 (kW) |
| `azimuth` | str | 모듈 방위각 |
| `angle` | str | 모듈 경사각 |

### `pv_raw_hourly.parquet` 주요 컬럼

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `timestamp` | datetime | KST 기준 시각 (1시간 단위) |
| `plant_seq` | int | 발전소 식별자 |
| `cid_seq` | int | CID 식별자 |
| `pow_gen_kwh` | float | 시간 발전량 (kWh) |
| `cur_pow_kw` | float | 현재 출력 (kW) |
| `pv_pow_kw` | float | PV 입력 출력 (kW) |
| `pv_volt` | float | PV 전압 (V) |
| `pv_ampe` | float | PV 전류 (A) |
| `line_volt_rs/st/tr` | float | 계통 선간전압 (V, 삼상) |
| `line_amp_r/s/t` | float | 계통 전류 (A, 삼상) |
| `factor` | float | 역률 (%) |
| `frequency` | float | 주파수 (Hz) |

## 수집 대상 DB 테이블

```
tb_data_pv_hour  ← 시계열 (phase_type='tp' 삼상 필터)
    │
    ├─ plant_seq ─→ tb_power_plant  (위경도·지역)
    └─ cid_seq   ─→ tb_power_cid_pv (인버터 용량·방위각)
```

## 증분 수집 전략

```bash
# 최초 실행 (전체 이력)
python run.py all --start 2022-01-01

# 이후 주기적 실행 (신규 데이터만 추가)
python run.py hourly --incremental
```

증분 수집은 `pv_raw_hourly.parquet` 의 마지막 `timestamp` + 1시간부터 오늘까지 수집한 뒤  
기존 파일에 병합하고 중복을 제거합니다.
