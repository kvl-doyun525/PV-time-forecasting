"""
site(cid_seq) ↔ KMA 격자·ASOS 지점 매핑.

plant_meta.parquet 을 읽어 각 site에 대해:
  - 기상청 5km 격자 (fcst_nx, fcst_ny)
  - 가장 가까운 ASOS 지점 (asos_stn_id, dist_to_asos_km)
  - 가장 가까운 AWS 지점  (aws_stn_id,  dist_to_aws_km)
를 계산하여 site_to_kma_grid.csv 로 저장한다.

ASOS 지점 목록: 동 디렉토리의 asos_stations.csv (번들)
AWS  지점 목록: asos_stations.csv 와 동일하게 사용 가능 (AWS는 지점 수가 훨씬 많으나
               별도 목록이 없을 경우 ASOS 목록으로 대체)
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import pandas as pd

from config import ASOS_STATIONS_CSV, WEATHER_API_DIR, collect_config

log = logging.getLogger(__name__)

# weather_api.py 의 WeatherAPI.get_grid_coordinates() 재사용
sys.path.insert(0, str(WEATHER_API_DIR))
from weather_api import WeatherAPI  # noqa: E402

_weather_api = WeatherAPI("")  # 격자 변환만 사용 → 키 불필요


def latlon_to_kma_grid(latitude: float, longitude: float) -> tuple[int, int]:
    """위경도 → 기상청 5km 격자 (nx, ny)."""
    return _weather_api.get_grid_coordinates(latitude, longitude)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 간 Haversine 거리 (km)."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_asos_stations(path: Path | None = None) -> pd.DataFrame:
    """
    번들 asos_stations.csv 로드.

    컬럼: stnId, name, lat, lon, area_1
    """
    path = path or ASOS_STATIONS_CSV
    df = pd.read_csv(path, dtype={"stnId": str})
    required = {"stnId", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"asos_stations.csv 에 필수 컬럼 누락: {missing}")
    return df


def build_site_to_kma_mapping(
    plant_meta: pd.DataFrame,
    asos_stations: pd.DataFrame,
    aws_stations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    plant_meta 의 각 (plant_seq, cid_seq) 에 대해 매핑을 생성한다.

    Parameters
    ----------
    plant_meta    : plant_meta.parquet 로드 결과 (cid_seq, latitude, longitude 포함)
    asos_stations : ASOS 지점 DataFrame (stnId, lat, lon)
    aws_stations  : AWS 지점 DataFrame  (stnId, lat, lon). None이면 asos_stations 사용

    Returns
    -------
    DataFrame 컬럼:
        plant_seq, cid_seq,
        fcst_nx, fcst_ny,
        asos_stn_id, asos_stn_name, dist_to_asos_km,
        aws_stn_id,  dist_to_aws_km
    """
    if aws_stations is None:
        aws_stations = asos_stations

    # latitude/longitude 유효한 행만 처리
    meta = plant_meta.dropna(subset=["latitude", "longitude"]).copy()
    if meta.empty:
        log.warning("plant_meta 에 위경도 데이터가 없습니다.")
        return pd.DataFrame()

    rows: list[dict] = []
    for _, site in meta.iterrows():
        lat, lon = float(site["latitude"]), float(site["longitude"])

        try:
            nx, ny = latlon_to_kma_grid(lat, lon)
        except Exception as e:
            log.warning("격자 변환 실패 cid_seq=%s: %s", site["cid_seq"], e)
            nx, ny = None, None

        # 최근접 ASOS
        asos_dists = asos_stations.apply(
            lambda r: haversine_km(lat, lon, float(r["lat"]), float(r["lon"])), axis=1
        )
        idx_asos   = asos_dists.idxmin()
        near_asos  = asos_stations.loc[idx_asos]

        # 최근접 AWS
        aws_dists = aws_stations.apply(
            lambda r: haversine_km(lat, lon, float(r["lat"]), float(r["lon"])), axis=1
        )
        idx_aws  = aws_dists.idxmin()
        near_aws = aws_stations.loc[idx_aws]

        rows.append({
            "plant_seq":        site.get("plant_seq"),
            "cid_seq":          site["cid_seq"],
            "fcst_nx":          nx,
            "fcst_ny":          ny,
            "asos_stn_id":      near_asos["stnId"],
            "asos_stn_name":    near_asos.get("name", ""),
            "dist_to_asos_km":  round(asos_dists.min(), 2),
            "aws_stn_id":       near_aws["stnId"],
            "dist_to_aws_km":   round(aws_dists.min(), 2),
        })

    return pd.DataFrame(rows)


def run(
    plant_meta_path: Path | None = None,
    output_path: Path | None = None,
    asos_stations_path: Path | None = None,
) -> pd.DataFrame:
    """
    plant_meta.parquet → site_to_kma_grid.csv 생성 메인 함수.
    """
    out_dir = collect_config.output_dir
    plant_meta_path  = plant_meta_path  or (out_dir / "plant_meta.parquet")
    output_path      = output_path      or (out_dir / "site_to_kma_grid.csv")

    log.info("plant_meta 로드: %s", plant_meta_path)
    if not plant_meta_path.exists():
        raise FileNotFoundError(
            f"plant_meta.parquet 을 찾을 수 없습니다: {plant_meta_path}\n"
            "먼저 pv_collector 로 메타데이터를 수집하세요."
        )
    plant_meta = pd.read_parquet(plant_meta_path)

    log.info("ASOS 지점 목록 로드 ...")
    asos_stations = load_asos_stations(asos_stations_path)
    log.info("ASOS 지점 %d 개 로드", len(asos_stations))

    log.info("매핑 계산 중 (총 %d 개 site) ...", plant_meta["cid_seq"].nunique())
    df = build_site_to_kma_mapping(plant_meta, asos_stations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("저장 완료: %s  (%d 행)", output_path, len(df))

    return df
