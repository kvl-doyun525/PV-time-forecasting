"""
pvlib 기반 태양 위치 및 맑은하늘 일사량 계산 모듈.

별도 데이터 다운로드 없이 위경도 + 타임스탬프만으로 계산 가능하다.
pvlib Ineichen 모델을 사용하며, Linke turbidity 값은 패키지 내장 테이블을 참조한다.

출력 컬럼
---------
solar_elevation  : 태양 고도각 (°, 지평선 위 +)
solar_azimuth    : 태양 방위각 (°, 북=0, 동=90)
clearsky_ghi     : 맑은 하늘 수평면 전일사 (W/m²)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_solar_features(
    timestamps: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    altitude_m: float = 0.0,
    tz: str = "Asia/Seoul",
) -> pd.DataFrame:
    """
    단일 위치에 대한 태양 위치 + clear-sky 계산.

    Parameters
    ----------
    timestamps  : KST timezone-naive 또는 aware DatetimeIndex
    latitude    : 위도 (°N)
    longitude   : 경도 (°E)
    altitude_m  : 고도 (m), 기본 0
    tz          : 타임존 문자열

    Returns
    -------
    DataFrame with columns: solar_elevation, solar_azimuth, clearsky_ghi
    (index = timestamps)
    """
    try:
        import pvlib  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("`pip install pvlib` 를 실행하세요.") from e

    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=tz,
        altitude=altitude_m,
        name=f"lat{latitude:.2f}_lon{longitude:.2f}",
    )

    # timezone-aware 로 변환 (naive면 tz_localize)
    if timestamps.tz is None:
        ts = timestamps.tz_localize(tz)
    else:
        ts = timestamps.tz_convert(tz)

    solar_pos = location.get_solarposition(ts)
    clearsky  = location.get_clearsky(ts, model="ineichen")

    result = pd.DataFrame(index=timestamps)
    result["solar_elevation"] = solar_pos["apparent_elevation"].values
    result["solar_azimuth"]   = solar_pos["azimuth"].values
    result["clearsky_ghi"]    = clearsky["ghi"].values

    # 고도각 음수(지평선 아래) 시 clear-sky 0으로 보정
    result.loc[result["solar_elevation"] < 0, "clearsky_ghi"] = 0.0

    return result


def add_solar_to_site(
    site_df: pd.DataFrame,          # timestamp index
    latitude: float,
    longitude: float,
    altitude_m: float = 0.0,
) -> pd.DataFrame:
    """
    site_df 에 태양 위치 feature를 인플레이스로 추가하여 반환.
    timestamp index 가 KST(Asia/Seoul) 기준이어야 한다.
    """
    solar = compute_solar_features(
        site_df.index, latitude=latitude, longitude=longitude, altitude_m=altitude_m
    )
    site_df["solar_elevation"] = solar["solar_elevation"].values
    site_df["solar_azimuth"]   = solar["solar_azimuth"].values
    site_df["clearsky_ghi"]    = solar["clearsky_ghi"].values
    return site_df


def build_solar_lookup(
    meta: pd.DataFrame,   # plant_meta DataFrame: cid_seq, latitude, longitude
    timestamps: pd.DatetimeIndex,
) -> dict[int, pd.DataFrame]:
    """
    전체 site 태양 위치를 배치 계산하여 dict{cid_seq: DataFrame} 반환.

    동일 위경도 site는 캐싱하여 중복 연산을 방지한다.
    """
    try:
        import pvlib  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("`pip install pvlib` 를 실행하세요.") from e

    meta = meta[["cid_seq", "latitude", "longitude"]].drop_duplicates()

    # 중복 위경도 그룹화
    coord_groups: dict[tuple, list[int]] = {}
    for _, row in meta.iterrows():
        key = (round(row["latitude"], 4), round(row["longitude"], 4))
        coord_groups.setdefault(key, []).append(int(row["cid_seq"]))

    result: dict[int, pd.DataFrame] = {}
    from tqdm import tqdm  # noqa: PLC0415
    for (lat, lon), cids in tqdm(
        coord_groups.items(), desc="태양 위치 계산", unit="coord"
    ):
        solar = compute_solar_features(timestamps, lat, lon)
        for cid in cids:
            result[cid] = solar.copy()

    log.info("태양 위치 계산 완료: %d 좌표 → %d site", len(coord_groups), len(result))
    return result
