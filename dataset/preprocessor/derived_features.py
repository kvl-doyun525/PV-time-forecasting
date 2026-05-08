"""
파생 feature 계산 모듈.

1. 롤링 통계  : pv_roll_mean/std_{24,72,168}h
2. 래그 feature: pv_lag_{24,168}h
3. 캘린더     : hour, dayofweek, month, dayofyear, is_holiday
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import ROLLING_WINDOWS, LAG_HOURS

log = logging.getLogger(__name__)

# 한국 공휴일 캐시 (연도별)
_KR_HOLIDAYS: Optional[set] = None


def _get_kr_holidays(years: list[int]) -> set:
    """holidays 라이브러리로 한국 공휴일 날짜 집합 반환."""
    global _KR_HOLIDAYS
    if _KR_HOLIDAYS is not None:
        return _KR_HOLIDAYS
    try:
        import holidays as hd  # noqa: PLC0415
        kr = hd.KR(years=years)
        _KR_HOLIDAYS = set(kr.keys())
    except Exception as e:
        log.warning("한국 공휴일 로드 실패 (%s) — is_holiday=0 으로 처리", e)
        _KR_HOLIDAYS = set()
    return _KR_HOLIDAYS


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "pv_power_kw",
    windows: list[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Rolling mean/std 및 lag feature 추가.

    NaN이 있는 구간은 min_periods=80% 로 처리하여 결측이 과도하게 전파되지 않도록 한다.
    """
    series = df[target_col]

    for w in windows:
        min_p = max(1, int(w * 0.8))
        df[f"pv_roll_mean_{w}h"] = series.rolling(w, min_periods=min_p).mean()
        df[f"pv_roll_std_{w}h"]  = series.rolling(w, min_periods=min_p).std()

    for h in LAG_HOURS:
        df[f"pv_lag_{h}h"] = series.shift(h)

    return df


def add_calendar_features(
    df: pd.DataFrame,
    holiday_years: list[int],
    ts_col: Optional[str] = None,  # None이면 index 사용
) -> pd.DataFrame:
    """
    캘린더 feature 추가.

    Columns added:
        hour, dayofweek, month, dayofyear, is_holiday
    """
    ts = df.index if ts_col is None else pd.to_datetime(df[ts_col])

    df["hour"]       = ts.hour.astype("float32")
    df["dayofweek"]  = ts.dayofweek.astype("float32")
    df["month"]      = ts.month.astype("float32")
    df["dayofyear"]  = ts.dayofyear.astype("float32")

    holidays = _get_kr_holidays(holiday_years)
    df["is_holiday"] = ts.normalize().map(lambda d: float(d.date() in holidays))

    return df


def add_all_derived(
    df: pd.DataFrame,
    holiday_years: list[int],
    target_col: str = "pv_power_kw",
) -> pd.DataFrame:
    """rolling + calendar + lag 한 번에 추가."""
    df = add_rolling_features(df, target_col=target_col)
    df = add_calendar_features(df, holiday_years=holiday_years)
    return df
