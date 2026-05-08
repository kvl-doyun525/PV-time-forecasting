"""
PV 원시 데이터 정제 모듈.

입력 : pv_raw_hourly.parquet  (timestamp, cid_seq, pv_pow_kw, ...)
출력 : cid_seq × timestamp 인덱스의 cleaned PV DataFrame
       컬럼: pv_power_kw, normalized_power, capacity_kw, is_imputed
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import PreprocessConfig, SNAPSHOT_DIR

log = logging.getLogger(__name__)


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _to_float(s: pd.Series) -> pd.Series:
    """object/string 타입 kW 컬럼 → float64. Decimal 표기(0E-9 등) 처리."""
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _hourly_resample(
    df: pd.DataFrame,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
) -> pd.DataFrame:
    """
    timestamp 기준 1시간 단위 리샘플링.
    같은 시간대에 중복 레코드가 있으면 평균을 취한다.

    time_start / time_end 를 지정하면 전역 시간 범위로 reindex한다.
    (미지정 시 site별 min/max 사용)
    """
    df = df.set_index("timestamp").sort_index()
    df = df.groupby(df.index).mean(numeric_only=True)

    if time_start and time_end:
        full_idx = pd.date_range(time_start, time_end, freq="1h")
    elif not df.empty:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    else:
        df.index.name = "timestamp"
        return df

    df = df.reindex(full_idx)
    df.index.name = "timestamp"
    return df


def _impute(df: pd.DataFrame, col: str, max_gap: int) -> pd.DataFrame:
    """짧은 연속 결측(≤ max_gap) 선형 보간. 초과 구간은 NaN 유지."""
    is_nan_before = df[col].isna()
    df[col] = df[col].interpolate(
        method="linear",
        limit=max_gap,
        limit_direction="both",
    )
    # 새로 채워진 셀을 is_imputed=True 로 표시
    df["is_imputed"] = df["is_imputed"] | (is_nan_before & df[col].notna())
    return df


def _clip(df: pd.DataFrame, col: str, cap_col: str = "capacity_kw") -> pd.DataFrame:
    """물리 범위 클리핑: 0 이하 → 0, 용량 초과 → 용량."""
    df[col] = df[col].clip(lower=0.0, upper=df[cap_col])
    return df


# ── 공개 API ──────────────────────────────────────────────────────────────────

def load_raw_pv(path: Optional[Path] = None) -> pd.DataFrame:
    """pv_raw_hourly.parquet 로드 + 기본 타입 정제."""
    path = path or (SNAPSHOT_DIR / "pv_raw_hourly.parquet")
    df = pd.read_parquet(path)

    df["timestamp"]  = pd.to_datetime(df["timestamp"])
    df["pv_pow_kw"]  = _to_float(df["pv_pow_kw"])
    df["cid_seq"]    = df["cid_seq"].astype(int)
    return df


def load_capacity(path: Optional[Path] = None) -> pd.Series:
    """
    plant_meta.parquet 에서 cid_seq → ivt_capacity_kw (kW) 매핑을 반환.
    ivt_capacity_kw 가 0이거나 NaN인 경우 모듈 용량 합계로 대체한다.
    """
    path = path or (SNAPSHOT_DIR / "plant_meta.parquet")
    meta = pd.read_parquet(path)

    cap = meta.set_index("cid_seq")["ivt_capacity_kw"].astype(float)

    # 이상치 보정: 0 또는 NaN → module_capacity_kw / 1000 (W→kW 변환)
    module_kw = (
        meta.set_index("cid_seq")["module_capacity_kw"].astype(float) / 1000.0
    )
    mask = cap.isna() | (cap <= 0)
    cap[mask] = module_kw[mask]
    return cap.rename("capacity_kw")


def clean_site(
    raw_df: pd.DataFrame,        # pv_raw_hourly 에서 해당 site 행
    capacity_kw: float,
    cfg: PreprocessConfig,
) -> Optional[pd.DataFrame]:
    """
    단일 site PV 데이터 정제.

    Returns
    -------
    DataFrame with columns:
        pv_power_kw   : 정제된 실측 발전량 (kW)
        normalized_power : pv_power_kw / capacity_kw
        capacity_kw   : 인버터 정격 용량 (kW)
        is_imputed    : 보간 채움 여부

    데이터 커버리지가 min_site_coverage 미만이면 None 반환.
    """
    if raw_df.empty:
        return None

    df = raw_df[["timestamp", "pv_pow_kw"]].copy()
    df["pv_pow_kw"] = _to_float(df["pv_pow_kw"])
    df = _hourly_resample(
        df,
        time_start=cfg.global_time_start,
        time_end=cfg.global_time_end,
    ).rename(columns={"pv_pow_kw": "pv_power_kw"})

    df["capacity_kw"] = float(capacity_kw) if capacity_kw and capacity_kw > 0 else np.nan
    df["is_imputed"]  = False

    # 물리 범위 클리핑
    df = _clip(df, "pv_power_kw", "capacity_kw")

    # 단기 결측 보간
    df = _impute(df, "pv_power_kw", cfg.max_interp_gap_hours)

    # 커버리지 체크
    total    = len(df)
    n_valid  = df["pv_power_kw"].notna().sum()
    coverage = n_valid / total if total > 0 else 0.0
    if coverage < cfg.min_site_coverage:
        log.warning(
            "cid_seq 데이터 커버리지 부족 (%.1f%%) — 건너뜀", coverage * 100
        )
        return None

    # 정규화
    cap = df["capacity_kw"].iloc[0]
    df["normalized_power"] = (
        df["pv_power_kw"] / cap if (cap and cap > 0) else np.nan
    )

    return df[["pv_power_kw", "normalized_power", "capacity_kw", "is_imputed"]]


def clean_all_sites(
    cfg: PreprocessConfig,
    pv_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> dict[int, pd.DataFrame]:
    """
    전체 site PV 데이터를 정제하여 dict{cid_seq: DataFrame} 반환.
    메모리 절약을 위해 groupby 후 site 단위로 순차 처리한다.
    """
    log.info("PV 원시 데이터 로드 중...")
    raw = load_raw_pv(pv_path)
    cap_map = load_capacity(meta_path)

    log.info("전체 site 수: %d", raw["cid_seq"].nunique())

    result: dict[int, pd.DataFrame] = {}
    groups = raw.groupby("cid_seq")

    from tqdm import tqdm  # noqa: PLC0415
    for cid, grp in tqdm(groups, desc="PV 정제", unit="site"):
        cap = cap_map.get(cid, np.nan)
        cleaned = clean_site(grp, cap, cfg)
        if cleaned is not None:
            result[cid] = cleaned

    log.info("정제 완료: %d / %d site", len(result), len(groups))
    return result
