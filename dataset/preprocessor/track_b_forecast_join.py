"""
Track B: wide fan `fcst_{var}_{h:03d}` 조인 (issue_target / ERA5 hourly valid).

복구: recup_dir.7/f567300728.txt, f567522584.txt, f567522456.txt.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "timestamp" in df.columns:
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        return out.set_index("timestamp").sort_index()
    raise ValueError("DataFrame에 DatetimeIndex 또는 timestamp 컬럼이 필요합니다.")


def _normalize_fcst_frame(fc_df: pd.DataFrame, site_col: str) -> pd.DataFrame:
    out = fc_df.copy()
    out[site_col] = out[site_col].astype(int)
    for c in ("issue_time", "target_time"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c])
    return out


def enrich_frame_with_forecast_covariates(
    df: pd.DataFrame,
    fc_df: pd.DataFrame,
    *,
    cid_seq: int,
    horizon_max: int,
    value_cols: Sequence[str],
    site_col: str = "cid_seq",
) -> pd.DataFrame:
    work = ensure_datetime_index(df)
    idx = work.index
    H = int(horizon_max)
    if H < 1:
        raise ValueError("horizon_max는 1 이상이어야 합니다.")

    fc_site = _normalize_fcst_frame(fc_df, site_col)
    fc_site = fc_site[fc_site[site_col] == cid_seq].copy()

    if fc_site.empty:
        log.warning("cid_seq=%s 에 대한 예보 행이 없습니다. fcst_* 컬럼은 NaN으로 채웁니다.", cid_seq)
        out = work.copy()
        for col in value_cols:
            for h in range(1, H + 1):
                out[f"fcst_{col}_{h:03d}"] = np.nan
        return out

    need_cols = [site_col, "issue_time", "target_time", *list(value_cols)]
    missing = [c for c in need_cols if c not in fc_site.columns]
    if missing:
        raise KeyError(f"예보 테이블에 컬럼이 없습니다: {missing}")

    multi = pd.MultiIndex.from_product([idx, range(1, H + 1)], names=["t0", "h"])
    long_df = multi.to_frame(index=False)
    long_df["target_time"] = long_df["t0"] + pd.to_timedelta(long_df["h"], unit="h")

    m = long_df.merge(fc_site[need_cols], on="target_time", how="left")
    m = m[m["issue_time"].notna() & (m["issue_time"] <= m["t0"])]
    if not m.empty:
        m = m.sort_values(["t0", "h", "issue_time"]).groupby(["t0", "h"], as_index=False).last()

    wide_parts: list[pd.DataFrame] = []
    for col in value_cols:
        if m.empty:
            piv = pd.DataFrame(index=idx, columns=range(1, H + 1), dtype="float64")
        else:
            piv = m.pivot(index="t0", columns="h", values=col)
            piv = piv.reindex(index=idx)
        piv = piv.reindex(columns=list(range(1, H + 1)))
        piv.columns = [f"fcst_{col}_{int(c):03d}" for c in piv.columns]
        wide_parts.append(piv)

    extra = pd.concat(wide_parts, axis=1)
    out = work.join(extra, how="left")
    return out


def enrich_frame_with_hourly_era5_valid_covariates(
    df: pd.DataFrame,
    era_site: pd.DataFrame,
    *,
    horizon_max: int,
    value_cols: Sequence[str],
    time_col: str = "timestamp",
) -> pd.DataFrame:
    work = ensure_datetime_index(df)
    idx = work.index
    H = int(horizon_max)
    if H < 1:
        raise ValueError("horizon_max는 1 이상이어야 합니다.")

    out = work.copy()
    if era_site is None or era_site.empty:
        log.warning("ERA5 site 시계열이 비어 있습니다. fcst_* 는 NaN으로 채웁니다.")
        for col in value_cols:
            for h in range(1, H + 1):
                out[f"fcst_{col}_{h:03d}"] = np.nan
        return out

    if time_col not in era_site.columns:
        raise KeyError(f"era_site에 시간 컬럼 '{time_col}' 이 필요합니다: {list(era_site.columns)}")

    era = era_site.copy()
    era[time_col] = pd.to_datetime(era[time_col])
    era = era.set_index(time_col).sort_index()
    missing = [c for c in value_cols if c not in era.columns]
    if missing:
        raise KeyError(f"ERA5 시계열에 value 컬럼이 없습니다: {missing}")

    for col in value_cols:
        ser = era[col]
        for h in range(1, H + 1):
            tgt = idx + pd.Timedelta(hours=h)
            out[f"fcst_{col}_{h:03d}"] = ser.reindex(tgt).to_numpy()

    return out


def forecast_covariate_column_names(value_cols: Iterable[str], horizon_max: int) -> list[str]:
    names: list[str] = []
    for col in value_cols:
        for h in range(1, int(horizon_max) + 1):
            names.append(f"fcst_{col}_{h:03d}")
    return names


def assert_forecast_join_no_leakage(
    sample_t0: pd.Timestamp,
    joined_row: pd.Series,
    fc_df: pd.DataFrame,
    *,
    horizon_max: int,
    value_cols: Sequence[str],
    site_col: str = "cid_seq",
    cid_seq: int,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    fc_site = _normalize_fcst_frame(fc_df, site_col)
    fc_site = fc_site[fc_site[site_col] == cid_seq]
    for h in range(1, int(horizon_max) + 1):
        T = sample_t0 + pd.Timedelta(hours=h)
        for col in value_cols:
            key = f"fcst_{col}_{h:03d}"
            val = joined_row.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            val_f = float(val)
            pool = fc_site[
                (fc_site["target_time"] == T) & (fc_site["issue_time"] <= sample_t0)
            ]
            if pool.empty:
                raise AssertionError(f"누수 검증 실패: 후보 없음 t0={sample_t0}, T={T}, col={col}")
            best = pool.loc[pool["issue_time"].idxmax()]
            if not np.isclose(float(best[col]), val_f, rtol=rtol, atol=atol):
                raise AssertionError(
                    f"누수 검증 실패: t0={sample_t0}, h={h}, col={col}, val={val_f}, expected={best[col]}"
                )


def era5_hourly_to_shortterm_service_frame(df: pd.DataFrame) -> pd.DataFrame:
    """ERA5 격자 hourly → 단기예보 슬롄 이름 proxy (문서 §2.6.2)."""
    out = df.copy()
    if "t2m_c" in out.columns:
        out["tmp"] = out["t2m_c"]
    if "tp_mm" in out.columns:
        out["pcp"] = out["tp_mm"]
        tp = out["tp_mm"].astype(float)
        out["pty"] = np.where(tp > 0.1, 1.0, 0.0)
        out["pop"] = np.minimum(tp * 20.0, 100.0)
    else:
        out["pty"] = 0.0
        out["pop"] = 0.0
    if "tcc" in out.columns:
        tcc = out["tcc"].clip(lower=0.0, upper=1.0).astype(float)
        out["sky"] = (1.0 + 3.0 * tcc).clip(1.0, 4.0)
    out["sno"] = 0.0
    return out


def attach_shortterm_channels_from_era5_site(era_site: pd.DataFrame) -> pd.DataFrame:
    if era_site is None or era_site.empty:
        return era_site if era_site is not None else pd.DataFrame()
    return era5_hourly_to_shortterm_service_frame(era_site)
