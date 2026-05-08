"""
기상 데이터 join 모듈.

ASOS 관측(kma_obs_asos_hourly.parquet)과
ERA5 NWP(era5_nwp_input_raw.parquet)를
site_to_kma_grid.csv 매핑을 기준으로 site별 DataFrame에 결합한다.

우선순위
--------
- 기온/습도/풍속 : ASOS 관측 (ERA5 bias correction 전까지 ASOS 우선)
- 일사량 (si)   : ASOS icsr 전용 (ERA5 에서는 수집 안 함)
- ERA5 컬럼     : t2m_c, reh, wsd, vec, tp_mm, tcc (별도 컬럼으로 부가)

설계 원칙
---------
- 메모리 효율을 위해 ASOS 는 station 단위로 먼저 groupby 후 cid_seq에 매핑
- ERA5 는 이미 cid_seq 기준으로 저장되어 있으므로 merge만 수행
- 모두 UTC → KST(Asia/Seoul) 기준으로 timestamp 정렬
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    SNAPSHOT_DIR,
    ASOS_FEATURE_COLS,
    ERA5_FEATURE_COLS,
    PreprocessConfig,
)

log = logging.getLogger(__name__)


# ── ASOS ─────────────────────────────────────────────────────────────────────

def load_asos(path: Optional[Path] = None) -> pd.DataFrame:
    """ASOS 시간 관측 로드 + 기본 전처리."""
    path = path or (SNAPSHOT_DIR / "kma_obs_asos_hourly.parquet")
    df = pd.read_parquet(path)

    df["timestamp"] = pd.to_datetime(df["tm"])
    df["stnId"]     = df["stnId"].astype(str).str.strip()

    # 수치형 컬럼만 float 변환
    numeric_cols = list(ASOS_FEATURE_COLS.keys())
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["timestamp", "stnId"] + [c for c in numeric_cols if c in df.columns]
    return df[keep].copy()


def load_station_mapping(path: Optional[Path] = None) -> pd.DataFrame:
    """site_to_kma_grid.csv 로드 → cid_seq, asos_stn_id 매핑."""
    path = path or (SNAPSHOT_DIR / "site_to_kma_grid.csv")
    df = pd.read_csv(path)
    df["asos_stn_id"] = df["asos_stn_id"].astype(str).str.strip()
    return df[["cid_seq", "asos_stn_id"]].drop_duplicates("cid_seq")


def build_asos_for_site(
    cid_seq: int,
    asos_by_stn: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    단일 site의 ASOS feature DataFrame 반환.
    asos_by_stn: {stnId: DataFrame} 사전 (미리 groupby 결과)
    """
    row = mapping.loc[mapping["cid_seq"] == cid_seq]
    if row.empty:
        return None
    stn_id = str(row.iloc[0]["asos_stn_id"])
    asos = asos_by_stn.get(stn_id)
    if asos is None or asos.empty:
        return None

    # ASOS 컬럼 이름 → feature 컬럼 이름으로 변경
    rename = {k: v for k, v in ASOS_FEATURE_COLS.items() if k in asos.columns}
    asos = asos.rename(columns=rename)
    asos = asos.set_index("timestamp")
    # 중복 timestamp 제거 (같은 시각 중복 레코드 평균)
    asos = asos.groupby(asos.index).mean(numeric_only=True)
    return asos[list(rename.values())]


def build_asos_lookup(asos_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """ASOS DataFrame을 stnId 기준으로 분할하여 dict 반환 (한 번만 실행)."""
    return {stn: grp.drop(columns=["stnId"]) for stn, grp in asos_df.groupby("stnId")}


# ── ERA5 ──────────────────────────────────────────────────────────────────────

def load_era5(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """ERA5 NWP 로드. 파일이 없으면 None 반환."""
    path = path or (SNAPSHOT_DIR / "era5_nwp_input_raw.parquet")
    if not path.exists():
        log.info("ERA5 파일 없음 — ERA5 feature 비활성화")
        return None
    log.info("ERA5 로드 중: %s", path)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cid_seq"]   = df["cid_seq"].astype(int)
    keep = ["timestamp", "cid_seq"] + [c for c in ERA5_FEATURE_COLS if c in df.columns]
    return df[keep].copy()


def build_era5_for_site(
    cid_seq: int,
    era5_by_site: Optional[dict[int, pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    """단일 site ERA5 DataFrame 반환. era5_by_site 없으면 None."""
    if era5_by_site is None:
        return None
    era5 = era5_by_site.get(cid_seq)
    if era5 is None or era5.empty:
        return None
    era5 = era5.set_index("timestamp")
    era5 = era5.groupby(era5.index).mean(numeric_only=True)
    return era5


def build_era5_lookup(era5_df: Optional[pd.DataFrame]) -> Optional[dict[int, pd.DataFrame]]:
    """ERA5 DataFrame을 cid_seq 기준으로 분할하여 dict 반환."""
    if era5_df is None:
        return None
    return {
        int(cid): grp.drop(columns=["cid_seq"])
        for cid, grp in era5_df.groupby("cid_seq")
    }


# ── 통합 join ─────────────────────────────────────────────────────────────────

def join_weather(
    site_df: pd.DataFrame,          # pv_cleaner 출력 (timestamp index)
    cid_seq: int,
    asos_lookup: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
    era5_lookup: Optional[dict[int, pd.DataFrame]],
    cfg: PreprocessConfig,
) -> pd.DataFrame:
    """
    site_df 에 ASOS / ERA5 날씨 feature를 join하여 반환.

    Parameters
    ----------
    site_df     : timestamp 인덱스의 cleaned PV DataFrame
    cid_seq     : 대상 site ID
    asos_lookup : {stnId: DataFrame}  (load_asos → build_asos_lookup 결과)
    mapping     : cid_seq ↔ asos_stn_id 매핑
    era5_lookup : {cid_seq: DataFrame} 또는 None
    """
    result = site_df.copy()

    # --- ASOS join ---
    asos = build_asos_for_site(cid_seq, asos_lookup, mapping)
    if asos is not None:
        # reindex 로 site 시간 범위에 맞춤 (없는 시간 → NaN)
        asos = asos.reindex(result.index)
        for col in asos.columns:
            result[col] = asos[col]
    else:
        log.debug("cid_seq=%d ASOS 데이터 없음", cid_seq)
        for col in ASOS_FEATURE_COLS.values():
            result[col] = float("nan")

    # --- ERA5 join (선택적) ---
    if cfg.use_era5:
        era5 = build_era5_for_site(cid_seq, era5_lookup)
        if era5 is not None:
            era5 = era5.reindex(result.index)
            for col in era5.columns:
                result[col] = era5[col]

    return result


def prepare_weather_lookups(
    cfg: PreprocessConfig,
) -> tuple[dict, pd.DataFrame, Optional[dict]]:
    """
    ASOS / ERA5 데이터를 한 번만 로드하여 lookup dict 생성.
    Returns: (asos_lookup, station_mapping, era5_lookup)
    """
    log.info("ASOS 관측 데이터 로드 중...")
    asos_df  = load_asos(cfg.snapshot_dir / "kma_obs_asos_hourly.parquet")
    asos_lkp = build_asos_lookup(asos_df)
    del asos_df

    mapping = load_station_mapping(cfg.snapshot_dir / "site_to_kma_grid.csv")

    era5_lkp: Optional[dict] = None
    if cfg.use_era5:
        era5_path = cfg.snapshot_dir / "era5_nwp_input_raw.parquet"
        era5_df = load_era5(era5_path)
        if era5_df is not None:
            log.info("ERA5 lookup 생성 중...")
            era5_lkp = build_era5_lookup(era5_df)
            del era5_df

    return asos_lkp, mapping, era5_lkp
