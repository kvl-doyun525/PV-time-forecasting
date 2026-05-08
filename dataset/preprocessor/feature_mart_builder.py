"""
Feature Mart 구축 오케스트레이터.

파이프라인
----------
1. PV 원시 데이터 정제  (pv_cleaner)
2. ASOS / ERA5 기상 join (weather_joiner)
3. 태양 위치 feature   (solar_features)
4. 파생 feature        (derived_features)
5. z-score 정규화      (train 구간 통계 기준)
6. train / valid / test 분할 후 site별 parquet 저장
7. scaler_stats.json 저장

출력 구조
---------
feature_mart/
  train/{cid_seq}.parquet
  valid/{cid_seq}.parquet
  test/{cid_seq}.parquet
  scaler_stats.json
  build_report.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FEATURE_COLS,
    ASOS_FEATURE_COLS,
    ERA5_FEATURE_COLS,
    SOLAR_FEATURE_COLS,
    PreprocessConfig,
    SNAPSHOT_DIR,
    FEATURE_MART,
    FEATURE_MART_PER_SITE,
)
from pv_cleaner import clean_all_sites
from weather_joiner import prepare_weather_lookups, join_weather
from solar_features import compute_solar_features
from derived_features import add_all_derived

log = logging.getLogger(__name__)


# ── 정규화 ────────────────────────────────────────────────────────────────────

# 정규화에서 제외할 컬럼 (이미 정규화됨 or 물리적 기준점이 중요한 값 or 범주형)
_NO_SCALE = {
    "normalized_power",   # 이미 [0, 1] 범위 (capacity 기준)
    "is_imputed",
    "is_holiday",
    "hour", "dayofweek", "month", "dayofyear",
    # 태양 위치: 물리적 기준점(0) 유지 필요
    #   solar_elevation < 0 → 야간, ≥ 0 → 낮 구분에 사용
    #   clearsky_ghi = 0 → 야간 구분에 사용
    "solar_elevation",
    "solar_azimuth",
    "clearsky_ghi",
}


def fit_scaler(train_df: pd.DataFrame, cols: list[str]) -> dict[str, dict]:
    """train 구간에서 z-score 파라미터(mean, std) 계산."""
    stats: dict[str, dict] = {}
    for col in cols:
        if col not in train_df.columns or col in _NO_SCALE:
            continue
        mean = float(train_df[col].mean(skipna=True))
        std  = float(train_df[col].std(skipna=True))
        if std < 1e-8:
            std = 1.0   # 분산 0 방지
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_scaler(df: pd.DataFrame, stats: dict[str, dict]) -> pd.DataFrame:
    """z-score 적용."""
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            df[col] = (df[col] - s["mean"]) / s["std"]
    return df


def load_scaler(path: Path) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    return data["features"]


def save_scaler(stats: dict[str, dict], path: Path, fit_end: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version":  "1.0",
        "fit_end":  fit_end,
        "features": stats,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("scaler_stats 저장: %s", path)


# ── 분할 ──────────────────────────────────────────────────────────────────────

def split_df(
    df: pd.DataFrame,
    train_end: str,
    valid_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """글로벌 날짜 경계 기준 분할."""
    train = df.loc[:train_end]
    valid = df.loc[pd.Timestamp(train_end) + pd.Timedelta(hours=1) : valid_end]
    test  = df.loc[pd.Timestamp(valid_end) + pd.Timedelta(hours=1) :]
    return train, valid, test


def per_site_split(
    df: pd.DataFrame,
    target_col: str = "normalized_power",
    ratios: tuple = (0.70, 0.15, 0.15),
    min_split_hours: int = 500,
    positive_threshold: float = 0.01,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], dict]:
    """
    site별 실제 발전 기간(낮 시간대 양의 발전 기준)에 비율을 적용해 분할.

    활성 기간 판단 방식
    ------------------
    1차: 낮 시간대(solar_elevation > 5) 중 target_col > positive_threshold 인 구간
         → 인버터 0 기록값 / 야간 NaN 을 활성 기간에서 제외
    2차 fallback: solar_elevation 컬럼 없는 경우 first/last_valid_index 사용

    Parameters
    ----------
    df                 : 전체 타임라인 DataFrame
    target_col         : 활성 기간 판단에 사용할 컬럼 (normalized_power)
    ratios             : (train, valid, test) 비율 (합산 1.0)
    min_split_hours    : 각 split당 최소 행 수. 미달 시 None 반환
    positive_threshold : 활성으로 간주할 target_col 최소값

    Returns
    -------
    (train_df, valid_df, test_df, boundaries_dict)
    실패 시 (None, None, None, {})
    """
    # ── 활성 기간: 낮 시간대 실제 양의 발전 구간 ─────────────────────────────
    if "solar_elevation" in df.columns:
        daytime_pos = df[
            (df["solar_elevation"] > 5) &
            (df[target_col] > positive_threshold)
        ]
        if len(daytime_pos) == 0:
            return None, None, None, {}
        # 하루 단위로 정렬: first 발전일 00:00 ~ last 발전일 23:00
        first_date = daytime_pos.index.min().normalize()
        last_date  = daytime_pos.index.max().normalize() + pd.Timedelta(hours=23)
        first = df.index[df.index >= first_date][0]  if (df.index >= first_date).any() else None
        last  = df.index[df.index <= last_date][-1]  if (df.index <= last_date).any() else None
    else:
        # fallback: solar_elevation 없을 때
        first = df[target_col].first_valid_index()
        last  = df[target_col].last_valid_index()

    if first is None or last is None:
        return None, None, None, {}

    active = df.loc[first:last]
    n = len(active)

    r_train, r_valid, _ = ratios
    i_train = int(n * r_train)
    i_valid = int(n * (r_train + r_valid))

    train = active.iloc[:i_train]
    valid = active.iloc[i_train:i_valid]
    test  = active.iloc[i_valid:]

    if len(train) < min_split_hours or len(valid) < min_split_hours or len(test) < min_split_hours:
        return None, None, None, {}

    boundaries = {
        "active_start":        str(first),
        "active_end":          str(last),
        "train_end":           str(train.index[-1]),
        "valid_end":           str(valid.index[-1]),
        "test_end":            str(test.index[-1]),
        "n_train":             len(train),
        "n_valid":             len(valid),
        "n_test":              len(test),
        "positive_threshold":  positive_threshold,
    }
    return train, valid, test, boundaries


# ── 메인 빌더 ─────────────────────────────────────────────────────────────────

def build_feature_mart(
    cfg: Optional[PreprocessConfig] = None,
    site_ids: Optional[list[int]] = None,
) -> dict:
    """
    전체 feature mart 빌드.

    Parameters
    ----------
    cfg      : PreprocessConfig (None이면 기본값 사용)
    site_ids : 특정 site만 처리 (None이면 전체)

    Returns
    -------
    build_report dict (저장 site 수, 누락 site 수, 총 행 수 등)
    """
    cfg = cfg or PreprocessConfig()

    # split_mode에 따라 출력 디렉터리 결정
    if cfg.split_mode == "per_site":
        out_dir = cfg.feature_mart_per_site_dir
        log.info("=== per_site split 모드: 출력 → %s ===", out_dir)
    else:
        out_dir = cfg.feature_mart_dir

    for split in ("train", "valid", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    # ── Step 1: PV 정제 ──────────────────────────────────────────────────────
    log.info("=== Step 1: PV 데이터 정제 ===")
    pv_map = clean_all_sites(cfg)
    if site_ids:
        pv_map = {k: v for k, v in pv_map.items() if k in site_ids}
    log.info("정제 완료: %d site", len(pv_map))

    # ── Step 2: 기상 lookup 준비 ─────────────────────────────────────────────
    log.info("=== Step 2: 기상 데이터 lookup 준비 ===")
    asos_lkp, mapping, era5_lkp = prepare_weather_lookups(cfg)

    # ── Step 3: plant_meta (위경도) 로드 ────────────────────────────────────
    meta = pd.read_parquet(cfg.snapshot_dir / "plant_meta.parquet")
    coord_map = (
        meta[["cid_seq", "latitude", "longitude"]]
        .drop_duplicates("cid_seq")
        .set_index("cid_seq")
    )

    # ── Step 4: scaler 수집용 train 샘플 누적 (메모리 절약을 위해 reservoir) ─
    # 각 feature별로 최대 50만 행 샘플만 수집하여 scaler 학습
    _SCALER_SAMPLE_MAX = 500_000
    scaler_samples: dict[str, list[float]] = {}

    # ── Step 5: site별 순차 처리 ─────────────────────────────────────────────
    log.info("=== Step 3~5: site별 feature 구축 및 분할 저장 (mode=%s) ===", cfg.split_mode)
    n_saved = 0
    n_skip  = 0
    total_rows = 0
    skipped_sites: list[int] = []
    per_site_boundaries: dict[int, dict] = {}   # per_site 모드 전용

    for cid, pv_df in tqdm(pv_map.items(), desc="feature 구축", unit="site"):
        try:
            site_df = _build_site_features(
                cid, pv_df, asos_lkp, mapping, era5_lkp, coord_map, cfg
            )
        except Exception as e:
            log.warning("cid_seq=%d feature 구축 실패: %s", cid, e)
            n_skip += 1
            skipped_sites.append(cid)
            continue

        if site_df is None or site_df.empty:
            n_skip += 1
            skipped_sites.append(cid)
            continue

        # train/valid/test 분할 (split_mode에 따라)
        if cfg.split_mode == "per_site":
            train_df, valid_df, test_df, boundaries = per_site_split(
                site_df,
                target_col="normalized_power",
                ratios=cfg.split_ratios,
                min_split_hours=cfg.min_split_hours,
                positive_threshold=cfg.positive_threshold,
            )
            if train_df is None:
                log.warning(
                    "cid_seq=%d per_site split 조건 미달 (min_split_hours=%d) — 건너뜀",
                    cid, cfg.min_split_hours,
                )
                n_skip += 1
                skipped_sites.append(cid)
                continue
            per_site_boundaries[cid] = boundaries
        else:
            train_df, valid_df, test_df = split_df(site_df, cfg.train_end, cfg.valid_end)

        # scaler 샘플 수집 (train 구간)
        for col in train_df.columns:
            if col in _NO_SCALE:
                continue
            vals = train_df[col].dropna().tolist()
            existing = scaler_samples.get(col, [])
            combined = existing + vals
            # 최대 크기 초과 시 무작위 서브샘플
            if len(combined) > _SCALER_SAMPLE_MAX:
                idx = np.random.choice(
                    len(combined), _SCALER_SAMPLE_MAX, replace=False
                )
                combined = [combined[i] for i in idx]
            scaler_samples[col] = combined

        # 임시 저장 (정규화 전)
        for split_name, split_df_part in [
            ("train", train_df), ("valid", valid_df), ("test", test_df)
        ]:
            p = out_dir / split_name / f"{cid}.parquet"
            split_df_part.to_parquet(p, engine="pyarrow")

        total_rows += len(site_df)
        n_saved += 1

    log.info("저장 완료: %d site / 누락: %d site", n_saved, n_skip)

    # per_site 모드: 경계 manifest 저장
    if cfg.split_mode == "per_site" and per_site_boundaries:
        manifest_out = {
            "mode": "per_site",
            "split_ratios": list(cfg.split_ratios),
            "min_split_hours": cfg.min_split_hours,
            "n_sites": len(per_site_boundaries),
            "sites": {str(k): v for k, v in per_site_boundaries.items()},
        }
        with open(cfg.per_site_manifest_path, "w") as f:
            json.dump(manifest_out, f, indent=2, ensure_ascii=False)
        log.info("per_site split manifest 저장: %s", cfg.per_site_manifest_path)

    # ── Step 6: scaler 학습 + 적용 ─────────────────────────────────────────
    log.info("=== Step 6: 정규화 scaler 학습 + 적용 ===")
    scaler_stats: dict[str, dict] = {}
    for col, vals in scaler_samples.items():
        arr = np.array(vals, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            log.warning("컬럼 %s: 유효 값 없음 — scaler 생략", col)
            continue
        mean = float(arr.mean())
        std  = float(arr.std())
        if std < 1e-8:
            std = 1.0
        scaler_stats[col] = {"mean": mean, "std": std}

    save_scaler(scaler_stats, out_dir / "scaler_stats.json", fit_end=cfg.train_end)

    log.info("정규화 적용 중 (%d site)...", n_saved)
    for cid in tqdm(pv_map.keys(), desc="정규화", unit="site"):
        for split_name in ("train", "valid", "test"):
            p = out_dir / split_name / f"{cid}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            df = apply_scaler(df, scaler_stats)
            df.to_parquet(p, engine="pyarrow")

    # ── Step 7: 빌드 리포트 ──────────────────────────────────────────────────
    report = {
        "built_at":        datetime.now().isoformat(),
        "split_mode":      cfg.split_mode,
        "n_sites_saved":   n_saved,
        "n_sites_skipped": n_skip,
        "skipped_cid_seqs": skipped_sites[:50],
        "total_rows":      total_rows,
        "era5_used":       era5_lkp is not None,
        "feature_cols":    FEATURE_COLS,
    }
    if cfg.split_mode == "global":
        report["train_end"] = cfg.train_end
        report["valid_end"] = cfg.valid_end
    else:
        report["split_ratios"]    = list(cfg.split_ratios)
        report["min_split_hours"] = cfg.min_split_hours
    report_path = out_dir / "build_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("빌드 리포트 저장: %s", report_path)

    return report


# ── site 단위 feature 구축 내부 함수 ─────────────────────────────────────────

def _build_site_features(
    cid: int,
    pv_df: pd.DataFrame,
    asos_lkp: dict,
    mapping: pd.DataFrame,
    era5_lkp: Optional[dict],
    coord_map: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Optional[pd.DataFrame]:
    """단일 site의 전체 feature DataFrame 구축."""
    # 1) 기상 join
    site_df = join_weather(pv_df, cid, asos_lkp, mapping, era5_lkp, cfg)

    # 2) 태양 위치
    if cid in coord_map.index:
        row = coord_map.loc[cid]
        solar = compute_solar_features(
            site_df.index, float(row["latitude"]), float(row["longitude"])
        )
        site_df["solar_elevation"] = solar["solar_elevation"].values
        site_df["solar_azimuth"]   = solar["solar_azimuth"].values
        site_df["clearsky_ghi"]    = solar["clearsky_ghi"].values
    else:
        for col in SOLAR_FEATURE_COLS:
            site_df[col] = np.nan

    # 3) si(일사량) 처리
    #    ASOS icsr (MJ/m²/h) → W/m² 단위로 통일
    #    icsr 센서 없는 station은 NaN 유지 (clearsky_ghi 와 구별 가능하도록)
    #
    #    단위 변환: 1 MJ/m²/h = 1,000,000 J / 3,600 s = 277.78 W/m²
    #
    #    두 컬럼의 의미:
    #      si            = ASOS 실측 일사량 (구름·에어로졸 감쇠 포함)
    #      clearsky_ghi  = pvlib 이론 최대값 (구름 없음 가정)
    #    모델이 두 신호를 독립적으로 학습하도록 fallback 없이 NaN 유지
    _MJH_TO_W = 277.78
    if "si" in site_df.columns:
        # MJ/m²/h → W/m² (센서 있는 관측소만 유효값, 없는 관측소는 NaN 유지)
        site_df["si"] = site_df["si"] * _MJH_TO_W

    # 4) 파생 feature (rolling, lag, calendar)
    site_df = add_all_derived(site_df, holiday_years=cfg.holiday_years)

    # 5) 최종 컬럼 선택 (불필요 컬럼 제거)
    keep_cols = [c for c in FEATURE_COLS if c in site_df.columns]
    # ERA5 컬럼이 있으면 추가
    extra_era5 = [c for c in ERA5_FEATURE_COLS if c in site_df.columns]
    keep_cols  = keep_cols + [c for c in extra_era5 if c not in keep_cols]
    site_df = site_df[keep_cols]

    return site_df
