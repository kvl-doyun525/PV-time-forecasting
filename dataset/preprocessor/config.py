"""
PV feature mart 전처리 공통 설정.

PhotoRec 조각(recup_dir.7/f567523592 등) 및 `feature_mart_builder.py` 역추적로 복구.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# ── 저장소 루트 (…/dataset/preprocessor/config.py → 레포 루트) ─────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]

SNAPSHOT_DIR = _REPO_ROOT / "project" / "artifacts" / "dataset_snapshot"
FEATURE_MART = _REPO_ROOT / "project" / "artifacts" / "feature_mart"
FEATURE_MART_PER_SITE = _REPO_ROOT / "project" / "artifacts" / "feature_mart_per_site"
FEATURE_MART_TRACK_B_PER_SITE = _REPO_ROOT / "project" / "artifacts" / "feature_mart_track_b_per_site"

MANIFEST_PATH = _REPO_ROOT / "project" / "artifacts" / "split_manifest.yaml"
PER_SITE_MANIFEST_PATH = _REPO_ROOT / "project" / "artifacts" / "per_site_split_manifest.json"

# split_manifest.yaml / 실행 문서와 동일한 기본 경계
DATA_START = "2022-01-01 00:00:00"
DATA_END = "2026-04-23 23:00:00"
TRAIN_END = "2024-12-31 23:00:00"
VALID_END = "2025-09-30 23:00:00"
TEST_END = "2026-04-23 23:00:00"

# ── Feature / Track B ───────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "normalized_power",
    "ta", "rn", "ws", "wd", "hm", "pa", "si", "ss", "dc10Tca",
    "solar_elevation", "solar_azimuth", "clearsky_ghi",
    "pv_roll_mean_24h", "pv_roll_mean_72h", "pv_roll_mean_168h",
    "pv_roll_std_24h", "pv_roll_std_72h", "pv_roll_std_168h",
    "pv_lag_24h", "pv_lag_168h",
    "hour", "dayofweek", "month", "dayofyear", "is_holiday",
    "t2m_c", "reh", "wsd", "vec", "tp_mm", "tcc",
]

ASOS_FEATURE_COLS: dict[str, str] = {
    "ta": "ta",
    "rn": "rn",
    "ws": "ws",
    "wd": "wd",
    "hm": "hm",
    "pa": "pa",
    "ss": "ss",
    "icsr": "si",
}

ERA5_FEATURE_COLS: tuple[str, ...] = ("t2m_c", "reh", "wsd", "vec", "tp_mm", "tcc")

SOLAR_FEATURE_COLS: tuple[str, ...] = ("solar_elevation", "solar_azimuth", "clearsky_ghi")

ROLLING_WINDOWS: tuple[int, ...] = (24, 72, 168)
LAG_HOURS: tuple[int, ...] = (24, 168)

TRACK_B_HORIZON_MAX: int = 72

TRACK_B_SERVICE_FCST_COLS: tuple[str, ...] = (
    "tmp", "reh", "wsd", "vec", "pcp", "sky", "pty", "pop", "sno",
)
TRACK_B_ERA5_NATIVE_FCST_COLS: tuple[str, ...] = ERA5_FEATURE_COLS
TRACK_B_FCST_VALUE_COLS: tuple[str, ...] = TRACK_B_ERA5_NATIVE_FCST_COLS


@dataclass
class PreprocessConfig:
    snapshot_dir: Path = field(default_factory=lambda: SNAPSHOT_DIR)
    feature_mart_dir: Path = field(default_factory=lambda: FEATURE_MART)
    feature_mart_per_site_dir: Path = field(default_factory=lambda: FEATURE_MART_PER_SITE)

    global_time_start: str = DATA_START
    global_time_end: str = DATA_END
    train_end: str = TRAIN_END
    valid_end: str = VALID_END
    test_end: str = TEST_END

    max_interp_gap_hours: int = 1
    min_site_coverage: float = 0.1
    use_era5: bool = True

    split_mode: str = "global"  # "global" | "per_site"
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    min_split_hours: int = 500
    positive_threshold: float = 0.01

    holiday_years: Tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)
    per_site_manifest_path: Path = field(default_factory=lambda: PER_SITE_MANIFEST_PATH)
