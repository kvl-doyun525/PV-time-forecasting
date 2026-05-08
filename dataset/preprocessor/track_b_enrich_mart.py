"""
Track B per-site mart enrich (fcst_* wide fan 추가).

복구: recup_dir.7/f567522456.txt, f567300744.txt, f567522896.txt.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from config import FEATURE_MART_TRACK_B_PER_SITE, SNAPSHOT_DIR
from track_b_forecast_join import (
    attach_shortterm_channels_from_era5_site,
    enrich_frame_with_forecast_covariates,
    enrich_frame_with_hourly_era5_valid_covariates,
    forecast_covariate_column_names,
)
from weather_joiner import build_era5_lookup, load_era5

log = logging.getLogger(__name__)

_SPLITS = ("train", "valid", "test")


def enrich_track_b_mart(
    input_mart_dir: Path,
    output_mart_dir: Path,
    *,
    join_mode: str,
    fcst_schema: str,
    forecast_path: Optional[Path],
    horizon_max: int,
    value_cols: Sequence[str],
    site_col: str = "cid_seq",
    site_ids: Optional[Sequence[int]] = None,
) -> dict:
    input_mart_dir = Path(input_mart_dir)
    output_mart_dir = Path(output_mart_dir)
    output_mart_dir.mkdir(parents=True, exist_ok=True)

    cols = list(value_cols)
    H = int(horizon_max)
    new_fc_names = forecast_covariate_column_names(cols, H)

    for name in ("scaler_stats.json", "build_report.json", "quality_report.json"):
        src = input_mart_dir / name
        if src.exists():
            shutil.copy2(src, output_mart_dir / name)

    fc_path = Path(forecast_path) if forecast_path else None
    if fc_path is None or not fc_path.exists():
        if join_mode == "issue_target":
            fc_path = SNAPSHOT_DIR / "era5_fcst_long.parquet"
        else:
            fc_path = SNAPSHOT_DIR / "era5_nwp_input_raw.parquet"

    fc_long: Optional[pd.DataFrame] = None
    era_lkp: Optional[dict[int, pd.DataFrame]] = None

    if join_mode == "issue_target":
        if not fc_path.exists():
            raise FileNotFoundError(f"issue_target 예보 parquet 없음: {fc_path}")
        fc_long = pd.read_parquet(fc_path)
    else:
        era5_df = load_era5(fc_path)
        era_lkp = build_era5_lookup(era5_df) if era5_df is not None else {}

    n_files = 0
    total_rows = 0

    for split in _SPLITS:
        in_split = input_mart_dir / split
        if not in_split.is_dir():
            log.warning("split 디렉터리 없음: %s", in_split)
            continue
        for p in sorted(in_split.glob("*.parquet")):
            cid = int(p.stem)
            if site_ids is not None and cid not in site_ids:
                continue
            df = pd.read_parquet(p)

            if join_mode == "issue_target":
                assert fc_long is not None
                enriched = enrich_frame_with_forecast_covariates(
                    df,
                    fc_long,
                    cid_seq=cid,
                    horizon_max=H,
                    value_cols=cols,
                    site_col=site_col,
                )
            else:
                era_site = era_lkp.get(cid) if era_lkp is not None else None
                era_in = (
                    era_site.reset_index()
                    if era_site is not None and not era_site.empty
                    else pd.DataFrame()
                )
                if fcst_schema == "shortterm_aligned" and not era_in.empty:
                    era_in = attach_shortterm_channels_from_era5_site(era_in)
                enriched = enrich_frame_with_hourly_era5_valid_covariates(
                    df,
                    era_in,
                    horizon_max=H,
                    value_cols=cols,
                )

            out_p = output_mart_dir / split / f"{cid}.parquet"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            enriched.to_parquet(out_p, engine="pyarrow")
            n_files += 1
            total_rows += len(enriched)

    track_b_report = {
        "track_b_version": "1.2",
        "join_mode": join_mode,
        "fcst_schema": fcst_schema,
        "input_mart_dir": str(input_mart_dir.resolve()),
        "forecast_path": str(fc_path.resolve()) if fc_path.exists() else None,
        "horizon_max": H,
        "fcst_value_cols": cols,
        "added_columns": new_fc_names,
        "n_parquet_written": n_files,
        "total_rows": total_rows,
    }
    report_path = output_mart_dir / "track_b_build_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(track_b_report, f, indent=2, ensure_ascii=False)
    log.info("트랙 B 리포트 저장: %s", report_path)

    br_path = output_mart_dir / "build_report.json"
    if br_path.exists():
        with open(br_path, encoding="utf-8") as f:
            br = json.load(f)
        br["track_b"] = {
            "join_mode": join_mode,
            "fcst_schema": fcst_schema,
            "forecast_path": track_b_report["forecast_path"],
            "horizon_max": H,
            "n_fcst_features": len(cols),
            "n_added_cols": len(new_fc_names),
        }
        with open(br_path, "w", encoding="utf-8") as f:
            json.dump(br, f, indent=2, ensure_ascii=False)

    return track_b_report


def default_output_dir_for_input(input_mart_dir: Path) -> Path:
    """per_site mart → track_b 기본 출력 경로."""
    p = Path(input_mart_dir).resolve()
    if p.name == "feature_mart_per_site" or "per_site" in p.name:
        return FEATURE_MART_TRACK_B_PER_SITE
    return p.parent / f"{p.name}_track_b"
