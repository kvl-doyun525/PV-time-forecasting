"""
ERA5 재분석 데이터 수집기 (학습용 NWP 입력 — 전략 B).

목적
----
PV 발전량 예측 모델 훈련 시 사용할 과거 기상 입력 데이터를 ECMWF CDS API로 수집한다.
일사량(`ssrd`)은 학습-추론 분포 불일치 및 ERA5 고편향 문제로 **의도적으로 제외**한다.

수집 변수
---------
| CDS 변수명                      | 출력 컬럼  | 설명         | 추론 대응 (단기예보)   |
|---------------------------------|-----------|--------------|----------------------|
| 2m_temperature                  | t2m       | 기온 (K)     | tmp (℃, -273.15)     |
| 10m_u_component_of_wind         | u10       | 동서풍 (m/s) | wsd + vec 계산       |
| 10m_v_component_of_wind         | v10       | 남북풍 (m/s) | —                    |
| 2m_dewpoint_temperature         | d2m       | 이슬점 (K)   | reh (%) 변환         |
| total_precipitation             | tp        | 강수량 (m)   | pcp (mm, ×1000)      |
| total_cloud_cover               | tcc       | 전운량 (0~1) | sky 코드 매핑         |

바이어스 교정 (`bias_correct=True`)
-------------------------------------
ASOS 관측(kma_obs_asos_hourly.parquet)을 기준으로 변수별 quantile mapping을 적용한다.
- t2m, u10, v10, d2m 에 적용
- tp, tcc 는 분포 특성상 quantile mapping 생략 (선택적)

출력 파일
---------
artifacts/dataset_snapshot/
  era5_nwp_input_raw.parquet          ← CDS 수집 원본 (사이트별 bilinear 보간)
  era5_nwp_bias_corrected.parquet     ← 바이어스 교정 후

사전 준비
---------
1. pip install cdsapi xarray netCDF4 scipy
2. ~/.cdsapirc 에 CDS API 키 설정 (또는 .env 에 CDS_API_KEY / CDS_API_URL)
   url: https://cds.climate.copernicus.eu/api
   key: <uid>:<api-key>
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from config import CollectConfig, collect_config

log = logging.getLogger(__name__)

# 수집 변수 (일사량 ssrd 제외)
ERA5_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "total_cloud_cover",
]

# CDS 변수명 → 출력 컬럼명 매핑
_VAR_MAP = {
    "t2m":  "t2m",
    "u10":  "u10",
    "v10":  "v10",
    "d2m":  "d2m",
    "tp":   "tp",
    "tcc":  "tcc",
}

# 한반도 영역 (N, W, S, E)
_KOREA_AREA = [38.5, 126.0, 33.0, 130.0]


class ERA5Collector:
    """
    ECMWF CDS API 를 통해 ERA5 재분석 데이터를 수집하고
    PV site 위경도로 bilinear interpolation 한 뒤 parquet 으로 저장한다.

    바이어스 교정 옵션을 지원하며, ASOS 관측 parquet 이 있으면
    quantile mapping 으로 체계적 편향을 제거한다.
    """

    def __init__(self, col_cfg: Optional[CollectConfig] = None):
        self.cfg = col_cfg or collect_config
        self._cds = None  # lazy init

    def _get_cds_client(self):
        """cdsapi.Client 지연 초기화."""
        if self._cds is not None:
            return self._cds
        try:
            import cdsapi  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "cdsapi 가 설치되지 않았습니다. `pip install cdsapi` 를 실행하세요."
            ) from e

        # .env 에 키가 있으면 ~/.cdsapirc 없이도 동작
        cds_key = os.environ.get("CDS_API_KEY", "")
        cds_url = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
        if cds_key:
            self._cds = cdsapi.Client(url=cds_url, key=cds_key, quiet=True)
        else:
            self._cds = cdsapi.Client(quiet=True)  # ~/.cdsapirc 사용
        return self._cds

    # ── 월별 NetCDF 다운로드 ─────────────────────────────────────────────────

    def download_month(
        self,
        year: int,
        month: int,
        tmp_dir: Optional[Path] = None,
    ) -> Path:
        """
        1개월 분량의 ERA5 데이터를 NetCDF 로 다운로드한다.

        CDS API 는 단일 요청 크기 제한(약 120,000 필드)이 있으므로
        연도 전체 대신 월 단위로 분할하여 요청한다.
          6변수 × 31일 × 24시간 = 4,464 필드/월 → 제한 이내

        Returns
        -------
        Path : 다운로드된 NetCDF 파일 경로
        """
        tmp = tmp_dir or (self.cfg.output_dir / "era5_tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        nc_path = tmp / f"era5_{year}_{month:02d}.nc"

        if nc_path.exists():
            log.info("이미 존재 — 건너뜀: %s", nc_path)
            return _unzip_if_needed(nc_path)

        log.info("ERA5 CDS API 요청: %d-%02d", year, month)
        days  = [f"{d:02d}" for d in range(1, 32)]
        hours = [f"{h:02d}:00" for h in range(24)]

        client = self._get_cds_client()
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable":     ERA5_VARIABLES,
                "year":         [str(year)],
                "month":        [f"{month:02d}"],
                "day":          days,
                "time":         hours,
                "area":         _KOREA_AREA,
                "format":       "netcdf",
            },
            str(nc_path),
        )
        log.info("다운로드 완료: %s (%.1f MB)", nc_path, nc_path.stat().st_size / 1e6)

        # CDS 신규 API는 NetCDF 파일들을 ZIP으로 묶어 반환한다.
        # ZIP 여부를 확인하여 자동 압축 해제 후 실제 .nc 경로를 반환.
        nc_path = _unzip_if_needed(nc_path)
        return nc_path

    def download_year(
        self,
        year: int,
        tmp_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        한 연도를 월별로 분할하여 순차 다운로드한다.

        Returns
        -------
        List[Path] : 월별 NetCDF 파일 경로 목록 (12개 또는 그 이하)
        """
        from datetime import date as _date  # noqa: PLC0415
        today = _date.today()
        paths: List[Path] = []
        for month in range(1, 13):
            # 미래 월은 건너뜀
            if year > today.year or (year == today.year and month > today.month):
                log.debug("미래 월 건너뜀: %d-%02d", year, month)
                continue
            nc_path = self.download_month(year, month, tmp_dir)
            paths.append(nc_path)
        return paths

    # ── NetCDF → site별 DataFrame ─────────────────────────────────────────────

    def nc_to_site_df(
        self,
        nc_paths: "Path | List[Path]",
        site_coords: pd.DataFrame,  # columns: cid_seq, lat, lon
    ) -> pd.DataFrame:
        """
        NetCDF 격자 데이터를 site 위경도로 nearest-neighbor 보간하여 DataFrame 반환.

        Parameters
        ----------
        nc_paths    : download_month() 경로 또는 경로 목록
        site_coords : cid_seq, lat, lon 컬럼을 가진 site 위경도 목록

        Returns
        -------
        DataFrame columns:
            timestamp (UTC), cid_seq, t2m, u10, v10, d2m, tp, tcc
        """
        try:
            import xarray as xr  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("`pip install xarray netCDF4` 를 실행하세요.") from e

        # 단일 경로도 리스트로 통일
        if isinstance(nc_paths, Path):
            # 디렉터리이면 하위 *.nc 전부 포함
            if nc_paths.is_dir():
                nc_paths = sorted(nc_paths.glob("*.nc"))
            else:
                nc_paths = [nc_paths]

        # Path 가 아닌 str 목록도 허용
        nc_paths = [Path(p) for p in nc_paths]
        log.info("NetCDF 로드: %d 파일", len(nc_paths))

        # dask 없이도 동작하도록 각 파일을 개별 open → xr.merge 로 합친다.
        # (open_mfdataset 은 dask 에 의존하므로 사용하지 않는다)
        with xr.set_options(use_new_combine_kwarg_defaults=True):
            datasets = [
                xr.open_dataset(p, engine="netcdf4") for p in nc_paths
            ]
        if len(datasets) == 1:
            ds = datasets[0]
        else:
            ds = xr.merge(datasets, compat="override")

        frames: list[pd.DataFrame] = []
        for _, row in site_coords.iterrows():
            site_ds = ds.sel(
                latitude=row["lat"],
                longitude=row["lon"],
                method="nearest",
            )
            df = site_ds.to_dataframe().reset_index()
            # valid_time 또는 time 컬럼 통일
            if "valid_time" in df.columns:
                df = df.rename(columns={"valid_time": "timestamp"})
            elif "time" in df.columns:
                df = df.rename(columns={"time": "timestamp"})
            keep = ["timestamp"] + [c for c in _VAR_MAP if c in df.columns]
            df = df[keep].copy()
            df["cid_seq"] = int(row["cid_seq"])
            frames.append(df)

        ds.close()
        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result = result.sort_values(["cid_seq", "timestamp"])
        return result

    # ── 단위 변환 (ERA5 → 단기예보 대응 단위) ────────────────────────────────

    @staticmethod
    def convert_units(df: pd.DataFrame) -> pd.DataFrame:
        """
        ERA5 원본 단위 → 단기예보와 비교 가능한 단위로 변환.

        변환 내용:
          t2m  : K → ℃  (- 273.15)
          d2m  : K → 상대습도 %  (Magnus 공식)
          tp   : m → mm (× 1000)
          u10/v10 → wsd (m/s), vec (°) 파생 컬럼 추가
        """
        df = df.copy()

        # 기온 K → ℃
        if "t2m" in df.columns:
            df["t2m_c"] = df["t2m"] - 273.15

        # 이슬점 → 상대습도 (Magnus)
        if "d2m" in df.columns and "t2m" in df.columns:
            T  = df["t2m"] - 273.15
            Td = df["d2m"] - 273.15
            df["reh"] = 100.0 * np.exp(
                (17.625 * Td) / (243.04 + Td) - (17.625 * T) / (243.04 + T)
            ).clip(0, 100)

        # 강수 m → mm
        if "tp" in df.columns:
            df["tp_mm"] = df["tp"] * 1000.0

        # 풍속 벡터 → 속도 / 방향
        if "u10" in df.columns and "v10" in df.columns:
            df["wsd"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
            df["vec"] = (270.0 - np.degrees(np.arctan2(df["v10"], df["u10"]))) % 360.0

        return df

    # ── 바이어스 교정 ─────────────────────────────────────────────────────────

    @staticmethod
    def bias_correct(
        era5_df: pd.DataFrame,
        asos_path: Path,
        site_grid_path: Path,
        variables: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        ASOS 관측 기준 quantile mapping 바이어스 교정.

        ERA5 격자 → ASOS 지점을 site_to_kma_grid.csv 의 asos_stn_id 로 매핑하고,
        각 변수별로 ERA5 분위수 → 관측 분위수로 매핑한다.

        Parameters
        ----------
        era5_df        : convert_units() 까지 적용된 ERA5 DataFrame
        asos_path      : kma_obs_asos_hourly.parquet 경로
        site_grid_path : site_to_kma_grid.csv 경로
        variables      : 교정할 컬럼 목록 (None이면 t2m_c, wsd, reh)
        """
        if variables is None:
            variables = ["t2m_c", "wsd", "reh"]

        if not asos_path.exists():
            log.warning("ASOS parquet 없음 — 바이어스 교정 생략: %s", asos_path)
            return era5_df

        log.info("바이어스 교정 시작 (변수: %s)", variables)
        asos = pd.read_parquet(asos_path)
        asos["tm"] = pd.to_datetime(asos["tm"])
        asos["hour"] = asos["tm"].dt.hour

        grid = pd.read_csv(site_grid_path, dtype={"asos_stn_id": str})

        # ASOS 풍속 컬럼 추가
        if "wsd" not in asos.columns and "ws" in asos.columns:
            asos = asos.rename(columns={"ws": "wsd"})

        # ASOS 컬럼: ta → t2m_c, hm → reh 대응
        asos_col_map = {"t2m_c": "ta", "wsd": "wsd", "reh": "hm"}

        result_frames: list[pd.DataFrame] = []

        for cid_seq, group in era5_df.groupby("cid_seq"):
            stn_rows = grid[grid["cid_seq"] == cid_seq]
            if stn_rows.empty:
                result_frames.append(group)
                continue
            stn_id = str(stn_rows.iloc[0]["asos_stn_id"])
            asos_stn = asos[asos["stnId"].astype(str) == stn_id]
            if asos_stn.empty:
                result_frames.append(group)
                continue

            group = group.copy()
            for var in variables:
                asos_var = asos_col_map.get(var)
                if asos_var is None or asos_var not in asos_stn.columns:
                    continue
                obs_vals  = asos_stn[asos_var].dropna().values
                era5_vals = group[var].dropna().values
                if len(obs_vals) < 10 or len(era5_vals) < 10:
                    continue
                # quantile mapping
                q_obs  = np.nanquantile(obs_vals,  np.linspace(0, 1, 100))
                q_era5 = np.nanquantile(era5_vals, np.linspace(0, 1, 100))
                group[var] = np.interp(
                    group[var].values, q_era5, q_obs, left=q_obs[0], right=q_obs[-1]
                )
                log.debug("교정 완료: cid_seq=%s var=%s", cid_seq, var)

            result_frames.append(group)

        corrected = pd.concat(result_frames, ignore_index=True)
        log.info("바이어스 교정 완료: %d 행", len(corrected))
        return corrected

    # ── 수집 + 저장 일괄 처리 ────────────────────────────────────────────────

    def collect_and_save(
        self,
        years: List[int],
        site_coords: pd.DataFrame,    # cid_seq, lat, lon
        output_path: Optional[Path] = None,
        bias_correct: bool = True,
        asos_path: Optional[Path] = None,
        site_grid_path: Optional[Path] = None,
        col_cfg: Optional[CollectConfig] = None,
    ) -> pd.DataFrame:
        """
        연도 목록 전체를 수집 → 단위 변환 → (옵션) 바이어스 교정 → parquet 저장.

        Parameters
        ----------
        years          : 수집할 연도 목록 (예: [2022, 2023, 2024])
        site_coords    : cid_seq, lat, lon 컬럼을 가진 DataFrame
        output_path    : 저장 경로 (None이면 config 기본값)
        bias_correct   : True이면 ASOS 기준 quantile mapping 적용
        asos_path      : kma_obs_asos_hourly.parquet 경로
        site_grid_path : site_to_kma_grid.csv 경로
        """
        cfg = col_cfg or self.cfg
        out_raw = output_path or (cfg.output_dir / "era5_nwp_input_raw.parquet")
        out_corr = cfg.output_dir / "era5_nwp_bias_corrected.parquet"
        out_raw.parent.mkdir(parents=True, exist_ok=True)

        asos_path      = asos_path      or (cfg.output_dir / "kma_obs_asos_hourly.parquet")
        site_grid_path = site_grid_path or (cfg.output_dir / "site_to_kma_grid.csv")

        from datetime import date as _date  # noqa: PLC0415
        from tqdm import tqdm  # noqa: PLC0415

        today = _date.today()
        # (year, month) 전체 목록 생성 (미래 제외, 중복 제거)
        year_months = sorted({
            (y, m)
            for y in set(years)
            for m in range(1, 13)
            if not (y > today.year or (y == today.year and m > today.month))
        })
        log.info("수집 대상: %d개 월 (%s ~ %s)",
                 len(year_months),
                 f"{year_months[0][0]}-{year_months[0][1]:02d}",
                 f"{year_months[-1][0]}-{year_months[-1][1]:02d}")

        # 이미 저장된 월 목록 파악 (증분 이어받기)
        done_months: set[tuple[int, int]] = set()
        if out_raw.exists():
            existing = pd.read_parquet(out_raw, columns=["timestamp"])
            existing["timestamp"] = pd.to_datetime(existing["timestamp"])
            done_months = {
                (ts.year, ts.month)
                for ts in existing["timestamp"].drop_duplicates()
            }
            log.info("기수집 월: %d개 → 건너뜀", len(done_months))

        for year, month in tqdm(year_months, desc="ERA5 월별 수집"):
            if (year, month) in done_months:
                continue

            try:
                nc_path = self.download_month(year, month)
            except Exception as e:
                log.warning("다운로드 실패 %d-%02d: %s", year, month, e)
                continue

            try:
                df = self.nc_to_site_df(nc_path, site_coords)
            except Exception as e:
                log.warning("변환 실패 %d-%02d: %s", year, month, e)
                continue

            if df.empty:
                log.warning("%d-%02d 수집 결과 없음", year, month)
                continue

            df = self.convert_units(df)

            # 월 완료 → 즉시 append 저장
            _append_parquet(df, out_raw, dedup_cols=["timestamp", "cid_seq"])
            log.info("%d-%02d 저장 완료: %d 행", year, month, len(df))

        if not out_raw.exists():
            log.error("수집된 ERA5 데이터가 없습니다.")
            return pd.DataFrame()

        raw = pd.read_parquet(out_raw)
        log.info("ERA5 원본 최종: %s (%d 행)", out_raw, len(raw))

        if bias_correct:
            corrected = self.bias_correct(
                raw,
                asos_path      = asos_path,
                site_grid_path = site_grid_path,
            )
            corrected.to_parquet(out_corr, index=False, engine="pyarrow")
            log.info("바이어스 교정본 저장: %s (%d 행)", out_corr, len(corrected))
            return corrected

        return raw


# ─── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _unzip_if_needed(path: Path) -> Path:
    """
    CDS 신규 API(2024+)는 NetCDF 파일을 ZIP으로 묶어 반환한다.
    ZIP이면 같은 이름의 디렉터리에 압축 해제 후 해당 디렉터리 경로를 반환.
    NC 파일이면 그대로 반환.
    """
    import zipfile  # noqa: PLC0415

    with open(path, "rb") as f:
        magic = f.read(4)

    # ZIP magic bytes: PK\x03\x04
    if magic[:2] != b"PK":
        return path  # 이미 NetCDF

    extract_dir = path.parent / path.stem  # era5_2022_01/
    if extract_dir.exists() and any(extract_dir.glob("*.nc")):
        log.debug("이미 압축 해제됨: %s", extract_dir)
        return extract_dir

    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)
    extracted = sorted(extract_dir.glob("*.nc"))
    log.info("ZIP 압축 해제: %s → %d 파일", extract_dir, len(extracted))
    return extract_dir


def _append_parquet(
    df: pd.DataFrame,
    path: Path,
    dedup_cols: list[str],
) -> None:
    """기존 parquet 에 df 를 append 후 중복 제거하여 저장."""
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
        except Exception as e:
            log.warning("기존 parquet 읽기 실패 (%s) — 덮어씁니다: %s", path.name, e)
            combined = df
        combined.drop_duplicates(subset=dedup_cols).sort_values(dedup_cols).to_parquet(
            path, index=False, engine="pyarrow"
        )
    else:
        df.to_parquet(path, index=False, engine="pyarrow")
