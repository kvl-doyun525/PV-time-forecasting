"""
PV 데이터 수집기.

주요 기능:
  - collect_meta()    : 발전소·CID 메타데이터 수집 → plant_meta.parquet
  - collect_hourly()  : 시간 집계 시계열 수집     → pv_raw_hourly.parquet
  - collect_all()     : 메타 + 시계열 일괄 수집

수집 전략:
  - 시계열은 COLLECT_BATCH_DAYS 단위로 분할 fetch (메모리 보호)
  - 증분 수집: 기존 파일 마지막 timestamp 이후만 가져옴
  - 진행 상황은 tqdm 프로그레스바 + logging으로 표시
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from config import CollectConfig, DbConfig, collect_config, db_config
from db import fetch_df, fetch_df_chunked, get_connection
from queries import (
    SQL_ACTIVE_CIDS,
    SQL_DATE_RANGE,
    SQL_PLANT_META,
    SQL_PV_HOURLY,
    sql_pv_hourly_filtered,
)

log = logging.getLogger(__name__)


class PvCollector:
    """
    사내 DB에서 PV 발전 데이터를 수집하고 parquet 으로 저장하는 클래스.

    Parameters
    ----------
    db_cfg      : DB 접속 설정 (None이면 전역 설정 사용)
    col_cfg     : 수집 동작 설정 (None이면 전역 설정 사용)
    output_dir  : 저장 경로 Override (None이면 col_cfg.output_dir 사용)
    """

    def __init__(
        self,
        db_cfg:     Optional[DbConfig]      = None,
        col_cfg:    Optional[CollectConfig] = None,
        output_dir: Optional[Path]          = None,
    ):
        self.db_cfg  = db_cfg  or db_config
        self.col_cfg = col_cfg or collect_config
        self.out_dir = Path(output_dir or self.col_cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        log.info("출력 디렉토리: %s", self.out_dir)

    # ─── 공개 API ───────────────────────────────────────────────────────────

    def collect_meta(self, save: bool = True) -> pd.DataFrame:
        """
        발전소·CID 메타데이터 수집.

        Returns
        -------
        DataFrame: plant_meta 전체
        """
        log.info("발전소·CID 메타데이터 수집 시작 ...")
        df = fetch_df(SQL_PLANT_META, cfg=self.db_cfg)

        df = self._clean_meta(df)
        log.info("메타 수집 완료: %d 건 (cid %d 개)", len(df), df["cid_seq"].nunique())

        if save:
            path = self.out_dir / "plant_meta.parquet"
            df.to_parquet(path, index=False, engine="pyarrow")
            log.info("저장 완료: %s", path)

        return df

    def collect_hourly(
        self,
        start_date: Optional[date] = None,
        end_date:   Optional[date] = None,
        cid_list:   Optional[List[int]] = None,
        incremental: bool = False,
        save:        bool = True,
    ) -> pd.DataFrame:
        """
        시간 집계 시계열 수집.

        Parameters
        ----------
        start_date  : 수집 시작일. None이면 config 값 사용
        end_date    : 수집 종료일. None이면 config 값 사용
        cid_list    : 수집할 cid_seq 목록. None이면 전체
        incremental : True이면 기존 parquet 의 마지막 timestamp 이후만 수집
        save        : True이면 결과를 parquet 으로 저장

        Returns
        -------
        DataFrame: 수집된 전체 시계열
        """
        start = start_date or self.col_cfg.start_date
        end   = end_date   or self.col_cfg.end_date

        out_path = self.out_dir / "pv_raw_hourly.parquet"

        if incremental and out_path.exists():
            existing_df = pd.read_parquet(out_path)
            last_ts = pd.to_datetime(existing_df["timestamp"]).max()
            start   = (last_ts + timedelta(hours=1)).date()
            log.info("증분 수집: 마지막 저장 timestamp=%s 이후부터 수집", last_ts)
            if start > end:
                log.info("이미 최신 데이터입니다. 수집 건너뜀.")
                return existing_df

        log.info(
            "시계열 수집: %s ~ %s  cid_list=%s",
            start, end,
            f"{len(cid_list)}개" if cid_list else "전체",
        )

        chunks = self._date_batches(start, end, self.col_cfg.batch_days)
        all_frames: list[pd.DataFrame] = []

        with tqdm(total=len(chunks), desc="배치 수집", unit="batch") as pbar:
            for batch_start, batch_end in chunks:
                df_batch = self._fetch_batch(batch_start, batch_end, cid_list)
                if not df_batch.empty:
                    all_frames.append(df_batch)
                pbar.set_postfix({
                    "기간": f"{batch_start}~{batch_end}",
                    "건수": f"{len(df_batch):,}",
                })
                pbar.update(1)

        if not all_frames:
            log.warning("수집된 데이터가 없습니다.")
            result = pd.DataFrame()
        else:
            result = pd.concat(all_frames, ignore_index=True)

        log.info(
            "시계열 수집 완료: %d 건 / cid %d 개 / 기간 %s ~ %s",
            len(result),
            result["cid_seq"].nunique() if not result.empty else 0,
            start, end,
        )

        if save and not result.empty:
            if incremental and out_path.exists():
                existing_df = pd.read_parquet(out_path)
                result = pd.concat([existing_df, result], ignore_index=True)
                result = result.drop_duplicates(
                    subset=["timestamp", "plant_seq", "modem_seq", "cid_seq"]
                ).sort_values(["cid_seq", "timestamp"])

            result.to_parquet(out_path, index=False, engine="pyarrow")
            log.info("저장 완료: %s  (총 %d 건)", out_path, len(result))

        return result

    def collect_all(
        self,
        start_date: Optional[date] = None,
        end_date:   Optional[date] = None,
        cid_list:   Optional[List[int]] = None,
        incremental: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """메타 + 시계열 일괄 수집."""
        meta    = self.collect_meta(save=True)
        hourly  = self.collect_hourly(start_date, end_date, cid_list, incremental, save=True)
        return {"meta": meta, "hourly": hourly}

    def show_db_summary(self) -> None:
        """DB 에 저장된 데이터 범위와 CID 수를 출력한다."""
        log.info("DB 데이터 범위 조회 중 ...")
        df_range = fetch_df(SQL_DATE_RANGE, params=(self.col_cfg.phase_type,), cfg=self.db_cfg)
        df_cids  = fetch_df(SQL_ACTIVE_CIDS, params=(self.col_cfg.phase_type,), cfg=self.db_cfg)

        row = df_range.iloc[0]
        print("\n── DB 데이터 요약 ────────────────────────────────")
        print(f"  phase_type 필터 : {self.col_cfg.phase_type} (삼상)")
        print(f"  데이터 기간     : {row['min_dt']}  ~  {row['max_dt']}")
        print(f"  고유 CID 수     : {int(row['cid_count']):,} 개")
        print(f"\n  CID별 레코드 수 (상위 10개):")
        print(df_cids.head(10).to_string(index=False))
        print("──────────────────────────────────────────────\n")

    # ─── 내부 헬퍼 ──────────────────────────────────────────────────────────

    def _fetch_batch(
        self,
        start: date,
        end:   date,
        cid_list: Optional[List[int]],
    ) -> pd.DataFrame:
        """단일 배치(날짜 범위)를 fetch."""
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt   = datetime.combine(end + timedelta(days=1), datetime.min.time())

        if cid_list:
            sql    = sql_pv_hourly_filtered(len(cid_list))
            params = (self.col_cfg.phase_type, start_dt, end_dt, *cid_list)
        else:
            sql    = SQL_PV_HOURLY
            params = (self.col_cfg.phase_type, start_dt, end_dt)

        frames: list[pd.DataFrame] = []
        for chunk in fetch_df_chunked(sql, params=params, chunksize=100_000, cfg=self.db_cfg):
            frames.append(chunk)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    @staticmethod
    def _date_batches(
        start: date, end: date, batch_days: int
    ) -> list[tuple[date, date]]:
        """start~end 범위를 batch_days 단위로 분할."""
        batches: list[tuple[date, date]] = []
        cur = start
        while cur <= end:
            batch_end = min(cur + timedelta(days=batch_days - 1), end)
            batches.append((cur, batch_end))
            cur = batch_end + timedelta(days=1)
        return batches

    @staticmethod
    def _clean_meta(df: pd.DataFrame) -> pd.DataFrame:
        """메타 DataFrame 후처리: 위경도 float 변환, 결측값 정리."""
        for col in ("latitude", "longitude"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 수치형 컬럼 타입 정리
        for col in ("ivt_capacity_kw", "module_capacity_kw", "module_per_capacity_w"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
