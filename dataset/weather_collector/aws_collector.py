"""
기상청 AWS(방재기상관측) 시간 관측 이력 수집기.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  AWS 관측 API 사용 전 반드시 확인할 제약사항
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

공공데이터포털 (data.go.kr) AWS API [15057084]:
  - 서비스명  : 기상청_지상(방재, AWS)기상관측자료 조회서비스
  - 엔드포인트: Aws1miInfoService/getAws1miList
  - 제공 자료 : 1분 자료만 제공 (시간 단위 집계 없음)
  - 조회 기간 : 최근 2일 이내만 가능 → 과거 이력 수집 불가
  - 이용 제한 : ※ 공공기관 전용 — 방재기상업무 수행을 위해 공공기관에 한해 제공

과거 AWS 시간 이력 수집 방법:
  ① 기상청 API허브 (apihub.kma.go.kr) — awsh.php (시간통계) 별도 키 필요
  ② 기상자료개방포털 (data.kma.go.kr) 파일셋 수동 다운로드
     https://data.kma.go.kr/data/grnd/selectAwsRltmList.do?pgmNo=56

이 클래스는 기상청 API허브 방식을 사용한다.
  - 엔드포인트: https://apihub.kma.go.kr/api/typ01/url/awsh.php
  - 인증키 : apihub.kma.go.kr 에서 별도 발급 필요 (공공데이터포털 키와 다름)
  - API허브 키를 .env 의 ASOS_API_KEY 에 입력하여 사용

출력: artifacts/dataset_snapshot/kma_obs_aws_hourly.parquet
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from config import ApiConfig, CollectConfig, api_config, collect_config

log = logging.getLogger(__name__)

# 기상청 API허브 AWS 시간통계 엔드포인트
# https://apihub.kma.go.kr/apiList.do (방재기상관측 → 3. AWS 시간통계 자료 조회)
BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/awsh.php"
_PAGE_SIZE = 999


class AWSCollector:
    """
    AWS 방재기상관측 시간통계 수집기.

    기상청 API허브(apihub.kma.go.kr) 의 awsh.php 를 사용한다.
    APIHUB_KEY 가 설정되지 않은 경우 모든 수집 메서드가 빈 DataFrame 을 반환하며
    경고 메시지를 출력한다.
    """

    def __init__(self, api_cfg: Optional[ApiConfig] = None):
        self.cfg = api_cfg or api_config

    @property
    def _key_available(self) -> bool:
        return self.cfg.has_apihub_key

    def _warn_no_key(self) -> None:
        log.warning(
            "APIHUB_KEY 가 설정되지 않아 AWS 수집을 건너뜁니다.\n"
            "  기상청 API허브(https://apihub.kma.go.kr)에서 키를 발급받아\n"
            "  .env 파일의 APIHUB_KEY 에 입력하세요."
        )

    def get_hourly_obs(
        self,
        station_id: str,
        target_dt:  str,   # "YYYYMMDDHH00" — 기준 시각
    ) -> pd.DataFrame:
        """
        AWS 시간통계 1회 API 호출.

        APIHUB_KEY 미설정 시 빈 DataFrame 반환.
        주요 컬럼: tm, stnId, ta(기온℃), rn(강수mm), ws(풍속m/s), wd(풍향°), hm(습도%), pa(기압hPa)
        """
        if not self._key_available:
            self._warn_no_key()
            return pd.DataFrame()

        params = {
            "tm":      target_dt,   # YYYYMMDDHH00
            "stn":     station_id,
            "help":    "0",
            "authKey": self.cfg.apihub_key,
        }
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()

        # API허브는 고정 폭 텍스트 형식으로 반환 — 헤더 행 파싱
        lines = [ln for ln in resp.text.splitlines() if ln.strip() and not ln.startswith("#")]
        if not lines:
            return pd.DataFrame()

        # 헤더 행과 데이터 행 분리
        records = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 10:
                records.append(parts)

        if not records:
            return pd.DataFrame()

        # awsh.php 컬럼 순서 (help=1 로 확인): TM STN TA RN WS WD HM PA
        cols = ["tm", "stnId", "ta", "rn", "ws", "wd", "hm", "pa"]
        rows = []
        for rec in records:
            row = dict(zip(cols, rec[:len(cols)]))
            rows.append(row)

        df = pd.DataFrame(rows)
        df["stnId"] = station_id
        return df

    def collect_range(
        self,
        station_ids:      List[str],
        start_date:       date,
        end_date:         date,
        request_interval: float = 0.5,
        chunk_days:       int   = 30,
    ) -> pd.DataFrame:
        """
        여러 지점, 긴 기간을 시간 단위로 분할 수집.

        APIHUB_KEY 미설정 시 빈 DataFrame 반환.
        """
        if not self._key_available:
            self._warn_no_key()
            return pd.DataFrame()

        all_frames: list[pd.DataFrame] = []
        total_hours = int((end_date - start_date).days * 24 + 25)
        total = len(station_ids) * total_hours

        with tqdm(total=total, desc="AWS 수집", unit="req") as pbar:
            for stn in station_ids:
                cursor_dt = datetime.combine(start_date, datetime.min.time())
                end_dt    = datetime.combine(end_date, datetime.min.time()) + timedelta(hours=23)
                while cursor_dt <= end_dt:
                    tm_str = cursor_dt.strftime("%Y%m%d%H00")
                    pbar.set_postfix({"stn": stn, "tm": tm_str})
                    try:
                        df = self.get_hourly_obs(station_id=stn, target_dt=tm_str)
                        if not df.empty:
                            all_frames.append(df)
                    except Exception as e:
                        log.warning("AWS stn=%s %s 수집 실패: %s", stn, tm_str, e)
                    time.sleep(request_interval)
                    pbar.update(1)
                    cursor_dt += timedelta(hours=1)

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        result = _clean_obs(result)
        return result

    def collect_and_save(
        self,
        station_ids:      List[str],
        start_date:       Optional[date] = None,
        end_date:         Optional[date] = None,
        output_path:      Optional[Path] = None,
        incremental:      bool           = False,
        col_cfg:          Optional[CollectConfig] = None,
    ) -> pd.DataFrame:
        """수집 + parquet 저장. APIHUB_KEY 미설정 시 빈 DataFrame 반환."""
        if not self._key_available:
            self._warn_no_key()
            return pd.DataFrame()

        cfg   = col_cfg or collect_config
        start = start_date or cfg.start_date
        end   = end_date   or cfg.end_date
        out   = output_path or (cfg.output_dir / "kma_obs_aws_hourly.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)

        if incremental and out.exists():
            existing = pd.read_parquet(out)
            if not existing.empty and "tm" in existing.columns:
                last_tm = pd.to_datetime(existing["tm"]).max()
                start   = (last_tm + timedelta(hours=1)).date()
                log.info("증분 수집: %s 이후부터", start)
                if start > end:
                    log.info("이미 최신 데이터입니다.")
                    return existing

        df = self.collect_range(
            station_ids,
            start_date       = start,
            end_date         = end,
            request_interval = cfg.request_interval,
            chunk_days       = cfg.batch_days,
        )

        if df.empty:
            log.warning("수집된 AWS 데이터가 없습니다.")
            return df

        if incremental and out.exists():
            existing = pd.read_parquet(out)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["tm", "stnId"]).sort_values(["stnId", "tm"])

        df.to_parquet(out, index=False, engine="pyarrow")
        log.info("저장 완료: %s  (%d 행)", out, len(df))
        return df


def _clean_obs(df: pd.DataFrame) -> pd.DataFrame:
    if "tm" in df.columns:
        df["tm"] = pd.to_datetime(df["tm"], errors="coerce")
    numeric_cols = ["ta", "rn", "ws", "wd", "hm", "pa", "ps"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
