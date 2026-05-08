"""
기상청 ASOS 시간 관측 이력 수집기.

서비스: 기상청_지상(종관, ASOS) 시간자료 조회서비스
API명(영문): AsosHourlyInfoService
엔드포인트: http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList
공공데이터포털: https://www.data.go.kr/data/15057210/openapi.do
활용가이드: 기상청01_지상(종관,ASOS)시간자료_조회서비스_오픈API활용가이드.docx

수집 지역:
    site_to_kma_grid.csv 의 asos_stn_id 열에서 자동 추출 →
    실제 PV site 와 연관된 ASOS 지점만 수집한다.

내결함성:
    - 5xx / 429 에러: 지수 백오프(1→2→4초)로 최대 3회 재시도
    - 지점 완료마다 parquet 에 즉시 append 저장 (중단 시 손실 최소화)
    - --incremental 재시작: 기존 parquet 에서 지점별 마지막 날짜를 읽어
      아직 수집되지 않은 구간만 이어서 수집

출력: artifacts/dataset_snapshot/kma_obs_asos_hourly.parquet
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

BASE_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
_PAGE_SIZE = 999

# 재시도 설정
_MAX_RETRY      = 3
_RETRY_BASE_SEC = 2.0   # 지수 백오프 기저 (초)
_RATELIMIT_WAIT = 60.0  # 429 수신 시 대기 시간 (초)


class ASOSCollector:
    """
    ASOS 시간 관측 수집기.

    수집 지역은 site_to_kma_grid.csv 의 asos_stn_id 에서 자동 결정되므로
    PV site 와 관련 없는 지점은 수집하지 않는다.
    """

    def __init__(self, api_cfg: Optional[ApiConfig] = None):
        self.cfg = api_cfg or api_config
        self.cfg.validate_asos()

    # ── 단일 API 호출 (재시도 포함) ──────────────────────────────────────────

    def get_hourly_obs(
        self,
        station_id: str,
        start_dt:   str,   # "YYYYMMDDHH"
        end_dt:     str,   # "YYYYMMDDHH"
    ) -> pd.DataFrame:
        """
        ASOS 시간 관측 1회 API 호출 (최대 _MAX_RETRY 재시도).

        Returns
        -------
        DataFrame 원본 컬럼 (활용가이드 응답명세 기준):
            tm(일시), stnId(지점번호), stnNm(지점명),
            ta(기온℃), rn(강수mm), ws(풍속m/s), wd(풍향16방위),
            hm(습도%), pv(증기압hPa), td(이슬점℃),
            pa(현지기압hPa), ps(해면기압hPa),
            icsr(일사MJ/m²), ss(일조hr),
            dsnw(적설cm), dc10Tca(전운량10분위),
            vs(시정10m), ts(지면온도℃)
        """
        params = {
            "serviceKey": self.cfg.asos_key,
            "pageNo":     "1",
            "numOfRows":  str(_PAGE_SIZE),
            "dataType":   "JSON",
            "dataCd":     "ASOS",
            "dateCd":     "HR",
            "startDt":    start_dt[:8],
            "startHh":    start_dt[8:],
            "endDt":      end_dt[:8],
            "endHh":      end_dt[8:],
            "stnIds":     station_id,
        }

        for attempt in range(1, _MAX_RETRY + 2):   # 최초 1회 + 재시도 _MAX_RETRY회
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)

                # 429 Rate Limit
                if resp.status_code == 429:
                    log.warning(
                        "API 호출 제한(429) — %d초 대기 후 재시도 (stn=%s)",
                        _RATELIMIT_WAIT, station_id,
                    )
                    time.sleep(_RATELIMIT_WAIT)
                    continue

                resp.raise_for_status()

                body   = resp.json()
                header = body["response"]["header"]
                if header["resultCode"] != "00":
                    msg = header["resultMsg"]
                    # NO_DATA 는 재시도 불필요 — 즉시 빈 DataFrame 반환
                    if "NO_DATA" in msg or header["resultCode"] == "03":
                        return pd.DataFrame()
                    raise RuntimeError(f"ASOS API 오류: {msg}")

                items = body["response"]["body"]["items"]["item"]
                if not items:
                    return pd.DataFrame()

                df = pd.DataFrame(items)
                df["stnId"] = station_id
                return df

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if attempt <= _MAX_RETRY and status in (500, 502, 503, 504):
                    wait = _RETRY_BASE_SEC * (2 ** (attempt - 1))
                    log.warning(
                        "HTTP %d — %d초 후 재시도 %d/%d (stn=%s)",
                        status, wait, attempt, _MAX_RETRY, station_id,
                    )
                    time.sleep(wait)
                else:
                    raise

            except requests.exceptions.Timeout:
                if attempt <= _MAX_RETRY:
                    wait = _RETRY_BASE_SEC * (2 ** (attempt - 1))
                    log.warning("Timeout — %d초 후 재시도 %d/%d (stn=%s)", wait, attempt, _MAX_RETRY, station_id)
                    time.sleep(wait)
                else:
                    raise

        return pd.DataFrame()   # 재시도 모두 실패 시 빈 반환

    # ── 범위 수집 ─────────────────────────────────────────────────────────────

    def collect_range(
        self,
        station_ids:      List[str],
        start_date:       date,
        end_date:         date,
        request_interval: float = 0.2,
        chunk_days:       int   = 30,
        output_path:      Optional[Path] = None,
        resume_from:      Optional[dict[str, date]] = None,
    ) -> pd.DataFrame:
        """
        여러 지점, 긴 기간을 chunk_days 단위로 분할 수집.

        지점 완료마다 output_path 에 즉시 append 저장한다.
        중단 후 재시작 시 resume_from(지점별 마지막 날짜) 을 넘기면
        이미 수집된 구간을 건너뛴다.

        Parameters
        ----------
        station_ids      : ASOS 지점 번호 목록 (site_to_kma_grid 에서 자동 결정)
        start_date       : 수집 시작일
        end_date         : 수집 종료일
        request_interval : 요청 간 대기 시간 (초)
        chunk_days       : 한 번에 수집할 날짜 범위 (일)
        output_path      : 지점 완료마다 저장할 parquet 경로 (None이면 중간 저장 없음)
        resume_from      : {stnId: last_saved_date} — 이 날짜 이후부터 수집
        """
        resume_from = resume_from or {}
        total_chunks = len(station_ids) * math.ceil(
            (end_date - start_date).days / chunk_days + 1
        )
        all_frames: list[pd.DataFrame] = []

        with tqdm(total=total_chunks, desc="ASOS 수집", unit="req") as pbar:
            for stn in station_ids:
                # 이미 수집된 구간 건너뜀
                stn_start = start_date
                if stn in resume_from:
                    stn_start = resume_from[stn] + timedelta(days=1)
                    if stn_start > end_date:
                        skipped = math.ceil((end_date - start_date).days / chunk_days + 1)
                        pbar.update(skipped)
                        log.info("stn=%s 이미 최신 — 건너뜀", stn)
                        continue

                stn_frames: list[pd.DataFrame] = []
                cursor = stn_start
                while cursor <= end_date:
                    chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_date)
                    pbar.set_postfix({"stn": stn, "날짜": str(cursor)})
                    try:
                        df = self.get_hourly_obs(
                            station_id = stn,
                            start_dt   = cursor.strftime("%Y%m%d") + "00",
                            end_dt     = chunk_end.strftime("%Y%m%d") + "23",
                        )
                        if not df.empty:
                            stn_frames.append(df)
                    except Exception as e:
                        log.warning("ASOS stn=%s %s~%s 수집 실패: %s", stn, cursor, chunk_end, e)

                    time.sleep(request_interval)
                    pbar.update(1)
                    cursor = chunk_end + timedelta(days=1)

                # 지점 완료 → 즉시 저장
                if stn_frames and output_path:
                    stn_df = pd.concat(stn_frames, ignore_index=True)
                    stn_df = _clean_obs(stn_df)
                    _append_parquet(stn_df, output_path)
                    log.info("stn=%s 저장 완료 (%d행) → %s", stn, len(stn_df), output_path)
                    all_frames.append(stn_df)
                elif stn_frames:
                    all_frames.extend(stn_frames)

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        if output_path is None:
            result = _clean_obs(result)
        return result

    # ── 수집 + 저장 일괄 처리 ────────────────────────────────────────────────

    def collect_and_save(
        self,
        station_ids:  List[str],
        start_date:   Optional[date] = None,
        end_date:     Optional[date] = None,
        output_path:  Optional[Path] = None,
        incremental:  bool           = False,
        col_cfg:      Optional[CollectConfig] = None,
    ) -> pd.DataFrame:
        """
        수집 + parquet 저장 일괄 처리.

        incremental=True 이면 기존 parquet 에서 지점별 마지막 날짜를 읽어
        그 이후 구간만 수집한다 (지점 단위 이어받기).
        """
        cfg   = col_cfg or collect_config
        start = start_date or cfg.start_date
        end   = end_date   or cfg.end_date
        out   = output_path or (cfg.output_dir / "kma_obs_asos_hourly.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)

        # 지점별 이어받기: 기존 parquet 에서 지점별 마지막 날짜 계산
        resume_from: dict[str, date] = {}
        if incremental and out.exists():
            existing = pd.read_parquet(out)
            if not existing.empty and "tm" in existing.columns:
                existing["tm"] = pd.to_datetime(existing["tm"])
                last_dates = (
                    existing.groupby("stnId")["tm"]
                    .max()
                    .apply(lambda ts: ts.date())
                    .to_dict()
                )
                resume_from = {str(k): v for k, v in last_dates.items()}
                log.info("증분 수집: %d개 지점의 마지막 날짜 로드", len(resume_from))

        df = self.collect_range(
            station_ids,
            start_date       = start,
            end_date         = end,
            request_interval = cfg.request_interval,
            chunk_days       = cfg.batch_days,
            output_path      = out,      # 지점 완료마다 즉시 저장
            resume_from      = resume_from,
        )

        if df.empty and not out.exists():
            log.warning("수집된 ASOS 데이터가 없습니다.")
            return df

        # 전체 parquet 재로드하여 반환
        if out.exists():
            result = pd.read_parquet(out)
            log.info("최종 저장 완료: %s  (총 %d 행)", out, len(result))
            return result

        return df


# ─── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _clean_obs(df: pd.DataFrame) -> pd.DataFrame:
    """API 원본 응답 후처리 (활용가이드 응답명세 기준)."""
    if "tm" in df.columns:
        df["tm"] = pd.to_datetime(df["tm"], errors="coerce")

    numeric_cols = [
        "ta", "rn", "ws", "wd", "hm",
        "pv", "td",
        "pa", "ps",
        "icsr", "ss",
        "dsnw", "hr3Fhsc",
        "dc10Tca", "dc10LmcsCa",
        "vs", "ts",
        "m005Te", "m01Te", "m02Te", "m03Te",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _append_parquet(df: pd.DataFrame, path: Path) -> None:
    """기존 parquet 에 df 를 append 후 중복 제거하여 저장."""
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["tm", "stnId"]
        ).sort_values(["stnId", "tm"])
        combined.to_parquet(path, index=False, engine="pyarrow")
    else:
        df.to_parquet(path, index=False, engine="pyarrow")
