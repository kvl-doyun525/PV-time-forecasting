"""
기상청 단기예보 / 초단기예보 수집기.

기존 weather_api/weather_api.py 의 WeatherAPI 를 래핑하여:
  1. issue_time + target_time 쌍으로 저장 (leakage 방지)
  2. site_to_kma_grid.csv 의 (fcst_nx, fcst_ny) 격자를 루프 수집
  3. parquet 누적 저장

단기예보 발표 시각 (KST): 02, 05, 08, 11, 14, 17, 20, 23시

내결함성:
    - 5xx / 429 에러: 지수 백오프(2→4→8초)로 최대 3회 재시도
    - site 완료마다 parquet 에 즉시 append 저장 (중단 시 손실 최소화)
    - --incremental 재시작:
        * collect_sites_now  : 기존 parquet 에서 이번 issue_time 이미 수집된 cid_seq 건너뜀
        * collect_range      : 기존 parquet 에서 완료된 issue_time 건너뜀

출력:
    artifacts/dataset_snapshot/kma_fcst_shortterm.parquet   ← 단기예보
    artifacts/dataset_snapshot/kma_fcst_ultrashort.parquet  ← 초단기예보
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from config import WEATHER_API_DIR, ApiConfig, CollectConfig, api_config, collect_config

log = logging.getLogger(__name__)

sys.path.insert(0, str(WEATHER_API_DIR))
from weather_api import WeatherAPI  # noqa: E402

# 단기예보 발표 시각 (KST, 4자리 HHMM)
ISSUE_HOURS_SHORT     = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]
ISSUE_HOURS_ULTRASHORT = [f"{h:02d}00" for h in range(24)]  # 매 시간

# 재시도 설정
_MAX_RETRY      = 3
_RETRY_BASE_SEC = 2.0
_RATELIMIT_WAIT = 60.0

# site 완료마다 저장 (flush) 간격
_FLUSH_EVERY = 20   # site 개수


class KMAForecastCollector:
    """
    기존 WeatherAPI 래퍼.

    collect_sites_now()  : 현재 시점 기준 최신 예보 수집 (실시간 스케줄러용)
    collect_range()      : 날짜 범위 과거 예보 이력 수집 (훈련 데이터용)
    """

    def __init__(self, api_cfg: Optional[ApiConfig] = None):
        self.cfg = api_cfg or api_config
        self.cfg.validate_kma()
        self._api = WeatherAPI(self.cfg.kma_key)

    # ── 단일 API 호출 (재시도 포함) ──────────────────────────────────────────

    def _fetch_with_retry(
        self,
        nx: int, ny: int,
        base_date: str,  # "YYYYMMDD"
        base_time: str,  # "HHMM"
        shortterm: bool,
    ) -> list:
        """WeatherAPI 호출. 5xx / 429 시 지수 백오프 재시도.

        weather_api.py 가 requests.HTTPError 를 잡아서 일반 Exception 으로
        재포장하므로, 메시지 문자열로 상태 코드를 판별한다.
        """
        def _is_429(exc: Exception) -> bool:
            return "429" in str(exc)

        def _is_5xx(exc: Exception) -> bool:
            return any(code in str(exc) for code in ("500", "502", "503", "504"))

        for attempt in range(1, _MAX_RETRY + 2):
            try:
                raw = self._api.get_weather_forecast(
                    nx=nx, ny=ny,
                    base_date=base_date, base_time=base_time,
                    use_short_term=shortterm,
                )
                return raw or []
            except requests.exceptions.HTTPError as e:
                # weather_api.py 가 직접 raise_for_status() 를 노출하는 경우 (방어)
                status = e.response.status_code if e.response is not None else 0
                if status == 429:
                    log.warning("API 호출 제한(429) — %ds 대기 후 재시도 %d/%d",
                                _RATELIMIT_WAIT, attempt, _MAX_RETRY)
                    time.sleep(_RATELIMIT_WAIT)
                    continue
                if attempt <= _MAX_RETRY and status in (500, 502, 503, 504):
                    wait = _RETRY_BASE_SEC * (2 ** (attempt - 1))
                    log.warning("HTTP %d — %ds 후 재시도 %d/%d", status, wait, attempt, _MAX_RETRY)
                    time.sleep(wait)
                else:
                    raise
            except requests.exceptions.Timeout:
                if attempt <= _MAX_RETRY:
                    wait = _RETRY_BASE_SEC * (2 ** (attempt - 1))
                    log.warning("Timeout — %ds 후 재시도 %d/%d", wait, attempt, _MAX_RETRY)
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                # weather_api.py 는 HTTPError 를 Exception("API 요청 실패: ...") 으로
                # 재포장하므로 메시지에서 상태 코드를 판별한다.
                if _is_429(e):
                    log.warning("API 호출 제한(429) — %ds 대기 후 재시도 %d/%d",
                                _RATELIMIT_WAIT, attempt, _MAX_RETRY)
                    time.sleep(_RATELIMIT_WAIT)
                    continue
                if attempt <= _MAX_RETRY and _is_5xx(e):
                    wait = _RETRY_BASE_SEC * (2 ** (attempt - 1))
                    log.warning("5xx 오류 — %ds 후 재시도 %d/%d", wait, attempt, _MAX_RETRY)
                    time.sleep(wait)
                    continue
                raise
        return []

    def _parse_shortterm(
        self, raw: list, cid_seq: int, issue_dt: datetime,
    ) -> pd.DataFrame:
        rows = []
        for item in raw:
            try:
                target_dt = datetime.strptime(item["datetime"], "%Y-%m-%d %H:%M")
            except (KeyError, ValueError):
                continue
            rows.append({
                "cid_seq":     cid_seq,
                "issue_time":  issue_dt,
                "target_time": target_dt,
                "tmp":         item.get("temperature"),
                "reh":         item.get("humidity"),
                "wsd":         item.get("wind_speed"),
                "vec":         item.get("wind_direction"),
                "sky":         item.get("sky_code"),
                "pty":         item.get("precipitation_type_code"),
                "pop":         item.get("precipitation_probability"),
                "pcp":         item.get("precipitation_amount"),
                "sno":         item.get("snowfall"),
            })
        return pd.DataFrame(rows)

    def _parse_ultrashort(
        self, raw: list, cid_seq: int, issue_dt: datetime,
    ) -> pd.DataFrame:
        rows = []
        for item in raw:
            try:
                target_dt = datetime.strptime(item["datetime"], "%Y-%m-%d %H:%M")
            except (KeyError, ValueError):
                continue
            rows.append({
                "cid_seq":     cid_seq,
                "issue_time":  issue_dt,
                "target_time": target_dt,
                "tmp":         item.get("temperature"),
                "reh":         item.get("humidity"),
                "wsd":         item.get("wind_speed"),
                "vec":         item.get("wind_direction"),
                "sky":         item.get("sky_code"),
                "pty":         item.get("precipitation_type_code"),
                "pcp":         item.get("precipitation_amount"),
            })
        return pd.DataFrame(rows)

    # ── 현재 시점 전체 site 수집 (실시간 스케줄러용) ─────────────────────────

    def collect_sites_now(
        self,
        site_grid_map:    pd.DataFrame,
        shortterm:        bool           = True,
        output_path:      Optional[Path] = None,
        request_interval: float          = 0.2,
        incremental:      bool           = False,
        col_cfg:          Optional[CollectConfig] = None,
    ) -> pd.DataFrame:
        """
        현재 발표 기준으로 전체 site 예보 수집 후 parquet 누적 저장.

        incremental=True 이면 이번 issue_time 에 이미 수집된 cid_seq 를 건너뛴다.
        site 완료마다 _FLUSH_EVERY 단위로 parquet 에 즉시 저장한다.

        Parameters
        ----------
        site_grid_map    : site_to_kma_grid.csv 로드 결과 (cid_seq, fcst_nx, fcst_ny)
        shortterm        : True=단기예보, False=초단기예보
        output_path      : 저장 경로 (None이면 config 기본값)
        request_interval : API 요청 간 대기 (초)
        incremental      : 이미 수집된 (issue_time, cid_seq) 건너뜀
        """
        cfg  = col_cfg or collect_config
        now  = datetime.now()
        base_date = now.strftime("%Y%m%d")
        _, base_time = self._api.get_base_time()
        issue_dt  = datetime.strptime(f"{base_date}{base_time}", "%Y%m%d%H%M")

        fname = "kma_fcst_shortterm.parquet" if shortterm else "kma_fcst_ultrashort.parquet"
        out   = output_path or (cfg.output_dir / fname)
        out.parent.mkdir(parents=True, exist_ok=True)

        # 이미 수집된 cid_seq 집합 (증분)
        done_cid: set = set()
        if incremental and out.exists():
            existing = pd.read_parquet(out)
            if not existing.empty and "issue_time" in existing.columns:
                existing["issue_time"] = pd.to_datetime(existing["issue_time"])
                done_cid = set(
                    existing.loc[existing["issue_time"] == issue_dt, "cid_seq"].tolist()
                )
                log.info("증분: issue_time=%s 기수집 cid_seq=%d개 건너뜀", issue_dt, len(done_cid))

        buf: list[pd.DataFrame] = []
        parse_fn = self._parse_shortterm if shortterm else self._parse_ultrashort

        # 격자(nx,ny) 기준 중복 제거 — 같은 격자의 여러 cid_seq 는 한 번만 호출
        unique_grids = (
            site_grid_map[["fcst_nx", "fcst_ny"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # 격자 → cid_seq 목록 매핑
        grid_to_cids: dict[tuple, list[int]] = {}
        for _, row in site_grid_map.iterrows():
            key = (int(row["fcst_nx"]), int(row["fcst_ny"]))
            grid_to_cids.setdefault(key, []).append(int(row["cid_seq"]))

        total_grids = len(unique_grids)
        log.info(
            "격자 중복 제거: 전체 %d 사이트 → %d 고유 격자 (API 호출 %.1f%% 절감)",
            len(site_grid_map), total_grids,
            (1 - total_grids / len(site_grid_map)) * 100,
        )

        with tqdm(total=total_grids, desc="예보 수집(현재)", unit="grid") as pbar:
            for i, (_, row) in enumerate(unique_grids.iterrows()):
                nx, ny = int(row["fcst_nx"]), int(row["fcst_ny"])
                cids   = grid_to_cids[(nx, ny)]
                pbar.set_postfix({"grid": f"({nx},{ny})", "cids": len(cids)})

                # 이 격자의 모든 cid 가 이미 수집됐으면 건너뜀
                if all(c in done_cid for c in cids):
                    pbar.update(1)
                    continue

                try:
                    raw = self._fetch_with_retry(
                        nx=nx, ny=ny,
                        base_date=base_date, base_time=base_time,
                        shortterm=shortterm,
                    )
                    # 같은 격자의 모든 cid_seq 에 동일 데이터 복사
                    for cid in cids:
                        if cid in done_cid:
                            continue
                        df = parse_fn(raw, cid, issue_dt)
                        if not df.empty:
                            buf.append(df)
                except Exception as e:
                    log.warning("예보 수집 실패 grid=(%s,%s): %s", nx, ny, e)

                time.sleep(request_interval)
                pbar.update(1)

                # 중간 저장
                if buf and (i + 1) % _FLUSH_EVERY == 0:
                    _flush(buf, out, dedup_cols=["cid_seq", "issue_time", "target_time"])
                    log.debug("중간 저장: %d개 격자 완료", i + 1)
                    buf.clear()

        if buf:
            _flush(buf, out, dedup_cols=["cid_seq", "issue_time", "target_time"])

        result = pd.read_parquet(out) if out.exists() else pd.DataFrame()
        log.info("예보(현재) 저장 완료: %s  (총 %d 행)", out, len(result))
        return result

    # ── 과거 발표 이력 수집 (훈련 데이터용) ──────────────────────────────────

    def collect_range(
        self,
        site_grid_map:    pd.DataFrame,
        start_date:       Optional[date] = None,
        end_date:         Optional[date] = None,
        shortterm:        bool           = True,
        output_path:      Optional[Path] = None,
        request_interval: float          = 0.3,
        incremental:      bool           = False,
        col_cfg:          Optional[CollectConfig] = None,
    ) -> pd.DataFrame:
        """
        날짜 범위의 모든 발표시각 × 전체 site 조합을 수집한다.

        issue_time 단위로 완료 여부를 체크하여 이어받기를 지원한다.
        각 issue_time 수집 완료 후 parquet 에 즉시 저장한다.

        Parameters
        ----------
        site_grid_map    : site_to_kma_grid.csv
        start_date / end_date : 수집 날짜 범위
        shortterm        : True=단기(8회/일), False=초단기(24회/일)
        request_interval : API 요청 간 대기 (초)
        incremental      : 이미 수집된 issue_time 건너뜀

        주의
        ----
        기상청 단기예보 API 는 최근 약 3일 이전 발표 자료를 지원한다.
        그 이전 과거 자료는 기상자료개방포털(data.kma.go.kr) 파일셋으로 수집해야 한다.
        """
        cfg   = col_cfg or collect_config
        start = start_date or cfg.start_date
        end   = end_date   or cfg.end_date

        fname = "kma_fcst_shortterm.parquet" if shortterm else "kma_fcst_ultrashort.parquet"
        out   = output_path or (cfg.output_dir / fname)
        out.parent.mkdir(parents=True, exist_ok=True)

        issue_times_str = ISSUE_HOURS_SHORT if shortterm else ISSUE_HOURS_ULTRASHORT

        # 이미 완료된 issue_time 집합 (증분)
        done_issue_times: set = set()
        if incremental and out.exists():
            existing = pd.read_parquet(out)
            if not existing.empty and "issue_time" in existing.columns:
                existing["issue_time"] = pd.to_datetime(existing["issue_time"])
                # 해당 issue_time 에 모든 unique cid_seq 가 있으면 완료로 간주
                all_cids = set(site_grid_map["cid_seq"].astype(int).tolist())
                for it, grp in existing.groupby("issue_time"):
                    if all_cids.issubset(set(grp["cid_seq"].tolist())):
                        done_issue_times.add(it)
                log.info("증분: 완료된 issue_time %d개 건너뜀", len(done_issue_times))

        # 전체 issue_time 목록 생성
        # - 현재 시각보다 미래인 issue_time 은 미발표이므로 제외
        # - 현재 시각보다 API_WINDOW_HOURS 이상 오래된 issue_time 은 NO_DATA 반환 → 경고 후 제외
        _API_WINDOW_HOURS = 50   # 실측 기준 안전 마진 (기상청 실제 지원 ~48h)
        now = datetime.now()
        earliest_valid = now - timedelta(hours=_API_WINDOW_HOURS)

        all_issue_times: list[datetime] = []
        cursor = start
        while cursor <= end:
            for hhmm in issue_times_str:
                issue_dt = datetime.strptime(
                    cursor.strftime("%Y%m%d") + hhmm, "%Y%m%d%H%M"
                )
                if issue_dt > now:
                    continue   # 미발표 issue_time 제외
                if issue_dt < earliest_valid:
                    log.warning(
                        "issue_time=%s 는 현재 기준 %dh 이상 이전 — API NO_DATA 가능성 높음, 건너뜀",
                        issue_dt, _API_WINDOW_HOURS,
                    )
                    continue
                all_issue_times.append(issue_dt)
            cursor += timedelta(days=1)

        if not all_issue_times:
            log.warning(
                "수집 가능한 issue_time 이 없습니다. "
                "--start 를 현재 기준 최근 2일 이내로 설정하세요."
            )
            return pd.read_parquet(out) if out.exists() else pd.DataFrame()

        parse_fn = self._parse_shortterm if shortterm else self._parse_ultrashort

        # 격자(nx,ny) 기준 중복 제거
        unique_grids = (
            site_grid_map[["fcst_nx", "fcst_ny"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        grid_to_cids: dict[tuple, list[int]] = {}
        for _, row in site_grid_map.iterrows():
            key = (int(row["fcst_nx"]), int(row["fcst_ny"]))
            grid_to_cids.setdefault(key, []).append(int(row["cid_seq"]))

        log.info(
            "격자 중복 제거: 전체 %d 사이트 → %d 고유 격자 (API 호출 %.1f%% 절감)",
            len(site_grid_map), len(unique_grids),
            (1 - len(unique_grids) / len(site_grid_map)) * 100,
        )

        n_issues = len(all_issue_times)
        n_grids  = len(unique_grids)

        with tqdm(total=n_issues, desc="예보 이력 수집", unit="issue", position=0) as outer:
            for issue_dt in all_issue_times:
                issue_str = issue_dt.strftime("%Y-%m-%d %H:%M")
                outer.set_postfix({"issue": issue_str})

                if issue_dt in done_issue_times:
                    outer.update(1)
                    continue

                base_date = issue_dt.strftime("%Y%m%d")
                base_time = issue_dt.strftime("%H%M")
                buf: list[pd.DataFrame] = []

                with tqdm(
                    total=n_grids,
                    desc=f"  격자 수집 ({issue_str})",
                    unit="grid",
                    position=1,
                    leave=False,
                ) as inner:
                    for _, row in unique_grids.iterrows():
                        nx, ny = int(row["fcst_nx"]), int(row["fcst_ny"])
                        cids   = grid_to_cids[(nx, ny)]
                        inner.set_postfix({"grid": f"({nx},{ny})", "cids": len(cids)})

                        try:
                            raw = self._fetch_with_retry(
                                nx=nx, ny=ny,
                                base_date=base_date, base_time=base_time,
                                shortterm=shortterm,
                            )
                            for cid in cids:
                                df = parse_fn(raw, cid, issue_dt)
                                if not df.empty:
                                    buf.append(df)
                        except Exception as e:
                            log.warning(
                                "예보 이력 수집 실패 issue=%s grid=(%s,%s): %s",
                                issue_dt, nx, ny, e,
                            )
                        time.sleep(request_interval)
                        inner.update(1)

                # issue_time 완료 → 즉시 저장
                if buf:
                    _flush(buf, out, dedup_cols=["cid_seq", "issue_time", "target_time"])
                    log.debug("issue_time=%s 저장 (%d행)", issue_dt, sum(len(d) for d in buf))

                outer.update(1)

        result = pd.read_parquet(out) if out.exists() else pd.DataFrame()
        log.info("예보 이력 저장 완료: %s  (총 %d 행)", out, len(result))
        return result


# ─── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _flush(
    frames: list[pd.DataFrame],
    path: Path,
    dedup_cols: list[str],
) -> None:
    """메모리 버퍼 → parquet append + 중복 제거 저장."""
    if not frames:
        return
    new_df = pd.concat(frames, ignore_index=True)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=dedup_cols, keep="last").to_parquet(
            path, index=False, engine="pyarrow"
        )
    else:
        new_df.to_parquet(path, index=False, engine="pyarrow")
