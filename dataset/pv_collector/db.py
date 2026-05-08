"""
DB 연결 관리.

컨텍스트 매니저로 커넥션을 열고/닫으며,
pandas DataFrame 직접 반환 헬퍼를 제공한다.

pd.read_sql 은 pymysql raw 커넥션에서 컬럼명을 값으로 반환하는
버전 호환 문제가 있으므로, 커서 직접 방식으로 구현한다.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Iterator

import pandas as pd
import pymysql
import pymysql.cursors
from pymysql.connections import Connection

from config import DbConfig, db_config

log = logging.getLogger(__name__)


@contextlib.contextmanager
def get_connection(cfg: DbConfig | None = None) -> Iterator[Connection]:
    """pymysql 커넥션을 컨텍스트 매니저로 반환."""
    cfg = cfg or db_config
    cfg.validate()

    conn = pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        database=cfg.database,
        charset=cfg.charset,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=3600,
        autocommit=True,
    )
    log.debug("DB 연결: %s@%s:%s/%s", cfg.user, cfg.host, cfg.port, cfg.database)
    try:
        yield conn
    finally:
        conn.close()
        log.debug("DB 연결 종료")


def fetch_df(sql: str, params: tuple | None = None, cfg: DbConfig | None = None) -> pd.DataFrame:
    """
    단일 쿼리를 실행하고 DataFrame으로 반환.

    Parameters
    ----------
    sql    : 실행할 SQL 문
    params : 바인딩 파라미터 (pymysql %s 스타일)
    cfg    : DB 설정 (None이면 전역 설정 사용)
    """
    with get_connection(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_df_chunked(
    sql: str,
    params: tuple | None = None,
    chunksize: int = 50_000,
    cfg: DbConfig | None = None,
) -> Iterator[pd.DataFrame]:
    """
    대용량 쿼리를 chunksize 행씩 스트리밍으로 반환하는 제너레이터.

    for chunk in fetch_df_chunked(sql, params):
        process(chunk)
    """
    with get_connection(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            while True:
                rows = cur.fetchmany(chunksize)
                if not rows:
                    break
                yield pd.DataFrame(rows)


def test_connection(cfg: DbConfig | None = None) -> bool:
    """접속 가능 여부를 확인하고 서버 버전을 출력한다."""
    try:
        with get_connection(cfg) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT VERSION() AS ver")
                row = cur.fetchone()
                print(f"DB 연결 성공 — 서버 버전: {row['ver']}")
        return True
    except Exception as exc:
        print(f"DB 연결 실패: {exc}")
        return False
