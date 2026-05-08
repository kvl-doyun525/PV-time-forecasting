"""
사내 DB 수집용 SQL 쿼리 모음.

모든 쿼리는 파라미터 바인딩(%s)을 사용하여 SQL Injection 을 방지한다.
"""

# ─── 발전소·CID 메타데이터 ─────────────────────────────────────────────────────
SQL_PLANT_META = """
SELECT
    c.plant_seq,
    c.modem_seq,
    c.cid_seq,
    c.cid_no,

    -- 인버터 정보
    c.ivt_id,
    c.ivt_manufacturer,
    c.ivt_model,
    c.ivt_capacity / 1000.0          AS ivt_capacity_kw,
    c.ivt_sn,

    -- 모듈 정보
    c.module_type,
    c.module_manufacturer,
    c.module_model,
    c.module_per_capacity            AS module_per_capacity_w,
    c.module_capacity                AS module_capacity_kw,

    -- 설치 정보
    c.grid_connect,
    c.azimuth,
    c.angle,
    c.tracking,

    -- 발전소 정보
    p.plant_nm,
    p.country,
    p.area_1,
    p.area_2,
    p.area_3,
    p.addr,
    p.addr_detail,

    -- 위경도 (varchar → 파이썬에서 float 변환)
    p.latitude,
    p.longitude,

    -- 수집 시작 일시
    p.open_dt

FROM tb_power_cid_pv c
JOIN tb_power_plant   p ON p.plant_seq = c.plant_seq
WHERE c.del_yn = 'N'
  AND p.use_yn  = 'Y'
ORDER BY c.plant_seq, c.cid_seq
"""

# ─── 시간 집계 시계열 (삼상, 기간 범위) ────────────────────────────────────────
SQL_PV_HOURLY = """
SELECT
    h.sum_dt                            AS timestamp,
    h.plant_seq,
    h.modem_seq,
    h.cid_seq,

    -- 지역 (시계열 행에 비정규화 저장)
    h.area_1,
    h.area_2,
    h.area_3,

    -- 발전량 단위 변환 (Wh → kWh)
    h.pow_gen       / 1000.0            AS pow_gen_kwh,
    h.pow_accum_gen / 1000.0            AS pow_accum_gen_kwh,

    -- 출력 단위 변환 (W → kW)
    h.pv_pow  / 1000.0                  AS pv_pow_kw,
    h.cur_pow / 1000.0                  AS cur_pow_kw,

    -- PV 입력 측
    h.pv_volt,
    h.pv_ampe,

    -- 삼상 계통 전압 (V)
    h.line_volt_rs,
    h.line_volt_st,
    h.line_volt_tr,

    -- 삼상 계통 전류 (A)
    h.line_amp_r,
    h.line_amp_s,
    h.line_amp_t,

    -- 기타
    h.factor,
    h.frequency

FROM tb_data_pv_hour h
WHERE h.phase_type = %s
  AND h.sum_dt >= %s
  AND h.sum_dt <  %s
ORDER BY h.cid_seq, h.sum_dt
"""

# ─── 특정 cid 목록만 수집할 때 사용하는 변형 ──────────────────────────────────
# Python 에서 IN 절 자리수를 동적 생성해야 하므로 함수로 제공
def sql_pv_hourly_filtered(cid_count: int) -> str:
    """
    cid_seq 목록으로 필터링하는 쿼리 반환.

    Parameters
    ----------
    cid_count : cid_seq 리스트의 길이

    Returns
    -------
    SQL 문자열. 파라미터 순서: phase_type, start_dt, end_dt, *cid_list
    """
    placeholders = ", ".join(["%s"] * cid_count)
    return f"""
SELECT
    h.sum_dt                            AS timestamp,
    h.plant_seq,
    h.modem_seq,
    h.cid_seq,
    h.area_1,
    h.area_2,
    h.area_3,
    h.pow_gen       / 1000.0            AS pow_gen_kwh,
    h.pow_accum_gen / 1000.0            AS pow_accum_gen_kwh,
    h.pv_pow  / 1000.0                  AS pv_pow_kw,
    h.cur_pow / 1000.0                  AS cur_pow_kw,
    h.pv_volt,
    h.pv_ampe,
    h.line_volt_rs, h.line_volt_st, h.line_volt_tr,
    h.line_amp_r,   h.line_amp_s,   h.line_amp_t,
    h.factor,
    h.frequency
FROM tb_data_pv_hour h
WHERE h.phase_type = %s
  AND h.sum_dt >= %s
  AND h.sum_dt <  %s
  AND h.cid_seq IN ({placeholders})
ORDER BY h.cid_seq, h.sum_dt
"""

# ─── 수집 범위 자동 감지 ──────────────────────────────────────────────────────
SQL_DATE_RANGE = """
SELECT
    MIN(sum_dt) AS min_dt,
    MAX(sum_dt) AS max_dt,
    COUNT(DISTINCT cid_seq) AS cid_count
FROM tb_data_pv_hour
WHERE phase_type = %s
"""

# ─── 활성 CID 목록 ───────────────────────────────────────────────────────────
SQL_ACTIVE_CIDS = """
SELECT DISTINCT
    h.plant_seq,
    h.cid_seq,
    MIN(h.sum_dt) AS first_dt,
    MAX(h.sum_dt) AS last_dt,
    COUNT(*)       AS record_count
FROM tb_data_pv_hour h
WHERE h.phase_type = %s
GROUP BY h.plant_seq, h.cid_seq
ORDER BY h.plant_seq, h.cid_seq
"""
