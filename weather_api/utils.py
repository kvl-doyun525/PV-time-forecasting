"""
유틸리티 함수 모듈
"""
from datetime import datetime
from typing import Dict, List
import json


def format_datetime(dt_str: str) -> str:
    """
    날짜시간 문자열 포맷팅
    
    Args:
        dt_str: YYYYMMDDHHMM 형식의 문자열
        
    Returns:
        YYYY-MM-DD HH:MM 형식의 문자열
    """
    try:
        dt = datetime.strptime(dt_str, "%Y%m%d%H%M")
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return dt_str


def format_time(time_str: str) -> str:
    """
    시간 문자열 포맷팅 (HHMM -> HH:MM)
    
    Args:
        time_str: HHMM 형식의 문자열
        
    Returns:
        HH:MM 형식의 문자열
    """
    if len(time_str) == 4:
        return f"{time_str[:2]}:{time_str[2:]}"
    return time_str


def save_to_json(data: Dict, filename: str):
    """
    데이터를 JSON 파일로 저장
    
    Args:
        data: 저장할 데이터
        filename: 파일명
    """
    import os
    
    # data 디렉토리가 없으면 생성
    if not os.path.exists("data"):
        os.makedirs("data")
    
    filepath = os.path.join("data", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"데이터가 저장되었습니다: {filepath}")


def load_from_json(filename: str) -> Dict:
    """
    JSON 파일에서 데이터 로드
    
    Args:
        filename: 파일명
        
    Returns:
        로드된 데이터
    """
    import os
    
    filepath = os.path.join("data", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def print_forecast_summary(forecast_data: Dict):
    """
    예보 데이터 요약 출력
    
    Args:
        forecast_data: 예보 데이터 딕셔너리
    """
    print("\n" + "="*60)
    print(f"위치: {forecast_data['location']['name']}")
    print(f"좌표: ({forecast_data['location']['latitude']}, {forecast_data['location']['longitude']})")
    print("="*60)
    
    # 일출몰 정보
    if forecast_data.get("sun_moon_info"):
        sm = forecast_data["sun_moon_info"]
        print(f"\n일출몰 정보 ({sm['date']}):")
        print(f"  일출: {format_time(sm.get('sunrise', ''))}")
        print(f"  일몰: {format_time(sm.get('sunset', ''))}")
        print(f"  월출: {format_time(sm.get('moonrise', ''))}")
        print(f"  월몰: {format_time(sm.get('moonset', ''))}")
    
    # 조력 정보
    if forecast_data.get("tide_info"):
        tide_info = forecast_data["tide_info"]
        print(f"\n조력 정보 (조석 시계열 - 실행 시간 기준 24시간):")
        print(f"  총 {len(tide_info)}개 데이터")
        for tide in tide_info[:10]:  # 처음 10개 출력
            tide_level = tide.get('tide_level', 0)
            obs_name = tide.get('obs_name', 'N/A')
            pred_time = tide.get('prediction_time', 'N/A')
            # 조위는 cm 단위이므로 m로 변환
            tide_level_m = tide_level / 100.0 if tide_level else 0
            print(f"  {pred_time}: 예측조위 {tide_level_m:.2f}m ({tide_level:.1f}cm) - {obs_name}")
    
    # 24시간 예보
    print(f"\n24시간 예보:")
    print("-"*60)
    for fcst in forecast_data["forecast_24h"][:12]:  # 처음 12시간만 출력
        print(f"\n{fcst['datetime']}:")
        if "temperature" in fcst:
            print(f"  온도: {fcst['temperature']}°C")
        if "humidity" in fcst:
            print(f"  습도: {fcst['humidity']}%")
        if "sky" in fcst:
            print(f"  하늘: {fcst['sky']}")
        if "precipitation_type" in fcst:
            print(f"  강수: {fcst['precipitation_type']}")
        if "wind_speed" in fcst:
            print(f"  풍속: {fcst['wind_speed']} m/s")
        if "wind_direction" in fcst:
            print(f"  풍향: {fcst['wind_direction']}°")
        if "wave_height" in fcst:
            print(f"  파고: {fcst['wave_height']} m")
    
    print("\n" + "="*60)


def get_wind_direction_name(degree: float) -> str:
    """
    풍향 각도를 방향명으로 변환
    
    Args:
        degree: 풍향 각도 (0-360)
        
    Returns:
        방향명 (예: "북", "북동", "동" 등)
    """
    directions = [
        "북", "북북동", "북동", "동북동", "동", "동남동", "남동", "남남동",
        "남", "남남서", "남서", "서남서", "서", "서북서", "북서", "북북서"
    ]
    
    index = int((degree + 11.25) / 22.5) % 16
    return directions[index]
