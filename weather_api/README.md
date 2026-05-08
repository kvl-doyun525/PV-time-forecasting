# 한국 기상 데이터 수집 프로그램

한국 기상청 공공데이터 API를 사용하여 기상 예보 데이터를 수집하는 Python 프로그램입니다.

## 주요 기능

- **24시간 예보 데이터**: 온도, 습도, 날씨, 풍속, 풍향, 파고, 강수량, 강수확률, 적설량 등
- **단기예보 지원**: 초단기예보(6시간) 및 단기예보(24시간 이상, 3시간 간격 → 1시간 단위 보간)
- **일출몰 시간**: API 또는 계산 방식으로 일출, 일몰 시간 제공
- **조력 정보**: 조석 시계열 예보 API 사용 (현재 시간 기준 +24시간 미래 예측 데이터)
- **여러 지역 동시 조회**: 여러 교량 위치의 기상 데이터를 한 번에 조회
- **과거 데이터 관리**: 예보 및 센서 실측 데이터 저장 및 조회 기능

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. API 키 발급:
   - [공공데이터포털](https://www.data.go.kr) 회원가입
   - 기상청 단기예보조회서비스 활용신청
   - 마이페이지에서 API 인증키 확인 (승인까지 최소 1일 소요)

3. 설정 파일 수정:
   - `config.py` 파일에서 `WEATHER_SERVICE_KEY`에 발급받은 API 키 입력
   - 또는 환경 변수로 설정:
     ```bash
     export WEATHER_API_KEY='your_api_key_here'
     ```

## 사용 방법

### 기본 사용

```bash
python main.py
```

### 코드에서 직접 사용

```python
from weather_api import WeatherDataCollector
from config import Config

# 데이터 수집기 생성
collector = WeatherDataCollector(
    weather_service_key=Config.WEATHER_SERVICE_KEY,
    sun_moon_service_key=Config.SUN_MOON_SERVICE_KEY
)

# 서울 좌표로 24시간 예보 조회
seoul_lat = 37.5665
seoul_lon = 126.9780

data = collector.get_comprehensive_forecast(
    latitude=seoul_lat,
    longitude=seoul_lon,
    location="서울"
)

# 결과 확인
print(data)
```

## 제공되는 데이터

### 예보 데이터 (24시간)
- **온도** (T1H, TMP): 섭씨 온도
- **최고/최저기온** (TMX, TMN): 일 최고/최저 기온
- **습도** (REH): 상대습도 (%)
- **하늘상태** (SKY): 맑음/구름많음/흐림 (태양광 발전 예측용)
- **강수형태** (PTY): 없음/비/비눈/눈/소나기
- **강수량** (PCP): 1시간 강수량 (mm)
- **강수확률** (POP): 강수 확률 (%)
- **적설량** (SNO): 1시간 적설량 (cm) - 동계 결빙 예측용
- **풍속** (WSD): m/s (풍력 발전 예측용)
- **풍향** (VEC): 각도 (0-360)
- **파고** (WAV): m (해상 교량용)

### 일출몰 정보
- 일출 시간
- 일몰 시간
- 월출 시간
- 월몰 시간
- 월령

### 조력 정보
- **예측조위 (tdlvHgt)**: 예측된 조위 높이 (m)
- **실측조위 (bscTdlvHgt)**: 실제 측정된 조위 높이 (m)
- **관측소명 (obsvtrNm)**: 관측소 이름
- **관측일시 (obsrvnDt)**: 관측 일시
- **위도/경도 (lat/lot)**: 관측소 위치 좌표

## 파일 구조

```
weather_api/
├── weather_api.py          # 메인 API 클래스
├── weather_data_manager.py # 데이터 저장/관리 모듈
├── config.py               # 설정 파일
├── utils.py                # 유틸리티 함수
├── main.py                 # 실행 파일
├── requirements.txt        # 패키지 목록
├── data/                   # 데이터 저장 디렉토리
│   ├── forecast/           # 예보 데이터
│   ├── sensor/             # 센서 실측 데이터
│   └── historical/         # 과거 데이터
└── README.md              # 이 파일
```

## API 참고

### 기상청 단기예보 API
- **초단기예보**: 1~6시간 예보 (00, 03, 06, 09, 12, 15, 18, 21시 발표)
  - URL: `http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst`
- **단기예보**: 24시간 이상 예보 (02, 05, 08, 11, 14, 17, 20, 23시 발표)
  - URL: `http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst`
  - 3시간 간격 데이터를 1시간 단위로 선형 보간하여 제공
- **격자 좌표**: 위경도를 5km × 5km 격자 좌표로 자동 변환
- **참고 문서**: https://www.data.go.kr/data/15084084/openapi.do

### 일출일몰정보 조회서비스
- **API URL**: `http://apis.data.go.kr/B090041/openapi/service/RiseSetInfoService/getAreaRiseSetInfo`
- **참고 문서**: https://www.data.go.kr/data/15012688/openapi.do
- **제공 정보**: 일출, 일몰, 월출, 월몰, 일중, 월중, 시민박명, 항해박명, 천문박명
- **대체 방법**: API 실패 시 자동으로 천문학적 계산 방식 사용

### 조석(조력) 예측 정보 API
- **조석 예보(시계열) API** (기본 사용): 현재 시간 기준 +24시간 미래 예측 데이터 제공
  - **API URL**: `http://apis.data.go.kr/1192136/tideFcstTime/GetTideFcstTimeApiService`
  - **참고 문서**: https://www.data.go.kr/data/15156022/openapi.do
  - **제공 정보**: 예측일시, 조위높이(cm), 예보지점명, 위도/경도
  - **시간 간격**: 1분 ~ 60분 조절 가능 (기본: 60분)
  - **미래 예측**: ✅ 내일/모레 날짜 조회 시 전체 24시간 미래 예측 데이터 제공
- **조석 예측 정보 API** (과거 데이터용)
  - **API URL**: `http://apis.data.go.kr/1192136/surveyTideLevel/GetSurveyTideLevelApiService`
  - **참고 문서**: https://www.data.go.kr/data/15142507/openapi.do
  - **제공 정보**: 예측조위, 실측조위, 관측소 정보
- **관측소 코드**: DT_0001 (인천), DT_0004 (부산) 등

## 추가 기능

### 여러 지역 동시 조회
```python
locations = [
    {"name": "서울", "latitude": 37.5665, "longitude": 126.9780},
    {"name": "부산", "latitude": 35.1796, "longitude": 129.0756}
]

results = collector.get_multiple_locations_forecast(locations)
```

### 과거 데이터 관리
```python
from weather_data_manager import WeatherDataManager

manager = WeatherDataManager()
manager.save_forecast_data(forecast_data, "서울")
historical = manager.load_historical_data("서울", days=7)
```

## 주의사항

1. **API 키 관리**: 
   - 기상청 단기예보 API 키는 2년마다 갱신이 필요합니다.
   - 일출일몰 및 조력 API는 공공데이터포털에서 별도로 발급받아야 합니다.
   - 각 API 키는 `config.py` 파일에 설정하거나 환경 변수로 설정할 수 있습니다.

2. **API 호출 제한**: 
   - 공공데이터포털 API는 일일 호출 제한이 있을 수 있습니다.
   - 과도한 요청은 피하고, 필요시 캐싱을 활용하세요.

3. **관측소 코드**: 
   - 조력 정보는 관측소 코드를 사용합니다 (예: DT_0001 = 인천).
   - 관측소 코드는 공공데이터포털 문서에서 확인할 수 있습니다.

4. **데이터 형식**: 
   - 일출일몰 API는 XML 형식으로 응답합니다.
   - 조력 API는 JSON 형식으로 응답합니다.
   - 모든 응답은 자동으로 파싱됩니다.

## 라이선스

이 프로젝트는 교육 및 개인 사용 목적으로 제공됩니다.
