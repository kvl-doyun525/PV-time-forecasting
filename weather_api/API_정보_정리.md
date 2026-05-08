# 일출몰 정보 및 조력 정보 API 출처 정리

## 1. 일출몰 정보 API

### ❌ 기상청 API에서는 제공하지 않음

**기상청 공공데이터 API**는 일출몰 정보를 직접 제공하지 않습니다. 기상청이 제공하는 데이터는 다음과 같습니다:
- 기온, 습도, 강수량, 풍속, 풍향
- 하늘상태, 강수형태
- 기상특보, 생활기상지수

### ✅ 공공데이터포털의 별도 서비스

일출몰 정보는 **공공데이터포털(data.go.kr)**에서 별도 서비스로 제공됩니다:

**서비스명**: 일출일몰정보 조회서비스  
**API URL**: `http://apis.data.go.kr/B090041/openapi/service/RiseSetInfoService/getAreaRiseSetInfo`

**특징**:
- 기상청 API와는 별개의 서비스
- 동일한 공공데이터포털 API 키 사용 가능
- 지역명 기반 조회 (예: "서울", "부산")
- 일출, 일몰, 월출, 월몰 시간 제공

**현재 코드 상태**:
- `SunMoonAPI` 클래스에서 이미 구현되어 있음
- API 실패 시 자동으로 계산 방식으로 대체 (`calculate_sunrise_sunset` 메서드)

### 📝 사용 방법

```python
from weather_api import SunMoonAPI

# API 방식
sun_moon_api = SunMoonAPI(service_key)
info = sun_moon_api.get_sun_moon_info("서울")

# 계산 방식 (API 실패 시 또는 API 키 없을 때)
info = sun_moon_api.calculate_sunrise_sunset(latitude=37.5665, longitude=126.9780)
```

---

## 2. 조력(조수) 정보 API

### ❌ 기상청 API에서는 제공하지 않음

조력 정보는 기상 현상이 아니라 **해양 물리 현상**이므로 기상청에서 제공하지 않습니다.

### ✅ 공공데이터포털 조석 API

조력 정보는 **해양수산부 국립해양조사원**에서 제공하며, 공공데이터포털을 통해 접근할 수 있습니다.

#### 조석 예보(시계열) API (기본 사용) ⭐
- **서비스명**: 조석 예보(시계열) 조회서비스
- **API URL**: `http://apis.data.go.kr/1192136/tideFcstTime/GetTideFcstTimeApiService`
- **참고 문서**: https://www.data.go.kr/data/15156022/openapi.do
- **특징**:
  - ✅ 현재 시간 기준 +24시간 미래 예측 데이터 제공
  - ✅ 시간 간격 조절 가능 (1분 ~ 60분)
  - ✅ 관측소 코드 기반 조회 (예: "DT_0001" - 인천)
  - ✅ 예측일시, 조위높이(cm), 예보지점 정보 제공

#### 조석 예측 정보 API (과거 데이터용)
- **서비스명**: 조석 예측 정보 조회서비스
- **API URL**: `http://apis.data.go.kr/1192136/surveyTideLevel/GetSurveyTideLevelApiService`
- **참고 문서**: https://www.data.go.kr/data/15142507/openapi.do
- **특징**:
  - 과거 데이터만 제공 (현재 시간 이전)
  - 예측조위, 실측조위 제공

**주요 관측소 코드 예시**:
- DT_0001: 인천
- DT_0002: 목포
- DT_0003: 여수
- DT_0004: 부산
- DT_0005: 포항
- DT_0006: 울산
- DT_0007: 속초
- DT_0008: 묵호
- DT_0009: 후포
- DT_0010: 통영
- DT_0011: 거제도
- DT_0012: 완도
- DT_0013: 제주
- DT_0014: 서귀포

**현재 코드 상태**:
- `TideForecastTimeAPI` 클래스: 조석 예보(시계열) API 사용 (기본)
- `TideAPI` 클래스: 조석 예측 정보 API 사용 (과거 데이터용)
- `TideForecastAPI` 클래스: 조석 예보(고, 저조) API (별도 API 키 필요)

### 📝 사용 방법

#### 조석 예보(시계열) API (기본 사용)
```python
from weather_api import TideForecastTimeAPI

time_api = TideForecastTimeAPI(service_key="your_api_key")
# 현재 시간 기준 +24시간 미래 예측 데이터
tide_data = time_api.get_tide_forecast_time(
    obs_code="DT_0001",
    req_date="20260130",
    time_interval=60  # 1시간 간격
)
```

#### 조석 예측 정보 API (과거 데이터)
```python
from weather_api import TideAPI

tide_api = TideAPI(service_key="your_api_key")
tide_info = tide_api.get_tide_info("DT_0001", date="20260130")
```

### 🔍 관측소 코드 확인 방법

1. **국립해양조사원 웹사이트**: https://www.khoa.go.kr
2. **공공데이터포털**: 해양수산부 조석 예측 정보 서비스 검색
3. **API 문서**: 각 관측소별 코드 확인 가능

---

## 3. 대체 방법

### 일출몰 정보

1. **Python 라이브러리 사용** (권장)
   - `ephem`: 천문 계산 라이브러리
   - `skyfield`: 더 정확한 천문 계산
   - `astral`: 일출몰 시간 계산

2. **현재 구현된 계산 방식**
   - `calculate_sunrise_sunset` 메서드로 기본 계산 제공
   - 정확도는 라이브러리보다 낮지만 API 없이도 사용 가능

### 조력 정보

1. **과거 데이터 활용**
   - 조력은 주기적 패턴을 가지므로 과거 데이터로 예측 가능
   - 각 지역별 조력정보 일괄 적용 고려 (정리.txt 참고)

2. **기본값 처리**
   - 조력 정보 없을 경우 Default 값 처리
   - 과거 데이터만으로 예측 진행

---

## 4. API 키 발급 방법

### 일출몰 정보 API
1. [공공데이터포털](https://www.data.go.kr) 접속
2. "일출일몰정보 조회서비스" 검색
3. 활용신청 및 승인 대기 (1일 소요)
4. 마이페이지에서 API 키 확인

### 조력 정보 API
1. [국립해양조사원](https://www.khoa.go.kr) 또는 공공데이터포털 접속
2. "조석 예측 정보" 또는 "해양수산부 조력 정보" 검색
3. 활용신청 (API 키는 선택사항일 수 있음)

---

## 5. 권장 사항

### 일출몰 정보
- ✅ **현재 구현**: API 방식 + 계산 방식 자동 대체 (권장)
- 대안: `ephem` 라이브러리 추가 설치로 정확도 향상 가능

### 조력 정보
- ✅ **현재 구현**: 국립해양조사원 API 사용
- 대안: 조력 정보 없을 경우 과거 데이터 패턴 활용
- 각 지역별 관측소 코드 매핑 테이블 구축 권장

---

## 6. 현재 코드의 장점

1. **일출몰 정보**: API 실패 시 자동으로 계산 방식 사용
2. **조력 정보**: API 키 없어도 일부 데이터 조회 가능
3. **에러 처리**: 각 API 실패 시 적절한 에러 메시지 제공
4. **유연성**: API 키 없이도 기본 기능 사용 가능
