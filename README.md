# 육아 헬퍼 프로그램 (Childcare Assistant AI)

**데이터 기반 스마트 육아 관리 시스템**

> 공공 데이터와 AI 기술을 활용한 과학적이고 체계적인 육아 지원 플랫폼

---

## 목차

1. [프로젝트 소개](#-프로젝트-소개)
2. [주요 기능](#-주요-기능)
3. [기술 스택](#-기술-스택)
4. [시작하기](#-시작하기)
5. [데이터 수집](#-데이터-수집)
6. [사용 예시](#-사용-예시)
7. [프로젝트 구조](#-프로젝트-구조)
8. [참고 자료](#-참고-자료)

---

## 프로젝트 소개

육아 헬퍼 프로그램은 **OpenAI와 딥마인드 출신 AI 리서치 엔지니어** 수준의 기술적 접근으로 개발된 육아 관리 시스템입니다.

### 주요 특징

- **공공 데이터 활용**: 질병관리청, 보건복지부 등 공신력 있는 데이터 기반
- **과학적 성장 분석**: LMS(Lambda-Mu-Sigma) 방법을 통한 정밀한 백분위수 계산
- **실시간 인프라 정보**: 어린이집, 달빛어린이병원, 예방접종 위탁의료기관 정보
- **프라이버시 우선**: Supabase RLS(Row Level Security)를 통한 데이터 보호
- **확장 가능한 아키텍처**: 모듈화된 설계로 기능 확장 용이

### 핵심 가치

> "육아를 **감**이 아닌 **데이터**로"

부모들이 느끼는 불안을 해소하고, 근거 있는 의사결정을 돕는 파트너를 목표로 합니다.

---

## 주요 기능

### 1. 성장 분석 (Growth Analysis)

- **질병관리청 '2017 소아청소년 성장도표'** 기반 정밀 분석
- **Z-Score 및 백분위수** 자동 계산
- 키, 몸무게, 머리둘레 추적 및 시각화

### 2. 일상 로그 관리

- **수유 (Feeding)**: 모유/분유/이유식 기록
- **수면 (Sleep)**: 수면 패턴 분석 및 최적 낮잠 시간 추천
- **배변 (Excretion)**: 색상/형태 기반 건강 상태 체크
- **체온 (Temperature)**: 발열 감지 및 해열제 복용 가이드

### 3. 공공 API 연동

#### [위치] 전국 어린이집 정보 (한국사회보장정보원)
- 실시간 어린이집 위치, 유형, 정원/현원 정보
- 지도 기반 검색 및 거리 계산

#### [병원] 달빛어린이병원 (국립중앙의료원)
- 야간/휴일 진료 가능한 소아 전문 병원
- 현재 시각 기준 운영 중인 병원 필터링

#### [예방접종] 예방접종 위탁의료기관 (질병관리청)
- 국가필수예방접종(NIP) 가능 병원
- 생년월일 기반 접종 일정 자동 계산 및 알림

---

## 기술 스택

### Backend & Data
- **Database**: Supabase (PostgreSQL)
- **ORM**: SQLAlchemy
- **Data Processing**: Pandas, NumPy, SciPy

### API & Web
- **HTTP Client**: Requests, HTTPX
- **공공 API**: 공공데이터포털 오픈 API

### AI & Analytics
- **통계 분석**: SciPy (Z-Score, 백분위수 계산)
- **향후 확장**: LangChain, Hugging Face (RAG 시스템 구축)

---

## 시작하기

### 1. 사전 요구사항

- Python 3.9 이상
- Supabase 계정 ([가입하기](https://supabase.com/))
- 공공데이터포털 API 키 ([신청하기](https://www.data.go.kr/))

### 2. 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/childcare-assistant-ai.git
cd childcare-assistant-ai

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일 편집 (필수 항목 입력)
```

**Supabase 설정 방법:**

1. [Supabase 대시보드](https://app.supabase.com/) 접속
2. 새 프로젝트 생성
3. Settings → API에서 다음 값 복사:
   - **Project URL** → `SUPABASE_URL`
   - **anon public** → `SUPABASE_KEY`
   - **service_role** → `SUPABASE_SERVICE_ROLE_KEY`

### 4. 데이터베이스 초기화

```bash
# 1. Supabase SQL Editor에서 테이블 생성
# scripts/create_tables.sql 실행

# 2. 성장도표 데이터 다운로드
# 공공데이터포털에서 '질병관리청 소아청소년 성장도표' 다운로드
# → data/growth_standards/ 에 저장

# 3. 데이터베이스 초기화
python scripts/init_database.py --all
```

---

## 데이터 수집

### 성장도표 LMS 데이터 로드

```bash
python scripts/init_database.py --growth-chart-only
```

### 공공 API 데이터 수집

```bash
python scripts/init_database.py --public-api-only
```

**수집 데이터:**
- 전국 어린이집 정보
- 달빛어린이병원
- 예방접종 위탁의료기관

---

## 사용 예시

```python
from src.database.supabase_client import get_supabase_client
from src.collectors.growth_chart_parser import GrowthChartParser

# 1. Supabase 연결
client = get_supabase_client()

# 2. 아기 프로필 생성
baby = client.insert_data("babies", {
    "name": "지안",
    "birth_date": "2024-01-15",
    "gender": "M"
})

# 3. 성장 분석
parser = GrowthChartParser()
result = parser.assess_growth(
    value=10.2,
    gender="M",
    age_months=12,
    measure_type="weight"
)

print(result['message'])
# [결과] 백분위수: 65.3% (정상)
```

전체 예제:
```bash
python examples/usage_example.py
```

---

## 프로젝트 구조

```
childcare-assistant-ai/
├── src/
│   ├── database/
│   │   └── supabase_client.py
│   ├── collectors/
│   │   ├── public_api_collector.py
│   │   └── growth_chart_parser.py
│   ├── models/
│   ├── utils/
│   └── analysis/
├── scripts/
│   ├── init_database.py
│   └── create_tables.sql
├── examples/
│   └── usage_example.py
├── docs/
├── data/
└── .env.example
```

---

## 참고 자료

- [공공데이터포털](https://www.data.go.kr/)
- [Supabase 문서](https://supabase.com/docs)
- [질병관리청 성장도표](https://www.data.go.kr/data/15076588/fileData.do)

---

## 문의

질문이나 제안은 이슈로 등록해주세요!

**Made with Love by AI Research Engineers**