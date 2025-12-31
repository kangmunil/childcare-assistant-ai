from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, Date, Enum
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import enum

Base = declarative_base()

# --- Enums (데이터 무결성을 위한 열거형 타입) ---
class Gender(enum.Enum):
    MALE = "M"
    FEMALE = "F"

class FeedingType(enum.Enum):
    BREAST_MILK = "breast_milk" # 모유
    FORMULA = "formula"         # 분유
    BABY_FOOD = "baby_food"     # 이유식

class StoolColor(enum.Enum):
    GOLDEN = "golden" # 황금변
    GREEN = "green"   # 녹변
    BLACK = "black"   # 흑변
    RED = "red"       # 혈변 (주의)
    WHITE = "white"   # 회색변 (주의)

# --- 1. Static Data (기준 정보 - Read Only 권장) ---

class GrowthStandard(Base):
    """
    질병관리청 소아청소년 성장도표 (키, 몸무게, 머리둘레, BMI)
    LMS 파라미터 및 백분위수(Percentile) 데이터를 저장
    """
    __tablename__ = 'growth_standards'

    id = Column(Integer, primary_key=True, index=True)
    chart_type = Column(String, nullable=False, index=True) # 'height_for_age', 'weight_for_age', etc.
    gender = Column(Integer, nullable=False, index=True)    # 1: Male, 2: Female
    age_months = Column(Float, nullable=True, index=True)
    height_cm = Column(Float, nullable=True, index=True)    # Used for weight_for_height
    
    # LMS 파라미터
    l = Column(Float, nullable=True)
    m = Column(Float, nullable=True)
    s = Column(Float, nullable=True)
    
    # 주요 백분위수
    p3 = Column(Float, nullable=True)
    p5 = Column(Float, nullable=True)
    p10 = Column(Float, nullable=True)
    p25 = Column(Float, nullable=True)
    p50 = Column(Float, nullable=True)
    p75 = Column(Float, nullable=True)
    p90 = Column(Float, nullable=True)
    p95 = Column(Float, nullable=True)
    p97 = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now)

class VaccineSchedule(Base):
    """
    국가 필수 예방접종 일정 (NIP)
    """
    __tablename__ = 'vaccine_schedules'

    id = Column(Integer, primary_key=True)
    disease_name = Column(String, nullable=False) # 대상 감염병 (예: B형간염)
    vaccine_name = Column(String, nullable=False) # 백신명
    dose_number = Column(Integer, nullable=False) # 접종 차수 (1차, 2차...)
    start_month = Column(Integer, nullable=False) # 권장 접종 시작 월령
    end_month = Column(Integer, nullable=False)   # 권장 접종 종료 월령
    description = Column(Text, nullable=True)     # 설명

class DevelopmentMilestone(Base):
    """
    K-DST 발달 과업 (퀘스트 목록)
    """
    __tablename__ = 'development_milestones'

    id = Column(Integer, primary_key=True)
    min_month = Column(Integer, nullable=False) # 검사 권장 시작 월령
    max_month = Column(Integer, nullable=False) # 검사 권장 종료 월령
    category = Column(String, nullable=False)   # 대근육, 소근육, 인지, 언어 등
    question = Column(String, nullable=False)   # 질문 내용 (예: "뒤집기를 할 수 있나요?")
    importance_level = Column(Integer, default=1) # 중요도 가중치

# --- 2. Dynamic Data (사용자 로그 - Write Heavy) ---

class Baby(Base):
    __tablename__ = 'babies'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    birth_date = Column(Date, nullable=False)
    gender = Column(Enum(Gender), nullable=False)
    
    # Relationships
    logs = relationship("DailyLog", back_populates="baby")
    growth_records = relationship("GrowthRecord", back_populates="baby")

class DailyLog(Base):
    """
    통합 로그 테이블 (수유, 수면, 배변, 체온 등을 하나의 타임라인으로 관리)
    Log Type을 구분하여 관리하거나, 아래처럼 각각 별도 테이블로 구성 후 
    View로 합칠 수도 있습니다. 여기서는 명확한 구조를 위해 별도 테이블 정의를 선호하지만,
    편의상 공통 부모를 두는 개념으로 작성합니다.
    """
    __tablename__ = 'daily_logs'

    id = Column(Integer, primary_key=True, index=True)
    baby_id = Column(Integer, ForeignKey('babies.id'), nullable=False)
    log_type = Column(String, nullable=False, index=True) # feeding, sleep, excretion, temp
    recorded_at = Column(DateTime, default=datetime.now, index=True)
    
    # Common Fields (JSON으로 유연하게 저장하거나, 별도 테이블로 분리)
    # 여기서는 RAG 검색 효율을 위해 주요 데이터를 JSON으로 넣고 인덱싱 전략을 씁니다.
    details = Column(Text, nullable=True) # JSON String: { "amount": 120, "consistency": "soft" }
    memo = Column(Text, nullable=True)

    baby = relationship("Baby", back_populates="logs")

class GrowthRecord(Base):
    """
    사용자가 입력한 신체 계측 기록
    입력 시 GrowthStandard와 비교 로직 트리거
    """
    __tablename__ = 'growth_records'

    id = Column(Integer, primary_key=True)
    baby_id = Column(Integer, ForeignKey('babies.id'), nullable=False)
    measured_date = Column(Date, default=datetime.now)
    
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    head_circ = Column(Float, nullable=True)
    
    # 분석 결과 캐싱 (매번 계산하지 않도록 저장)
    height_percentile = Column(Float, nullable=True)
    weight_percentile = Column(Float, nullable=True)
    
    baby = relationship("Baby", back_populates="growth_records")
