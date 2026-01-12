from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.analysis.growth_analyzer import GrowthAnalyzer
from src.models.domain import Gender
from datetime import date
import os
from typing import Optional

app = FastAPI(title="Childcare Assistant AI API")

# DB 세션 의존성 (실제 구현 시 connection.py에서 가져옴)
def get_db():
    # pass # 실제 DB 엔진 연결 필요
    yield None 

class GrowthRequest(BaseModel):
    baby_id: int
    gender: str # 'M' or 'F'
    birth_date: date
    height: Optional[float] = None
    weight: Optional[float] = None

@app.get("/")
def read_root():
    return {"message": "Childcare Assistant AI Server is running"}

@app.post("/analyze/growth")
def analyze_baby_growth(req: GrowthRequest, db: Session = Depends(get_db)):
    """
    아이의 성장을 분석하는 API 엔드포인트
    """
    gender_enum = Gender.MALE if req.gender == 'M' else Gender.FEMALE
    analyzer = GrowthAnalyzer(db)
    
    result = analyzer.assess_growth(
        gender=gender_enum,
        birth_date=req.birth_date,
        height=req.height,
        weight=req.weight
    )
    return result

# --- LangChain Agent가 사용할 Service 로직 ---
class AIService:
    """
    LLM이 호출할 도구들의 집합
    """
    @staticmethod
    def get_vaccine_info(disease: str):
        # DB에서 백신 정보 조회 로직
        return f"{disease}에 대한 접종 일정은... (DB 조회 결과)"

    @staticmethod
    def get_expert_advice(query: str):
        # Vector DB(RAG)에서 육아 지식 검색
        return "육아 백과사전에 따르면... (Vector DB 조회 결과)"
