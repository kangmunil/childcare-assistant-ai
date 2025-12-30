"""
FastAPI 챗봇 API

육아 헬퍼 AI 에이전트를 위한 RESTful API 엔드포인트
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.childcare_agent import ChildcareAgent

# FastAPI 앱 초기화
app = FastAPI(
    title="육아 헬퍼 AI API",
    description="LLM 기반 에이전틱 RAG 챗봇 API",
    version="1.0.0"
)

# CORS 설정 (React 프론트엔드와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Request/Response 모델
# ========================================

class ChatRequest(BaseModel):
    """챗봇 요청 모델"""
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID (대화 히스토리 유지)")
    user_id: Optional[str] = Field(None, description="사용자 ID")


class ChatResponse(BaseModel):
    """챗봇 응답 모델"""
    reply: str = Field(..., description="AI 응답 메시지")
    session_id: str = Field(..., description="세션 ID")
    timestamp: str = Field(..., description="응답 시각")


class HealthCheckResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    message: str
    timestamp: str


# ========================================
# 세션 관리 (간단한 인메모리 저장소)
# ========================================

# 실제 프로덕션에서는 Redis 등을 사용
chat_sessions: Dict[str, List[Dict[str, str]]] = {}


def get_or_create_session(session_id: Optional[str]) -> str:
    """
    세션 ID를 가져오거나 새로 생성합니다.

    Args:
        session_id: 기존 세션 ID (없으면 None)

    Returns:
        세션 ID
    """
    if session_id and session_id in chat_sessions:
        return session_id

    # 새 세션 생성
    new_session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    chat_sessions[new_session_id] = []
    logger.info(f"새 세션 생성: {new_session_id}")

    return new_session_id


def add_to_session(session_id: str, role: str, content: str):
    """
    세션에 메시지 추가

    Args:
        session_id: 세션 ID
        role: 역할 (user 또는 assistant)
        content: 메시지 내용
    """
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_sessions[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """
    세션 히스토리 가져오기

    Args:
        session_id: 세션 ID

    Returns:
        메시지 히스토리
    """
    return chat_sessions.get(session_id, [])


# ========================================
# 에이전트 초기화 (싱글톤)
# ========================================

_agent_instance: Optional[ChildcareAgent] = None


def get_agent() -> ChildcareAgent:
    """
    ChildcareAgent 싱글톤 인스턴스를 반환합니다.

    Returns:
        ChildcareAgent 인스턴스
    """
    global _agent_instance

    if _agent_instance is None:
        logger.info("ChildcareAgent 초기화 중...")
        _agent_instance = ChildcareAgent()
        logger.success("ChildcareAgent 초기화 완료")

    return _agent_instance


# ========================================
# API 엔드포인트
# ========================================

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """
    루트 엔드포인트 (헬스 체크)
    """
    return HealthCheckResponse(
        status="healthy",
        message="육아 헬퍼 AI API가 정상 작동 중입니다.",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    헬스 체크 엔드포인트
    """
    return HealthCheckResponse(
        status="healthy",
        message="API 서버 정상",
        timestamp=datetime.now().isoformat()
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: ChildcareAgent = Depends(get_agent)
):
    """
    챗봇 대화 엔드포인트

    Args:
        request: 챗봇 요청 (메시지, 세션 ID)
        agent: ChildcareAgent 인스턴스 (의존성 주입)

    Returns:
        ChatResponse: AI 응답
    """
    try:
        # 세션 ID 가져오기/생성
        session_id = get_or_create_session(request.session_id)

        # 사용자 메시지 기록
        add_to_session(session_id, "user", request.message)

        logger.info(f"[세션: {session_id}] 사용자: {request.message}")

        # 세션 히스토리 가져오기 (최근 10개만)
        history = get_session_history(session_id)[-10:]

        # 에이전트 호출
        reply = agent.chat(
            user_input=request.message,
            chat_history=history
        )

        # AI 응답 기록
        add_to_session(session_id, "assistant", reply)

        logger.info(f"[세션: {session_id}] AI: {reply[:100]}...")

        return ChatResponse(
            reply=reply,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"챗봇 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """
    특정 세션의 대화 히스토리 조회

    Args:
        session_id: 세션 ID

    Returns:
        대화 히스토리
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    return {
        "session_id": session_id,
        "history": chat_sessions[session_id]
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    세션 삭제

    Args:
        session_id: 세션 ID

    Returns:
        삭제 결과
    """
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"세션 {session_id}가 삭제되었습니다."}
    else:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")


# ========================================
# 서버 실행
# ========================================

if __name__ == "__main__":
    import uvicorn

    # 서버 실행
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
