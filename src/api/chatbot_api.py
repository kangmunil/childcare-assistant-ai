"""
FastAPI 챗봇 API

육아 헬퍼 AI 에이전트를 위한 RESTful API 엔드포인트
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Iterable
from datetime import datetime, date
from loguru import logger
import os
import sys
import time
import re
from uuid import uuid4
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.childcare_agent import ChildcareAgent
from src.database.chat_session_manager import session_manager
from src.database.supabase_client import get_supabase_client

# FastAPI 앱 초기화
app = FastAPI(
    title="육아 헬퍼 AI API",
    description="LLM 기반 에이전틱 RAG 챗봇 API",
    version="1.0.0"
)

APP_ENV = os.getenv("APP_ENV", os.getenv("ENV", "production")).lower()
IS_DEVELOPMENT = APP_ENV in {"development", "dev", "local"}
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173" if IS_DEVELOPMENT else ""
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("AI_ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]
ENABLE_SESSION_API = os.getenv(
    "ENABLE_SESSION_API",
    "true" if IS_DEVELOPMENT else "false"
).lower() == "true"
AI_CHAT_META_ENABLED = os.getenv("AI_CHAT_META_ENABLED", "true").lower() == "true"
INTERNAL_SERVICE_TOKEN_HEADER = "X-Internal-Service-Token"
INTERNAL_TOKEN_HEADER = "X-Internal-Token"
AI_INTERNAL_TOKEN = (os.getenv("AI_INTERNAL_TOKEN") or "").strip()
AI_REQUIRE_INTERNAL_AUTH = os.getenv("AI_REQUIRE_INTERNAL_AUTH", "false").lower() == "true"

# CORS 설정 (React 프론트엔드와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Request-Id",
        INTERNAL_SERVICE_TOKEN_HEADER,
        INTERNAL_TOKEN_HEADER,
    ],
)

# ========================================
# Request/Response 모델
# ========================================


class ChatRequest(BaseModel):
    """챗봇 요청 모델"""
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID (대화 히스토리 유지)")
    child_id: Optional[int] = Field(None, description="자녀 ID")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    context_mode: Optional[str] = Field("AUTO", description="컨텍스트 모드 (AUTO|MANUAL)")
    profile_context: Optional[str] = Field(None, description="자녀 요약 프로필 컨텍스트")
    intent_hint: Optional[str] = Field(None, description="의도 힌트 (예: GROWTH_CHECK)")
    requested_profile_domains: Optional[list[str]] = Field(
        None,
        description="요청 대상 도메인 목록 (예: growth, sleep, feeding...)"
    )
    growth_context: Optional[Dict[str, Any]] = Field(None, description="성장 분석용 구조화 컨텍스트")

    @validator("message")
    def validate_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("메시지는 공백일 수 없습니다.")
        if len(normalized) > 2000:
            raise ValueError("메시지는 최대 2000자까지 입력할 수 있습니다.")
        return normalized

    @validator("context_mode", pre=True, always=True)
    def validate_context_mode(cls, value: Optional[str]) -> str:
        if value is None:
            return "AUTO"

        normalized = str(value).strip().upper()
        if normalized not in {"AUTO", "MANUAL"}:
            raise ValueError("context_mode는 AUTO 또는 MANUAL이어야 합니다.")
        return normalized

    @validator("profile_context")
    def validate_profile_context(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        normalized = value.strip()
        if len(normalized) > 4000:
            raise ValueError("profile_context는 최대 4000자까지 입력할 수 있습니다.")
        return normalized

    @validator("intent_hint")
    def validate_intent_hint(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        normalized = value.strip().upper()
        if not normalized:
            return None
        if len(normalized) > 50:
            raise ValueError("intent_hint는 최대 50자까지 입력할 수 있습니다.")
        return normalized

    @validator("requested_profile_domains")
    def validate_requested_profile_domains(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None

        normalized = []
        for raw_domain in value:
            if raw_domain is None:
                continue
            converted = str(raw_domain).strip().lower()
            if converted:
                normalized.append(converted)

        return normalized or None

    @validator("growth_context")
    def validate_growth_context(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("growth_context는 JSON 객체여야 합니다.")
        return value


class ChatResponseCitation(BaseModel):
    """챗봇 응답 메타-출처 모델"""
    label: str = Field(..., description="출처 라벨")
    source_type: str = Field(..., description="출처 타입 (PROFILE/GROWTH_HISTORY/KNOWLEDGE_BASE/PUBLIC_API/SYSTEM_POLICY)")
    basis_date: Optional[str] = Field(None, description="근거 기준일 (선택)")
    note: Optional[str] = Field(None, description="부가 설명")
    url: Optional[str] = Field(None, description="출처 링크 (선택)")


class ChatQuickAction(BaseModel):
    """챗봇 후속 액션"""
    id: str = Field(..., description="액션 식별자")
    label: str = Field(..., description="버튼 라벨")
    action_type: str = Field(..., description="NAVIGATE | OPEN_CHAT_QUERY")
    route: Optional[str] = Field(None, description="앱 내부 라우트")
    query: Optional[str] = Field(None, description="자동 질의 문구")
    intent_hint: Optional[str] = Field(None, description="의도 힌트")
    requested_profile_domains: Optional[list[str]] = Field(None, description="요청 대상 도메인 목록")


class ChatClarificationPayload(BaseModel):
    """재질문(clarification) 안내"""
    question: str = Field(..., description="재질문 문구")
    missing_fields: Optional[list[str]] = Field(None, description="누락된 필드 목록")
    options: Optional[list[ChatQuickAction]] = Field(None, description="선택 옵션 액션")


class ChatResponseMeta(BaseModel):
    """챗봇 UI용 메타데이터 (하위호환 확장)"""
    request_id: Optional[str] = Field(None, description="요청 추적 ID")
    response_mode: str = Field(..., description="ANSWER | CLARIFY | FALLBACK")
    intent: Optional[str] = Field(None, description="의도")
    confidence: Optional[float] = Field(None, description="응답 신뢰도(휴리스틱)")
    fallback_code: Optional[str] = Field(None, description="FALLBACK 사유 코드")
    citations: Optional[list[ChatResponseCitation]] = Field(None, description="출처 라벨 목록")
    follow_up_questions: Optional[list[str]] = Field(None, description="후속 질문 제안")
    quick_actions: Optional[list[ChatQuickAction]] = Field(None, description="빠른 액션")
    clarification: Optional[ChatClarificationPayload] = Field(None, description="재질문 payload")


class ChatResponse(BaseModel):
    """챗봇 응답 모델"""
    reply: str = Field(..., description="AI 응답 메시지")
    session_id: str = Field(..., description="세션 ID")
    timestamp: str = Field(..., description="응답 시각")
    meta: Optional[ChatResponseMeta] = Field(None, description="챗봇 UI 메타데이터")


class HealthCheckResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    message: str
    timestamp: str


def error_payload(code: str, message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "code": code,
        "message": message,
        "data": None,
    }


def _sanitize_profile_context(value: Optional[str], max_length: int = 4000) -> Optional[str]:
    if value is None:
        return None

    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", str(value))
    normalized = normalized.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""

    if len(normalized) > max_length:
        return normalized[:max_length]

    return normalized


def _is_internal_request(request: Request) -> bool:
    if not AI_INTERNAL_TOKEN:
        return False

    header_token = (
        request.headers.get(INTERNAL_SERVICE_TOKEN_HEADER)
        or request.headers.get(INTERNAL_TOKEN_HEADER)
    )
    return bool(header_token) and header_token == AI_INTERNAL_TOKEN


async def verify_internal_request(request: Request) -> bool:
    if not AI_INTERNAL_TOKEN:
        if AI_REQUIRE_INTERNAL_AUTH:
            raise HTTPException(
                status_code=500,
                detail=error_payload("AI_005_INTERNAL_CONFIG", "내부 인증 토큰이 설정되지 않았습니다.")
            )
        return False

    is_internal = _is_internal_request(request)
    if not is_internal and AI_REQUIRE_INTERNAL_AUTH:
        request_id = request.headers.get("X-Request-Id", str(uuid4()))
        provided_token = request.headers.get(INTERNAL_SERVICE_TOKEN_HEADER)
        status_code = 401 if not provided_token else 403
        message = (
            "내부 호출 인증 헤더가 필요합니다."
            if not provided_token else
            "내부 호출 인증 토큰이 일치하지 않습니다."
        )
        logger.warning(
            f"[request_id={request_id}] blocked internal request {INTERNAL_SERVICE_TOKEN_HEADER}={bool(provided_token)}"
        )
        raise HTTPException(
            status_code=status_code,
            detail=error_payload(
                "AI_005_UNAUTHORIZED" if status_code == 401 else "AI_006_FORBIDDEN",
                message
            )
        )

    if not is_internal and AI_INTERNAL_TOKEN and not AI_REQUIRE_INTERNAL_AUTH:
        # no header and optional auth mode: trace only
        request_id = request.headers.get("X-Request-Id")
        logger.debug(f"[request_id={request_id}] untrusted request for /chat")

    return is_internal


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError):
    message = "요청 형식이 올바르지 않습니다."
    if exc.errors():
        message = exc.errors()[0].get("msg", message)

    return JSONResponse(
        status_code=400,
        content=error_payload("AI_003_BAD_REQUEST", message),
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(_: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and {"status", "code", "message"}.issubset(exc.detail.keys()):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    if exc.status_code == 400:
        payload = error_payload("AI_003_BAD_REQUEST", str(exc.detail))
    elif exc.status_code == 404:
        payload = error_payload("AI_004_UNAVAILABLE", "지원하지 않는 엔드포인트입니다.")
    elif exc.status_code == 503:
        payload = error_payload("AI_004_UNAVAILABLE", "AI 서비스에 연결할 수 없습니다.")
    else:
        payload = error_payload("AI_002_UPSTREAM", str(exc.detail))

    return JSONResponse(status_code=exc.status_code, content=payload)


# ========================================
# 세션 관리 (SQLite 기반)
# ========================================

def get_or_create_session(session_id: Optional[str]) -> str:
    """
    세션 ID를 확인하고 없으면 새로 생성합니다.
    """
    if session_id:
        return session_id

    # 새 세션 생성
    new_session_id = str(uuid4())
    logger.info(f"새 세션 생성: {new_session_id}")

    return new_session_id


def require_session_api_enabled():
    if not ENABLE_SESSION_API:
        raise HTTPException(status_code=404, detail="Not Found")


def _limit_items(items: Optional[list], limit: int) -> Optional[list]:
    if not items:
        return None
    limited = items[:limit]
    return limited or None


def _normalize_intent_for_meta(message: str, intent_hint: Optional[str], requested_profile_domains: Optional[Iterable[str]]) -> str:
    normalized_intent = (intent_hint or "").strip().upper()
    if normalized_intent:
        if normalized_intent in {"GROWTH", "GROWTH_QUERY"}:
            return "GROWTH_CHECK"
        if normalized_intent == "ALLERGY_QUERY":
            return "ALLERGY"
        return normalized_intent

    if _looks_like_growth_check(message, intent_hint):
        return "GROWTH_CHECK"

    normalized_domains = [str(item).strip().lower() for item in (requested_profile_domains or []) if str(item).strip()]
    if "medical" in normalized_domains or "allergy" in normalized_domains:
        return "MEDICAL"
    if "vaccination" in normalized_domains:
        return "VACCINATION"
    if "sleep" in normalized_domains:
        return "SLEEP"
    if "feeding" in normalized_domains:
        return "FEEDING"
    if "development" in normalized_domains:
        return "DEVELOPMENT"
    if "routine" in normalized_domains:
        return "ROUTINE"

    normalized_text = (message or "").replace(" ", "")
    if any(keyword in normalized_text for keyword in ["예방접종", "백신", "접종"]):
        return "VACCINATION"
    if any(keyword in normalized_text for keyword in ["수면", "낮잠", "취침", "기상"]):
        return "SLEEP"
    if any(keyword in normalized_text for keyword in ["이유식", "수유", "식단", "분유"]):
        return "FEEDING"
    if any(keyword in normalized_text for keyword in ["발달", "마일스톤"]):
        return "DEVELOPMENT"
    if any(keyword in normalized_text for keyword in ["어린이집", "달빛어린이병원", "병원찾아", "주말병원", "야간병원"]):
        return "AUTO"
    if any(keyword in normalized_text for keyword in ["열", "증상", "병원", "약", "응급", "알레르기"]):
        return "MEDICAL"
    return "AUTO"


def _contains_location_hint(message: str) -> bool:
    text = str(message or "").strip()
    if not text:
        return False

    # 광역/행정구역 패턴을 느슨하게 탐지
    if re.search(r"(서울|부산|대구|인천|광주|대전|울산|세종|제주)", text):
        return True
    if re.search(r"[가-힣]{2,}(시|도|구|군|동|읍|면)\b", text):
        return True
    return False


def _detect_location_search_kind(message: str) -> Optional[str]:
    normalized = (message or "").replace(" ", "")
    if "어린이집" in normalized:
        return "CHILDCARE_CENTER"
    if "달빛어린이병원" in normalized or ("달빛" in normalized and "병원" in normalized):
        return "MOONLIGHT_HOSPITAL"
    if ("야간" in normalized or "주말" in normalized) and "병원" in normalized:
        return "MOONLIGHT_HOSPITAL"
    return None


def _build_quick_action(
    action_id: str,
    label: str,
    action_type: str,
    route: Optional[str] = None,
    query: Optional[str] = None,
    intent_hint: Optional[str] = None,
    requested_profile_domains: Optional[list[str]] = None,
) -> ChatQuickAction:
    return ChatQuickAction(
        id=action_id,
        label=label,
        action_type=action_type,
        route=route,
        query=query,
        intent_hint=intent_hint,
        requested_profile_domains=_limit_items(requested_profile_domains, 3)
    )


def _build_follow_up_questions(intent: Optional[str], response_mode: str) -> Optional[list[str]]:
    if response_mode == "FALLBACK":
        return _limit_items([
            "짧게 다시 질문해 주실래요?",
            "아이 월령/상황을 함께 알려주실래요?",
            "다른 육아 질문으로 도와드릴까요?"
        ], 3)

    normalized_intent = (intent or "AUTO").upper()
    mapping = {
        "VACCINATION": [
            "접종 후 주의사항도 알려드릴까요?",
            "다음 접종 일정을 캘린더에 정리해볼까요?"
        ],
        "GROWTH_CHECK": [
            "최근 측정값 기준으로 다시 설명해드릴까요?",
            "또래 평균과 함께 쉽게 정리해드릴까요?"
        ],
        "SLEEP": [
            "낮잠/밤잠 루틴 예시도 알려드릴까요?",
            "수면 기록 기반으로 조정 포인트를 볼까요?"
        ],
        "FEEDING": [
            "월령별 식단 예시도 드릴까요?",
            "알레르기 주의 식품도 같이 볼까요?"
        ],
        "MEDICAL": [
            "응급 신호 체크리스트도 안내해드릴까요?",
            "병원 방문 전 확인할 내용을 정리해드릴까요?"
        ],
    }
    return _limit_items(mapping.get(normalized_intent, [
        "같은 주제로 더 구체적으로 질문해보실래요?",
        "아이 월령/상황을 알려주시면 더 맞춤형으로 답변할 수 있어요."
    ]), 3)


def _build_default_quick_actions(
    response_mode: str,
    intent: Optional[str],
    original_message: Optional[str] = None
) -> Optional[list[ChatQuickAction]]:
    normalized_intent = (intent or "AUTO").upper()

    if response_mode == "FALLBACK":
        actions = [
            _build_quick_action(
                "retry_last",
                "다시 시도",
                "OPEN_CHAT_QUERY",
                query=(original_message or "").strip() or "다시 알려줘",
                intent_hint=None
            ),
            _build_quick_action(
                "open_guide",
                "가이드 보기",
                "NAVIGATE",
                route="/guide"
            ),
            _build_quick_action(
                "ask_sleep_example",
                "수면 질문 예시",
                "OPEN_CHAT_QUERY",
                query="지금 월령 수면 루틴 예시 알려줘",
                intent_hint="SLEEP",
                requested_profile_domains=["sleep", "routine"]
            ),
        ]
        return _limit_items(actions, 3)

    if response_mode == "CLARIFY":
        # clarification.options와 중복되더라도 1차에서는 UX 일관성을 위해 1개만 추가
        if normalized_intent == "GROWTH_CHECK":
            return _limit_items([
                _build_quick_action("open_record", "성장 기록 화면 열기", "NAVIGATE", route="/record")
            ], 3)
        return None

    if normalized_intent == "VACCINATION":
        actions = [
            _build_quick_action("open_calendar", "캘린더 열기", "NAVIGATE", route="/calendar"),
            _build_quick_action(
                "ask_vaccine_notice",
                "주의사항 물어보기",
                "OPEN_CHAT_QUERY",
                query="예방접종 후 주의사항 알려줘",
                intent_hint="VACCINATION",
                requested_profile_domains=["vaccination", "medical"]
            ),
        ]
    elif normalized_intent == "GROWTH_CHECK":
        actions = [
            _build_quick_action("open_record", "성장 기록 화면 열기", "NAVIGATE", route="/record"),
            _build_quick_action(
                "ask_growth_summary",
                "쉽게 다시 설명",
                "OPEN_CHAT_QUERY",
                query="성장 분석 결과를 쉽게 다시 설명해줘",
                intent_hint="GROWTH_CHECK",
                requested_profile_domains=["growth"]
            ),
        ]
    elif normalized_intent in {"SLEEP", "FEEDING"}:
        actions = [
            _build_quick_action("open_record", "기록 화면 열기", "NAVIGATE", route="/record"),
            _build_quick_action(
                "ask_next_step",
                "실행 방법 물어보기",
                "OPEN_CHAT_QUERY",
                query="오늘 바로 적용할 수 있게 3단계로 알려줘",
                intent_hint=normalized_intent
            ),
        ]
    else:
        actions = [
            _build_quick_action("open_dashboard", "대시보드 보기", "NAVIGATE", route="/dashboard"),
            _build_quick_action("open_guide", "가이드 보기", "NAVIGATE", route="/guide"),
        ]

    return _limit_items(actions, 3)


def _build_citations(
    message: str,
    reply: str,
    intent: Optional[str],
    requested_profile_domains: Optional[list[str]],
    profile_context: Optional[str],
    growth_context: Optional[Dict[str, Any]],
) -> Optional[list[ChatResponseCitation]]:
    citations: list[ChatResponseCitation] = []

    if isinstance(growth_context, dict) and growth_context:
        data_source = growth_context.get("data_source")
        basis_date = growth_context.get("measured_date")
        citations.append(
            ChatResponseCitation(
                label="성장 기록/측정 정보",
                source_type="GROWTH_HISTORY",
                basis_date=str(basis_date) if basis_date else None,
                note=f"data_source={data_source}" if data_source else "저장된 성장 기록 기반"
            )
        )

    if profile_context and profile_context.strip():
        citations.append(
            ChatResponseCitation(
                label="자녀 프로필 저장 정보",
                source_type="PROFILE",
                basis_date=None,
                note="자녀 저장 정보 기반"
            )
        )

    search_kind = _detect_location_search_kind(message)
    if search_kind:
        citations.append(
            ChatResponseCitation(
                label="공공/외부 검색 결과",
                source_type="PUBLIC_API",
                note="검색 도구 결과를 바탕으로 정리"
            )
        )
    else:
        normalized_intent = (intent or "AUTO").upper()
        if normalized_intent not in {"GROWTH_CHECK"}:
            citations.append(
                ChatResponseCitation(
                    label="내부 육아 지식베이스",
                    source_type="KNOWLEDGE_BASE",
                    note="내부 지식베이스 및 도구 기반"
                )
            )

    if any(token in (reply or "") for token in ["의학적 조언", "전문의", "응급", "주의"]):
        citations.append(
            ChatResponseCitation(
                label="안전 가드레일",
                source_type="SYSTEM_POLICY",
                note="안전/의료 안내 규칙 적용"
            )
        )

    # 중복 제거 (source_type + label 기준)
    deduped: list[ChatResponseCitation] = []
    seen = set()
    for item in citations:
        key = (item.source_type, item.label, item.basis_date)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return _limit_items(deduped, 3)


def _build_answer_meta(
    request_id: str,
    payload: ChatRequest,
    reply: str,
    requested_profile_domains: Optional[list[str]],
    profile_context: Optional[str],
    growth_context: Optional[Dict[str, Any]],
) -> Optional[ChatResponseMeta]:
    if not AI_CHAT_META_ENABLED:
        return None

    intent = _normalize_intent_for_meta(payload.message, payload.intent_hint, requested_profile_domains)
    confidence = 0.66
    if intent == "GROWTH_CHECK" and _is_growth_context_ready(growth_context):
        confidence = 0.86
    elif intent in {"SLEEP", "FEEDING", "VACCINATION", "DEVELOPMENT", "ROUTINE"}:
        confidence = 0.72
    elif intent == "MEDICAL":
        confidence = 0.64

    return ChatResponseMeta(
        request_id=request_id,
        response_mode="ANSWER",
        intent=intent,
        confidence=confidence,
        citations=_build_citations(
            message=payload.message,
            reply=reply,
            intent=intent,
            requested_profile_domains=requested_profile_domains,
            profile_context=profile_context,
            growth_context=growth_context
        ),
        follow_up_questions=_build_follow_up_questions(intent, "ANSWER"),
        quick_actions=_build_default_quick_actions("ANSWER", intent, original_message=payload.message),
    )


def _build_fallback_meta(
    request_id: str,
    payload: ChatRequest,
    fallback_code: str,
) -> Optional[ChatResponseMeta]:
    if not AI_CHAT_META_ENABLED:
        return None

    intent = _normalize_intent_for_meta(payload.message, payload.intent_hint, payload.requested_profile_domains)
    return ChatResponseMeta(
        request_id=request_id,
        response_mode="FALLBACK",
        intent=intent,
        fallback_code=fallback_code,
        quick_actions=_build_default_quick_actions("FALLBACK", intent, original_message=payload.message),
        follow_up_questions=_build_follow_up_questions(intent, "FALLBACK"),
        citations=_limit_items([
            ChatResponseCitation(
                label="시스템 오류 처리",
                source_type="SYSTEM_POLICY",
                note="안정성 폴백 응답"
            )
        ], 3)
    )


def _build_location_clarify_response(
    request_id: str,
    payload: ChatRequest,
    requested_profile_domains: Optional[list[str]]
) -> Optional[tuple[str, ChatResponseMeta]]:
    if not AI_CHAT_META_ENABLED:
        return None

    search_kind = _detect_location_search_kind(payload.message)
    if not search_kind:
        return None

    if _contains_location_hint(payload.message):
        return None

    if search_kind == "CHILDCARE_CENTER":
        reply = "어느 지역의 어린이집을 찾으시나요? 시/구(예: 서울 강남구)를 알려주시면 더 정확히 찾아드릴게요."
        example_query = "서울 강남구 어린이집 찾아줘"
    else:
        reply = "어느 지역 병원을 찾으시나요? 시/구(예: 서울 강남구)를 알려주시면 달빛어린이병원을 찾아드릴게요."
        example_query = "서울 강남구 달빛어린이병원 찾아줘"

    intent = _normalize_intent_for_meta(payload.message, payload.intent_hint, requested_profile_domains)
    options = _limit_items([
        _build_quick_action(
            "fill_location_example",
            "예시 질문 넣기",
            "OPEN_CHAT_QUERY",
            query=example_query,
            intent_hint=intent if intent != "AUTO" else None,
            requested_profile_domains=requested_profile_domains,
        ),
        _build_quick_action(
            "open_guide",
            "가이드 보기",
            "NAVIGATE",
            route="/guide"
        )
    ], 4)

    meta = ChatResponseMeta(
        request_id=request_id,
        response_mode="CLARIFY",
        intent=intent,
        clarification=ChatClarificationPayload(
            question="어느 지역 기준으로 찾을까요?",
            missing_fields=["location"],
            options=options
        ),
        quick_actions=_build_default_quick_actions("CLARIFY", intent, original_message=payload.message),
        follow_up_questions=_build_follow_up_questions(intent, "CLARIFY"),
        citations=_limit_items([
            ChatResponseCitation(
                label="검색 조건 확인",
                source_type="SYSTEM_POLICY",
                note="지역 정보가 필요해요"
            )
        ], 3)
    )
    return reply, meta


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


def _looks_like_growth_check(message: str, intent_hint: Optional[str]) -> bool:
    """
    간단한 성장/발달 의도 판별
    """
    if intent_hint and str(intent_hint).strip().upper() == "GROWTH_CHECK":
        return True

    normalized = (message or "").replace(" ", "")
    if not normalized:
        return False

    if "성장발달" in normalized:
        return True

    if "백분위" in normalized or "성장곡선" in normalized:
        return True

    if "성장" in normalized and any(keyword in normalized for keyword in ["확인", "분석", "정상", "또래"]):
        return True

    has_measure = any(keyword in normalized for keyword in ["키", "몸무게", "체중", "신장"])
    has_comparison = any(keyword in normalized for keyword in ["정상", "또래", "평균", "백분위"])
    if has_measure and has_comparison:
        return True

    return "성장발달" in normalized and "확인" in normalized


def _resolve_requested_profile_domains(
    requested_profile_domains: Optional[Iterable[str]],
    intent_hint: Optional[str]
) -> list[str]:
    allowed_domains = {
        "growth",
        "sleep",
        "feeding",
        "vaccination",
        "development",
        "routine",
        "medical",
        "allergy",
        "safety",
    }
    resolved: list[str] = []

    if requested_profile_domains:
        for raw_domain in requested_profile_domains:
            domain = str(raw_domain).strip().lower()
            if domain in allowed_domains and domain not in resolved:
                resolved.append(domain)

    if resolved:
        return resolved

    normalized_intent = (intent_hint or "").strip().upper()
    intent_to_domains = {
        "GROWTH_CHECK": ["growth"],
        "SLEEP": ["sleep", "routine"],
        "FEEDING": ["feeding", "routine"],
        "DEVELOPMENT": ["development"],
        "VACCINATION": ["vaccination", "medical"],
        "ROUTINE": ["routine", "sleep"],
        "MEDICAL": ["medical", "allergy", "safety"],
        "ALLERGY": ["medical", "allergy", "safety"],
    }
    return intent_to_domains.get(normalized_intent, [])


def _is_growth_request(
    message: str,
    intent_hint: Optional[str],
    requested_profile_domains: Optional[Iterable[str]]
) -> bool:
    if _looks_like_growth_check(message, intent_hint):
        return True

    if not requested_profile_domains:
        return False

    normalized = [str(item).strip().lower() for item in requested_profile_domains if str(item).strip()]
    return "growth" in normalized


def _to_birth_str(value: Any) -> Optional[str]:
    """
    날짜/문자열을 YYYY-MM-DD 형식 문자열로 정규화
    """
    if value is None:
        return None

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, datetime):
        return value.date().isoformat()

    text = str(value).strip()
    if not text:
        return None

    # "YYYY-MM-DDTHH:MM:SS" 또는 "YYYY-MM-DD" 모두 지원
    if "T" in text:
        return text.split("T", 1)[0]
    return text


def _to_iso_date(value: Any) -> Optional[date]:
    """
    성장 기록의 측정일자 계산용 날짜 파싱
    """
    if value is None:
        return None

    if isinstance(value, date):
        return value

    if isinstance(value, datetime):
        return value.date()

    text = str(value).strip()
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                return datetime.strptime(text, pattern).date()
            except ValueError:
                continue
        return None


def _normalize_growth_gender(value: Any) -> Optional[str]:
    """
    DB/입력 gender를 M/F로 정규화
    """
    if value is None:
        return None

    normalized = str(value).strip().upper()
    if normalized in {"M", "남", "남아", "남성", "MALE", "1"}:
        return "M"
    if normalized in {"F", "여", "여아", "여성", "FEMALE", "2"}:
        return "F"
    return None


def _is_growth_context_ready(context: Optional[Dict[str, Any]]) -> bool:
    """
    성장 분석에 필요한 최소 필수 필드를 모두 가진 컨텍스트인지 판단
    """
    if not isinstance(context, dict):
        return False

    required_fields = ("gender", "birth_date", "height_cm", "weight_kg")
    for field in required_fields:
        value = context.get(field)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
    if _normalize_growth_gender(context.get("gender")) is None:
        return False
    if _to_iso_date(context.get("birth_date")) is None:
        return False
    if _parse_positive_float(context.get("height_cm")) is None:
        return False
    if _parse_positive_float(context.get("weight_kg")) is None:
        return False
    return True


def _parse_positive_float(value: Any) -> Optional[float]:
    """
    문자열/숫자형에서 양수 실수만 추출해 반환
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None

    text = str(value).replace(",", ".").replace(" ", "")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None

    try:
        parsed = float(match.group(1))
    except ValueError:
        return None

    return parsed if parsed > 0 else None


def _resolve_growth_context_from_child(child_id: int) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    저장된 자녀 정보/성장 기록을 읽어 growth_context와 profile_context의 기초값을 만듭니다.
    """
    try:
        client = get_supabase_client(use_service_role=True)
    except Exception:
        client = get_supabase_client(use_service_role=False)
    baby_rows = []

    baby_id_filters = [
        ("id", {"id": child_id}),
        ("ch_seq", {"ch_seq": child_id}),
        ("child_id", {"child_id": child_id}),
    ]

    for filter_name, filter_payload in baby_id_filters:
        try:
            candidate_rows = client.select_data("babies", filters=filter_payload, limit=1)
            if candidate_rows:
                baby_rows = candidate_rows
                logger.debug(
                    f"[resolve_growth_context] resolved baby row via babies.{filter_name} for child_id={child_id}"
                )
                break
        except Exception as ex:
            logger.warning(
                f"[resolve_growth_context] failed babies lookup with {filter_name}: {str(ex)}"
            )

    if not baby_rows:
        return {}, None, None

    baby_row = baby_rows[0]
    profile_context: Dict[str, Any] = {}
    growth_context: Dict[str, Any] = {}
    if baby_row.get("name"):
        profile_context["name"] = baby_row.get("name")
    if baby_row.get("birth_date"):
        profile_context["birth_date"] = _to_birth_str(baby_row.get("birth_date"))
    if baby_row.get("gender"):
        profile_context["gender"] = _normalize_growth_gender(baby_row.get("gender"))

    # 아이 기본 프로필에 키/몸무게가 저장된 경우도 성장 분석의 보조값으로 사용
    # (DB 스키마에 따라 칼럼명 차이가 있을 수 있으므로 후보 키를 순회)
    child_height_candidates = ["height", "height_cm", "birth_height", "current_height", "latest_height"]
    child_weight_candidates = ["weight", "weight_kg", "birth_weight", "current_weight", "latest_weight"]

    def _pick_first_row_value(row: Dict[str, Any], keys: list[str]) -> Optional[Any]:
        for key in keys:
            if row.get(key) is not None:
                return row.get(key)
        return None

    fallback_height = _pick_first_row_value(baby_row, child_height_candidates)
    fallback_weight = _pick_first_row_value(baby_row, child_weight_candidates)

    if fallback_height is not None:
        growth_context["height_cm"] = fallback_height
        profile_context["base_height_cm"] = fallback_height
    if fallback_weight is not None:
        growth_context["weight_kg"] = fallback_weight
        profile_context["base_weight_kg"] = fallback_weight

    if profile_context.get("gender"):
        growth_context["gender"] = profile_context["gender"]
    if profile_context.get("birth_date"):
        growth_context["birth_date"] = profile_context["birth_date"]

    growth_records = []
    growth_record_lookup_order = [
        ("growth_records", "baby_id", "measured_date", "measured_date"),
        ("growth_records", "child_id", "measured_date", "measured_date"),
        ("growth_records", "ch_seq", "measured_date", "measured_date"),
        ("child_grow_history", "ch_seq", "gh_date", "gh_date"),
        ("child_grow_history", "baby_id", "gh_date", "gh_date"),
    ]

    growth_record_source = None
    growth_record = None

    for table_name, child_key, order_field, date_field in growth_record_lookup_order:
        try:
            candidate_rows = client.select_data(
                table_name,
                filters={child_key: child_id},
                order_by=f"{order_field}.desc",
                limit=1
            )
            if candidate_rows:
                growth_record = candidate_rows[0]
                growth_record_source = f"{table_name}.{child_key}"
                logger.debug(
                    f"[resolve_growth_context] resolved growth record via {table_name} filter {child_key}={child_id}"
                )
                break
        except Exception as ex:
            logger.debug(
                f"[resolve_growth_context] growth lookup failed for {table_name}.{child_key}: {str(ex)}"
            )

    if growth_record is None:
        logger.debug(f"[resolve_growth_context] no growth record found for child_id={child_id}")

    if growth_record:
        if growth_record_source:
            growth_context["data_source"] = growth_record_source

        height_candidates = ["height", "height_cm", "h_cm", "cm"]
        weight_candidates = ["weight", "weight_kg", "kg", "body_weight"]
        head_candidates = ["head_circ", "head_circumference", "head_circumference_cm", "head_cm"]
        measured_date_candidates = ["measured_date", "gh_date", "recorded_at", "reg_date"]

        def _pick_optional(record: Dict[str, Any], keys: list[str]) -> Optional[Any]:
            for key in keys:
                if record.get(key) is not None:
                    return record.get(key)
            return None

        latest_height = _pick_optional(growth_record, height_candidates)
        latest_weight = _pick_optional(growth_record, weight_candidates)
        latest_head = _pick_optional(growth_record, head_candidates)
        latest_measured_raw = _pick_optional(growth_record, measured_date_candidates)

        if latest_height is not None:
            growth_context["height_cm"] = latest_height
        if latest_weight is not None:
            growth_context["weight_kg"] = latest_weight
        if latest_head is not None:
            growth_context["head_circ"] = latest_head

        measured_date = _to_iso_date(latest_measured_raw)
        if measured_date:
            growth_context["measured_date"] = measured_date.isoformat()
            stale_days = (date.today() - measured_date).days
            growth_context["stale_days"] = stale_days
            profile_context["last_measured_date"] = measured_date.isoformat()

        if latest_height is not None:
            profile_context["last_height_cm"] = latest_height
        if latest_weight is not None:
            profile_context["last_weight_kg"] = latest_weight

    return growth_context, profile_context, baby_row


def _merge_growth_context(
    manual_context: Optional[Dict[str, Any]],
    resolved_context: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    사용자 입력 값 우선순위를 유지한 채 성장 컨텍스트 병합
    """
    if not manual_context and not resolved_context:
        return None

    merged = dict(resolved_context or {})
    if manual_context:
        for key, value in manual_context.items():
            if value is None:
                continue

            if isinstance(value, str) and not value.strip():
                continue

            if key in {"height_cm", "weight_kg", "head_circ"}:
                parsed = _parse_positive_float(value)
                if parsed is None:
                    continue
                merged[key] = parsed
                continue

            if key == "gender":
                normalized_gender = _normalize_growth_gender(value)
                if normalized_gender is None:
                    continue
                merged[key] = normalized_gender
                continue

            if key == "birth_date":
                normalized_date = _to_iso_date(value)
                if normalized_date is None:
                    continue
                merged[key] = normalized_date.isoformat()
                continue

            merged[key] = value

    return merged


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
    payload: ChatRequest,
    request: Request,
    response: Response,
    is_internal_request: bool = Depends(verify_internal_request),
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
    request_id = request.headers.get("X-Request-Id") or str(uuid4())
    response.headers["X-Request-Id"] = request_id
    started_at = time.perf_counter()

    session_id = payload.session_id
    try:
        # 세션 ID 가져오기/생성
        session_id = get_or_create_session(session_id)

        # 세션 히스토리 가져오기 (최근 10개)
        history = session_manager.get_history(session_id, limit=10)

        # 사용자 메시지 기록
        session_manager.add_message(session_id, "user", payload.message)

        logger.info(
            f"[request_id={request_id}][session_id={session_id}] chat request received "
            f"(message_length={len(payload.message)}, history_size={len(history)}, "
            f"child_id={payload.child_id}, context_mode={payload.context_mode}, "
            f"intent_hint={payload.intent_hint}, requested_profile_domains={payload.requested_profile_domains}, "
            f"has_profile_context={bool(payload.profile_context)}, "
            f"has_growth_context={bool(payload.growth_context)})"
        )
        logger.debug(
            f"[request_id={request_id}] incoming growth_context={payload.growth_context}"
        )

        effective_profile_context = _sanitize_profile_context(payload.profile_context)
        if payload.context_mode == "MANUAL" and effective_profile_context is None:
            effective_profile_context = ""

        effective_growth_context = payload.growth_context
        resolved_requested_profile_domains = _resolve_requested_profile_domains(
            payload.requested_profile_domains,
            payload.intent_hint
        )
        if (
            payload.child_id is not None
            and not payload.context_mode == "MANUAL"
            and is_internal_request
            and _is_growth_request(
                payload.message,
                payload.intent_hint,
                resolved_requested_profile_domains
            )
            and not _is_growth_context_ready(effective_growth_context)
        ):
            try:
                resolved_growth_context, resolved_profile_context, _ = _resolve_growth_context_from_child(
                    payload.child_id
                )
                logger.debug(
                    f"[request_id={request_id}] resolved growth_context_from_child={resolved_growth_context}, "
                    f"profile_context={resolved_profile_context}"
                )

                if payload.profile_context is None and resolved_profile_context:
                    if effective_profile_context == "":
                        effective_profile_context = None

                    if effective_profile_context is None:
                        profile_lines = [
                            "[자녀 저장 정보]",
                            f"- 이름: {resolved_profile_context.get('name', '기입되지 않음')}"
                        ]
                        if resolved_profile_context.get("birth_date"):
                            profile_lines.append(f"- 생년월일: {resolved_profile_context['birth_date']}")
                        if resolved_profile_context.get("gender"):
                            profile_lines.append(f"- 성별: {resolved_profile_context['gender']}")
                        if resolved_profile_context.get("last_measured_date"):
                            profile_lines.append(
                                f"- 최근 측정일: {resolved_profile_context['last_measured_date']}"
                            )
                        if resolved_profile_context.get("last_height_cm") is not None:
                            profile_lines.append(
                                f"- 최근 키: {resolved_profile_context['last_height_cm']} cm"
                            )
                        if resolved_profile_context.get("last_weight_kg") is not None:
                            profile_lines.append(
                                f"- 최근 몸무게: {resolved_profile_context['last_weight_kg']} kg"
                            )
                        effective_profile_context = _sanitize_profile_context("\n".join(profile_lines))

                effective_growth_context = _merge_growth_context(
                    effective_growth_context,
                    resolved_growth_context
                )
                logger.debug(
                    f"[request_id={request_id}] effective_growth_context={effective_growth_context}"
                )
            except Exception as ex:
                logger.warning(
                    f"[request_id={request_id}] failed to resolve growth context from child_id "
                    f"{payload.child_id}: {str(ex)}"
                )

        location_clarify = _build_location_clarify_response(
            request_id=request_id,
            payload=payload,
            requested_profile_domains=resolved_requested_profile_domains
        )
        if location_clarify:
            clarify_reply, clarify_meta = location_clarify
            session_manager.add_message(session_id, "assistant", clarify_reply)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                f"[request_id={request_id}][session_id={session_id}] chat request completed "
                f"(elapsed_ms={elapsed_ms}, reply_length={len(clarify_reply)}, response_mode=CLARIFY)"
            )
            return ChatResponse(
                reply=clarify_reply,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                meta=clarify_meta
            )

        # 에이전트 비동기 호출
        reply = await agent.achat(
            user_input=payload.message,
            chat_history=history,
            profile_context=effective_profile_context,
            intent_hint=payload.intent_hint,
            growth_context=effective_growth_context,
            requested_profile_domains=resolved_requested_profile_domains
        )

        # AI 응답 기록
        session_manager.add_message(session_id, "assistant", reply)
        meta = None
        if AI_CHAT_META_ENABLED:
            try:
                meta = _build_answer_meta(
                    request_id=request_id,
                    payload=payload,
                    reply=reply,
                    requested_profile_domains=resolved_requested_profile_domains,
                    profile_context=effective_profile_context,
                    growth_context=effective_growth_context
                )
            except Exception as meta_ex:
                logger.warning(
                    f"[request_id={request_id}] failed to build answer meta: {type(meta_ex).__name__}: {str(meta_ex)}"
                )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            f"[request_id={request_id}][session_id={session_id}] chat request completed "
            f"(elapsed_ms={elapsed_ms}, reply_length={len(reply)}, response_mode={meta.response_mode if meta else 'ANSWER'})"
        )

        return ChatResponse(
            reply=reply,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            meta=meta
        )

    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        if not session_id:
            session_id = get_or_create_session(payload.session_id)

        logger.error(
            f"[request_id={request_id}] chatbot error "
            f"(session_id={session_id}, elapsed_ms={elapsed_ms}, error_type={type(e).__name__})"
        )
        fallback_reply = "죄송합니다. AI 서비스에서 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        if session_id:
            try:
                session_manager.add_message(session_id, "assistant", fallback_reply)
            except Exception as append_error:
                logger.error(
                    f"[request_id={request_id}] fallback append failed "
                    f"(session_id={session_id}, error_type={type(append_error).__name__})"
                )
        fallback_code = "UNKNOWN_ERROR"
        error_type_name = type(e).__name__.upper()
        if "TIMEOUT" in error_type_name:
            fallback_code = "TIMEOUT"
        elif isinstance(e, ValueError):
            fallback_code = "VALIDATION_ERROR"
        elif "CONNECTION" in error_type_name:
            fallback_code = "UPSTREAM_UNAVAILABLE"
        elif "RUNTIME" in error_type_name:
            fallback_code = "UPSTREAM_ERROR"

        fallback_meta = None
        if AI_CHAT_META_ENABLED:
            try:
                fallback_meta = _build_fallback_meta(
                    request_id=request_id,
                    payload=payload,
                    fallback_code=fallback_code
                )
            except Exception as meta_ex:
                logger.warning(
                    f"[request_id={request_id}] failed to build fallback meta: {type(meta_ex).__name__}: {str(meta_ex)}"
                )
        return ChatResponse(
            reply=fallback_reply,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            meta=fallback_meta
        )


@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """
    특정 세션의 대화 히스토리 조회
    """
    require_session_api_enabled()
    history = session_manager.get_history(session_id, limit=50)
    if not history:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없거나 기록이 없습니다.")

    return {
        "session_id": session_id,
        "history": history
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    세션 삭제
    """
    require_session_api_enabled()
    session_manager.delete_session(session_id)
    return {"message": f"세션 {session_id}와 관련된 모든 대화 기록이 삭제되었습니다."}



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
