from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from loguru import logger

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional at runtime
    ChatOpenAI = None

from src.analysis.growth_analyzer import GrowthAnalyzer
from src.core.config import settings
from src.rag.knowledge_base import ChildcareKnowledgeBase
from src.safety import safety_manager


class ChildcareAgent:
    ALLOWED_PROFILE_DOMAINS = {
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
    PROFILE_DOMAIN_LABELS = {
        "growth": "성장",
        "sleep": "수면",
        "feeding": "수유",
        "vaccination": "예방접종",
        "development": "발달",
        "routine": "생활습관",
        "medical": "건강",
        "allergy": "알레르기",
        "safety": "안전",
    }
    GROWTH_INTENT_HINTS = {"GROWTH", "GROWTH_CHECK", "GROWTH_QUERY"}

    def __init__(self) -> None:
        self.model_name = settings.LLM_MODEL
        self.system_prompt = self._build_system_prompt()
        self._growth_analyzer = GrowthAnalyzer()
        self._knowledge_base: Optional[ChildcareKnowledgeBase] = None
        self._llm = self._create_llm()

        try:
            self._knowledge_base = ChildcareKnowledgeBase()
        except Exception as exc:
            logger.warning(f"Knowledge base initialization skipped: {exc}")

        logger.info("ChildcareAgent restored")
        logger.info(f"  - model: {self.model_name}")
        logger.info(f"  - kb_ready: {self._knowledge_base is not None}")
        logger.info(f"  - llm_ready: {self._llm is not None}")

    def _build_system_prompt(self) -> str:
        return (
            "당신은 영유아 돌봄 도우미입니다. "
            "근거가 부족하면 단정하지 말고, 건강 관련 질문에는 보수적으로 답하세요. "
            "성장 분석은 수치가 충분할 때만 수행하고, 부족하면 한 번에 한 항목만 요청하세요."
        )

    def _create_llm(self):
        if ChatOpenAI is None:
            return None
        if settings.EFFECTIVE_API_KEY == "MISSING_API_KEY":
            return None
        try:
            return ChatOpenAI(
                model=self.model_name,
                api_key=settings.EFFECTIVE_API_KEY,
                base_url=settings.EFFECTIVE_API_BASE,
                temperature=settings.LLM_TEMPERATURE,
            )
        except Exception as exc:
            logger.warning(f"LLM initialization skipped: {exc}")
            return None

    def _normalize_requested_profile_domains(self, requested_profile_domains: Optional[Iterable[Any]]) -> list[str]:
        if not requested_profile_domains:
            return []

        normalized: list[str] = []
        for raw_domain in requested_profile_domains:
            if raw_domain is None:
                continue
            converted = str(raw_domain).strip().lower()
            if converted and converted in self.ALLOWED_PROFILE_DOMAINS and converted not in normalized:
                normalized.append(converted)
        return normalized

    def _build_profile_context(
        self,
        profile_context: Optional[str],
        requested_profile_domains: Optional[Iterable[Any]] = None,
    ) -> Optional[str]:
        normalized_domains = self._normalize_requested_profile_domains(requested_profile_domains)
        parts: list[str] = []
        if normalized_domains:
            labels = [self.PROFILE_DOMAIN_LABELS.get(domain, domain) for domain in normalized_domains]
            parts.append(f"요청 도메인: {', '.join(labels)}")

        if profile_context and str(profile_context).strip():
            parts.append(str(profile_context).strip())

        if not parts:
            return None
        return "\n".join(parts)

    def _to_langchain_messages(self, chat_history: Optional[Iterable[Any]] = None) -> list[BaseMessage]:
        if not chat_history:
            return []

        normalized_messages: list[BaseMessage] = []
        for entry in chat_history:
            if isinstance(entry, BaseMessage):
                normalized_messages.append(entry)
                continue

            if not isinstance(entry, dict):
                continue

            content = str(entry.get("content", "")).strip()
            if not content:
                continue

            role = str(entry.get("role", "user")).strip().lower()
            if role == "assistant":
                normalized_messages.append(AIMessage(content=content))
            else:
                normalized_messages.append(HumanMessage(content=content))

        return normalized_messages

    def _normalize_gender(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        converted = str(getattr(value, "value", value)).strip().upper()
        if converted in {"M", "MALE", "1"}:
            return "M"
        if converted in {"F", "FEMALE", "2"}:
            return "F"
        return None

    def _parse_positive_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _parse_birth_date(self, value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        return None

    def _growth_prompt_for_missing_field(self, field_name: str) -> str:
        prompts = {
            "gender": "성별만 알려주세요. 예: M",
            "birth_date": "생년월일만 알려주세요. 예: 2024-01-01",
            "height_cm": "키만 알려주세요. 예: 92.4",
            "weight_kg": "몸무게만 알려주세요. 예: 13.2",
        }
        return prompts.get(field_name, "필요한 정보를 한 가지만 더 알려주세요.")

    def _is_growth_check_intent(self, message: Optional[str], intent_hint: Optional[str]) -> bool:
        normalized_hint = str(intent_hint or "").strip().upper()
        if normalized_hint in self.GROWTH_INTENT_HINTS:
            return True

        normalized = str(message or "").replace(" ", "")
        growth_keywords = ["성장", "백분위", "키", "몸무게", "체중", "발달"]
        growth_actions = ["확인", "분석", "평가", "정상", "알려", "비교"]

        if "백분위" in normalized:
            return True
        if "성장발달" in normalized:
            return True
        if any(keyword in normalized for keyword in growth_keywords) and any(action in normalized for action in growth_actions):
            return True
        return False

    def _build_growth_auto_response(self, growth_context: Optional[dict[str, Any]]) -> Optional[str]:
        if not isinstance(growth_context, dict):
            return None

        normalized_gender = self._normalize_gender(growth_context.get("gender"))
        birth_date = self._parse_birth_date(growth_context.get("birth_date"))
        height_cm = self._parse_positive_float(growth_context.get("height_cm"))
        weight_kg = self._parse_positive_float(growth_context.get("weight_kg"))
        head_circ = self._parse_positive_float(growth_context.get("head_circ"))
        stale_days = growth_context.get("stale_days")

        required_fields = [
            ("gender", normalized_gender),
            ("birth_date", birth_date),
            ("height_cm", height_cm),
            ("weight_kg", weight_kg),
        ]
        missing_fields = [field_name for field_name, field_value in required_fields if field_value is None]
        if missing_fields:
            return self._growth_prompt_for_missing_field(missing_fields[0])

        analyzer = getattr(self, "_growth_analyzer", None) or GrowthAnalyzer()
        result = analyzer.assess_growth(
            gender=normalized_gender,
            birth_date=birth_date,
            height=height_cm,
            weight=weight_kg,
            head_circ=head_circ,
        )

        if result.get("status") != "success":
            return result.get("message") or "성장 기준 데이터가 아직 준비되지 않아 정밀 분석을 할 수 없어요."

        analysis = result.get("analysis", {})
        lines = ["성장 분석 결과입니다."]

        height_result = analysis.get("height")
        if height_result:
            lines.append(
                f"키는 {height_result['percentile']}백분위로 {height_result['status']} 범위예요."
            )

        weight_result = analysis.get("weight")
        if weight_result:
            lines.append(
                f"몸무게는 {weight_result['percentile']}백분위로 {weight_result['status']} 범위예요."
            )

        weight_for_height_result = analysis.get("weight_for_height")
        if weight_for_height_result:
            lines.append(
                f"키 대비 몸무게는 {weight_for_height_result['percentile']}백분위로 {weight_for_height_result['status']} 범위예요."
            )

        head_result = analysis.get("head_circumference")
        if head_result:
            lines.append(
                f"머리둘레는 {head_result['percentile']}백분위로 {head_result['status']} 범위예요."
            )

        warnings = result.get("warnings") or []
        if warnings:
            lines.append("주의: " + warnings[0])

        if isinstance(stale_days, (int, float)) and stale_days >= 30:
            lines.append(f"다만 {int(stale_days)}일 전 데이터라 최근 측정값으로 다시 확인해 주세요.")

        return "\n".join(lines)

    def _is_health_related(self, message: Optional[str], requested_profile_domains: Optional[Iterable[Any]]) -> bool:
        normalized_domains = set(self._normalize_requested_profile_domains(requested_profile_domains))
        if normalized_domains.intersection({"medical", "allergy", "vaccination", "safety"}):
            return True

        normalized = str(message or "")
        keywords = ["열", "체온", "병원", "예방접종", "알레르기", "응급"]
        return any(keyword in normalized for keyword in keywords)

    def _has_local_knowledge_data(self) -> bool:
        persist_dir = Path(getattr(settings, "CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"))
        if not persist_dir.is_absolute():
            persist_dir = Path.cwd() / persist_dir
        return persist_dir.exists() and any(persist_dir.iterdir())

    def _search_knowledge_base(self, query: str) -> list[Any]:
        knowledge_base = getattr(self, "_knowledge_base", None)
        if knowledge_base is None:
            return []
        if not self._has_local_knowledge_data():
            return []
        try:
            return knowledge_base.search(query, k=3)
        except Exception as exc:
            logger.warning(f"Knowledge base search failed: {exc}")
            return []

    def _build_knowledge_context(self, query: str) -> Optional[str]:
        documents = self._search_knowledge_base(query)
        if not documents:
            return None

        excerpts: list[str] = []
        for index, document in enumerate(documents, start=1):
            content = str(getattr(document, "page_content", "")).strip()
            if not content:
                continue
            excerpts.append(f"자료 {index}: {content[:500]}")
        return "\n".join(excerpts) if excerpts else None

    def _fallback_response(
        self,
        user_input: str,
        profile_context: Optional[str],
        knowledge_context: Optional[str],
        health_related: bool,
    ) -> str:
        lines = []
        if knowledge_context:
            lines.append("저장된 육아 지식 기준으로 답변드릴게요.")
            lines.append(knowledge_context)
        else:
            lines.append("현재 저장된 지식 데이터가 없어 일반 안내 기준으로 답변드릴게요.")

        if profile_context:
            lines.append("프로필 컨텍스트를 반영했어요.")

        lines.append(f"질문: {user_input}")
        lines.append("필요하면 아이 월령, 증상, 키/몸무게처럼 한 가지 정보만 더 알려주세요.")
        response = "\n".join(lines)
        return safety_manager.process_output_safety(response, is_health_related=health_related)

    def _build_messages(
        self,
        user_input: str,
        chat_history: Optional[Iterable[Any]],
        profile_context: Optional[str],
        knowledge_context: Optional[str],
    ) -> list[BaseMessage]:
        context_parts = [self.system_prompt]
        if profile_context:
            context_parts.append("[프로필 컨텍스트]\n" + profile_context)
        if knowledge_context:
            context_parts.append("[지식 베이스 요약]\n" + knowledge_context)

        messages: list[BaseMessage] = [SystemMessage(content="\n\n".join(context_parts))]
        messages.extend(self._to_langchain_messages(chat_history))
        messages.append(HumanMessage(content=user_input))
        return messages

    def _invoke_llm_sync(self, messages: list[BaseMessage]) -> Optional[str]:
        if self._llm is None:
            return None
        try:
            response = self._llm.invoke(messages)
            return str(getattr(response, "content", "")).strip() or None
        except Exception as exc:
            logger.warning(f"LLM sync invoke failed: {exc}")
            return None

    async def _invoke_llm_async(self, messages: list[BaseMessage]) -> Optional[str]:
        if self._llm is None:
            return None
        try:
            response = await self._llm.ainvoke(messages)
            return str(getattr(response, "content", "")).strip() or None
        except Exception as exc:
            logger.warning(f"LLM async invoke failed: {exc}")
            return None

    def chat(
        self,
        user_input: str,
        chat_history: Optional[Iterable[Any]] = None,
        profile_context: Optional[str] = None,
        intent_hint: Optional[str] = None,
        growth_context: Optional[dict[str, Any]] = None,
        requested_profile_domains: Optional[Iterable[Any]] = None,
    ) -> str:
        normalized_domains = self._normalize_requested_profile_domains(requested_profile_domains)
        combined_profile_context = self._build_profile_context(profile_context, normalized_domains)

        if self._is_growth_check_intent(user_input, intent_hint):
            auto_response = self._build_growth_auto_response(growth_context)
            if auto_response:
                return auto_response

        health_related = self._is_health_related(user_input, normalized_domains)
        safety = safety_manager.check_input_safety(user_input)
        if safety.action_type == "BLOCK":
            return "\n".join(safety.warnings)

        knowledge_context = self._build_knowledge_context(user_input)
        messages = self._build_messages(user_input, chat_history, combined_profile_context, knowledge_context)
        llm_response = self._invoke_llm_sync(messages)
        if llm_response:
            response = llm_response
            if safety.warnings:
                response = "\n".join(safety.warnings + [response])
            return safety_manager.process_output_safety(response, is_health_related=health_related)

        fallback = self._fallback_response(user_input, combined_profile_context, knowledge_context, health_related)
        if safety.warnings:
            return "\n".join(safety.warnings + [fallback])
        return fallback

    async def achat(
        self,
        user_input: str,
        chat_history: Optional[Iterable[Any]] = None,
        profile_context: Optional[str] = None,
        intent_hint: Optional[str] = None,
        growth_context: Optional[dict[str, Any]] = None,
        requested_profile_domains: Optional[Iterable[Any]] = None,
    ) -> str:
        normalized_domains = self._normalize_requested_profile_domains(requested_profile_domains)
        combined_profile_context = self._build_profile_context(profile_context, normalized_domains)

        if self._is_growth_check_intent(user_input, intent_hint):
            auto_response = self._build_growth_auto_response(growth_context)
            if auto_response:
                return auto_response

        health_related = self._is_health_related(user_input, normalized_domains)
        safety = safety_manager.check_input_safety(user_input)
        if safety.action_type == "BLOCK":
            return "\n".join(safety.warnings)

        knowledge_context = self._build_knowledge_context(user_input)
        messages = self._build_messages(user_input, chat_history, combined_profile_context, knowledge_context)
        llm_response = await self._invoke_llm_async(messages)
        if llm_response:
            response = llm_response
            if safety.warnings:
                response = "\n".join(safety.warnings + [response])
            return safety_manager.process_output_safety(response, is_health_related=health_related)

        fallback = self._fallback_response(user_input, combined_profile_context, knowledge_context, health_related)
        if safety.warnings:
            return "\n".join(safety.warnings + [fallback])
        return fallback

