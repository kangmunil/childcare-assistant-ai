from dataclasses import dataclass, field
from typing import List, Optional
from src.safety.medical_guard import MedicalGuard
from src.safety.content_filter import ContentFilter

@dataclass
class SafetyAssessment:
    """
    안전성 평가 결과 객체
    [Refactor Note] 단순 문자열 리스트 반환의 한계를 극복하기 위해 구조화된 객체 사용
    """
    is_safe: bool = True          # 안전 여부 (Pass 가능 여부)
    is_emergency: bool = False    # 응급 상황 여부 (즉시 차단 필요)
    warnings: List[str] = field(default_factory=list) # 사용자에게 보여줄 메시지들
    action_type: str = "PASS"     # 'BLOCK', 'WARN_AND_ANSWER', 'PASS'

class SafetyManager:
    """
    프로젝트의 모든 안전 장치를 통합 관리하는 클래스 (Facade Pattern)
    """
    
    def __init__(self):
        self.medical = MedicalGuard()
        self.content = ContentFilter()

    def check_input_safety(self, user_text: str, age_months: Optional[float] = None) -> SafetyAssessment:
        """
        사용자의 질문을 분석하여 안전성 평가 결과를 반환합니다.
        
        [Refactor Note]
        - 응급 상황(Emergency Signs, High Fever) -> BLOCK (답변 거부하고 경고만 표시)
        - 단순 주의(Contraindications) -> WARN_AND_ANSWER (경고 표시 후 답변 진행)
        """
        warnings = []
        is_emergency = False
        action = "PASS"

        # 1. 응급 상황 체크 (가장 높은 우선순위 -> BLOCK)
        emergencies = self.medical.check_emergency(user_text)
        if emergencies:
            warnings.extend(emergencies)
            is_emergency = True
            action = "BLOCK"
        
        # 2. 발열 체크 (고열은 응급으로 간주 -> BLOCK)
        fevers = self.medical.check_fever(user_text, age_months)
        if fevers:
            warnings.extend(fevers)
            # 이미 BLOCK 상태가 아니라면 BLOCK으로 격상 (보수적 접근)
            if not is_emergency:
                is_emergency = True
                action = "BLOCK"

        # 3. 금기 사항 체크 (단순 질문일 수 있으므로 -> WARN_AND_ANSWER)
        # 예: "꿀 먹여도 돼?" -> 경고문 + "안 됩니다"라는 AI의 설명이 필요함.
        contra = self.medical.check_contraindications(user_text)
        if contra:
            warnings.extend(contra)
            # 현재 상태가 PASS라면 WARN_AND_ANSWER로 변경 (BLOCK이면 유지)
            if action == "PASS":
                action = "WARN_AND_ANSWER"
        
        return SafetyAssessment(
            is_safe=(action == "PASS"),
            is_emergency=is_emergency,
            warnings=warnings,
            action_type=action
        )

    def process_output_safety(self, ai_response: str, is_health_related: bool = False) -> str:
        """
        AI의 답변을 검수하고 안전 장치를 추가합니다.
        """
        # 1. 일반 필터링 (PII 마스킹 등)
        safe_response = self.content.filter_response(ai_response)
        
        # 2. 건강 관련 질문일 경우 면책 조항 추가
        if is_health_related:
            safe_response += self.medical.get_medical_disclaimer()
            
        return safe_response

# 싱글톤 인스턴스
safety_manager = SafetyManager()