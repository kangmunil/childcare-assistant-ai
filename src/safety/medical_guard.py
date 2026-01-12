import re
from typing import List, Dict, Optional
from loguru import logger

class MedicalGuard:
    """
    영유아(0~12개월) 특화 의학적 안전 가드레일.
    Recall > Precision 원칙에 따라 응급 상황을 보수적으로 감지합니다.
    """

    # 1. 즉시 응급실에 가야 하는 레드 플래그 (Red Flags)
    EMERGENCY_SIGNS = {
        "respiratory": ["숨을 안 쉬어요", "호흡 곤란", "청색증", "쌕쌕거림", "가슴 몰몰", "숨소리"],
        "neurological": ["경련", "발작", "의식이 없어요", "축 늘어져요", "눈이 돌아가요", "경기"],
        "gastrointestinal": ["분수토", "혈변", "회색변", "담도폐쇄", "심한 탈수"],
        "general": ["고열", "멈추지 않는 울음", "자지러지게 울어요"]
    }

    # 2. 월령별 발열 가이드라인
    FEVER_RULES = [
        {"max_month": 3, "temp": 38.0, "message": "생후 100일 미만 아기의 38도 이상 발열은 즉시 응급실 방문이 필요합니다."},
        {"max_month": 6, "temp": 39.0, "message": "생후 6개월 미만 아기의 39도 이상 고열은 위험할 수 있으니 즉시 진료를 받으세요."},
        {"max_month": 12, "temp": 39.0, "message": "돌 전 아기가 39도 이상의 고열이 있거나 발열이 24시간 지속되면 소아과 방문이 필수입니다."}
    ]

    # 3. 절대 금기 사항 (Taboos)
    CONTRAINDICATIONS = {
        "꿀": "돌 전 아기에게 꿀은 보툴리누스 균 중독 위험이 있어 절대 금지입니다.",
        "생우유": "돌 전 아기는 소화 기관이 미성숙하여 생우유를 마시면 장내 출혈이 발생할 수 있습니다.",
        "간장": "아기 상처에 간장이나 된장을 바르는 민간요법은 감염의 위험이 매우 큽니다."
    }

    def check_emergency(self, text: str) -> List[str]:
        """
        텍스트에서 응급 상황 키워드를 감지합니다.
        
        [Refactor Note]
        - 기존: 하나라도 찾으면 break (중복 증상 누락 가능성)
        - 변경: 모든 카테고리를 순회하며 감지. 단, 동일 카테고리 내에서는 1개만 찾으면 break 하여 효율성 확보.
        """
        detected_warnings = []
        for category, keywords in self.EMERGENCY_SIGNS.items():
            for kw in keywords:
                if kw in text:
                    detected_warnings.append(f"[응급 상황 감지: {kw}] 지금 당장 가까운 응급실이나 소아과를 방문하세요!")
                    # 해당 카테고리 내에서 증상을 발견했으므로, 다음 카테고리로 넘어감 (중복 방지 및 효율성)
                    break 
        return detected_warnings

    def check_fever(self, text: str, age_months: Optional[float] = None) -> List[str]:
        """
        온도 관련 언급이 있을 경우 월령별 가이드를 제공합니다.
        
        [Refactor Note]
        - 단위 없는 숫자만 있어도 문맥(열, 체온 등)이 있으면 체온으로 인식하도록 정규식 및 로직 강화.
        """
        warnings = []
        
        # 1. 숫자와 단위(옵션) 추출 정규식
        # 예: "38.5도", "39", "37.5" 등을 모두 포착
        candidates = re.findall(r'(\d{2}(?:\.\d)?)', text)
        
        # 2. 문맥 키워드 확인
        context_keywords = ['열', '체온', '뜨거', 'hot']
        has_context = any(kw in text for kw in context_keywords)
        
        # 3. 단위 확인을 위한 별도 정규식
        strict_pattern = re.compile(r'(\d{2}(?:\.\d)?)\s*(도|度|℃)')
        
        max_detected_temp = 0.0
        
        for num_str in candidates:
            try:
                val = float(num_str)
                is_valid_range = 35.0 <= val <= 42.0 # 유효 체온 범위
                
                # 조건 1: 명확한 단위가 붙어있는 경우
                # 조건 2: 단위는 없지만 문맥 키워드가 있고 유효 범위인 경우
                is_strict_match = bool(strict_pattern.search(text)) # 텍스트 내에 단위가 하나라도 있는지 (약식 체크)
                
                # 정밀 체크: 현재 숫자가 단위와 붙어있는지 확인하려면 복잡해지므로,
                # "문맥이 있거나" OR "숫자가 단위와 결합된 패턴에 포함되거나" 로 판단
                
                if is_valid_range:
                    if has_context:
                        max_detected_temp = max(max_detected_temp, val)
                    elif re.search(fr'{re.escape(num_str)}\s*(도|度|℃)', text):
                        max_detected_temp = max(max_detected_temp, val)
                        
            except ValueError:
                continue

        # 감지된 최고 체온으로 규칙 적용
        if max_detected_temp >= 37.5:
            logger.info(f"체온 감지: {max_detected_temp}도 (월령: {age_months})")
            if age_months is not None:
                for rule in self.FEVER_RULES:
                    if age_months <= rule["max_month"] and max_detected_temp >= rule["temp"]:
                        warnings.append(rule["message"])
                        break
            else:
                if max_detected_temp >= 38.0:
                    warnings.append(f"감지된 체온 {max_detected_temp}도는 고열입니다. 아기의 월령이 낮을수록 위험하니 소아과 진료를 권장합니다.")
                    
        return warnings

    def check_contraindications(self, text: str) -> List[str]:
        """
        치명적인 금기 사항을 체크합니다.
        """
        warnings = []
        for food, message in self.CONTRAINDICATIONS.items():
            if food in text:
                warnings.append(f"[주의] {message}")
        return warnings

    def get_medical_disclaimer(self) -> str:
        return "\n\n[면책 조항: 본 답변은 참고용이며 의학적 진단을 대신할 수 없습니다. 아이의 상태가 평소와 다르다면 즉시 전문의와 상담하세요.]"