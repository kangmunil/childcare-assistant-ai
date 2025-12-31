import re
from typing import List
from loguru import logger

class ContentFilter:
    """
    비속어, 민감 정보, 부적절한 민간요법 정보를 필터링합니다.
    """

    # 1. 신뢰할 수 없는 키워드 (민간요법 등)
    UNRELIABLE_KEYWORDS = [
        "민간요법", "카더라", "할머니가 그러는데", "옛날에는", 
        "과학적 근거는 없지만", "특효약", "기적의", "절대", "무조건"
    ]

    # 2. 개인정보 패턴 (PII)
    PII_PATTERNS = {
        "resident_id": r'\d{6}-\d{7}',
        "phone": r'010-\d{4}-\d{4}',
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    }

    def is_reliable(self, text: str) -> bool:
        """
        텍스트에 신뢰할 수 없는 키워드가 포함되어 있는지 검사합니다.
        """
        for keyword in self.UNRELIABLE_KEYWORDS:
            if keyword in text:
                logger.warning(f"Unreliable content detected: '{keyword}'")
                return False
        return True

    def mask_pii(self, text: str) -> str:
        """
        개인정보를 마스킹 처리합니다.
        """
        for name, pattern in self.PII_PATTERNS.items():
            text = re.sub(pattern, f"[{name.upper()}_REMOVED]", text)
        return text

    def filter_response(self, text: str) -> str:
        """
        최종 답변을 필터링합니다.
        """
        text = self.mask_pii(text)
        # 여기에 추가적인 비속어 필터링 등을 넣을 수 있습니다.
        return text
