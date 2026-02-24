import unittest
from unittest.mock import patch

from src.rag.childcare_agent import ChildcareAgent


class TestGrowthAutoResponse(unittest.TestCase):
    def setUp(self):
        # __init__를 우회해 외부 의존성(ChatOpenAI/벡터DB)을 로드하지 않도록 한다.
        self.agent = ChildcareAgent.__new__(ChildcareAgent)

    @patch("src.rag.childcare_agent.GrowthAnalyzer")
    def test_returns_analysis_immediately_when_required_fields_exist(self, analyzer_cls):
        analyzer = analyzer_cls.return_value
        analyzer.assess_growth.return_value = {
            "status": "success",
            "age_months": 24.1,
            "analysis": {
                "height": {"percentile": 58.2, "status": "정상 (보통)"},
                "weight": {"percentile": 62.0, "status": "정상 (보통)"},
                "weight_for_height": {"percentile": 55.3, "status": "정상"},
            },
        }

        response = self.agent._build_growth_auto_response({
            "gender": "M",
            "birth_date": "2024-02-01",
            "height_cm": 87.3,
            "weight_kg": 12.8,
            "stale_days": 5,
        })

        self.assertIn("성장 분석 결과", response)
        self.assertIn("키는 58.2백분위", response)
        self.assertIn("몸무게는 62.0백분위", response)
        self.assertNotIn("알려주세요", response)

    @patch("src.rag.childcare_agent.GrowthAnalyzer")
    def test_adds_stale_warning_when_measurement_is_old(self, analyzer_cls):
        analyzer = analyzer_cls.return_value
        analyzer.assess_growth.return_value = {
            "status": "success",
            "age_months": 30.0,
            "analysis": {
                "height": {"percentile": 45.0, "status": "정상 (보통)"},
                "weight": {"percentile": 49.0, "status": "정상 (보통)"},
            },
        }

        response = self.agent._build_growth_auto_response({
            "gender": "F",
            "birth_date": "2023-07-01",
            "height_cm": 93.1,
            "weight_kg": 14.0,
            "stale_days": 31,
        })

        self.assertIn("31일 전 데이터", response)

    def test_requests_only_one_missing_field(self):
        response = self.agent._build_growth_auto_response({
            "gender": "M",
            "birth_date": "2023-05-20",
            "height_cm": None,
            "weight_kg": 11.0,
        })

        self.assertEqual("키만 알려주세요. 예: 92.4", response)

    def test_growth_intent_detection_keyword(self):
        self.assertTrue(self.agent._is_growth_check_intent("아이 성장발달 확인하고 싶어", None))
        self.assertTrue(self.agent._is_growth_check_intent("백분위 알려줘", None))
        self.assertFalse(self.agent._is_growth_check_intent("수면 교육 방법 알려줘", None))


if __name__ == "__main__":
    unittest.main()
