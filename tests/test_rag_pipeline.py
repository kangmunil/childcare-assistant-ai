import unittest
from src.rag.pipeline import DataPipeline, MetadataExtractor, SafetyFilter
from langchain_core.documents import Document

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = DataPipeline()

    def test_step1_preprocess_html(self):
        html_text = "<div><p>생후 3개월 아기</p></div>"
        cleaned = self.pipeline.step1_preprocess(html_text)
        self.assertEqual(cleaned, "생후 3개월 아기")

    def test_step1_preprocess_units(self):
        text = "하루 1000cc 수유, 2hr 낮잠"
        cleaned = self.pipeline.step1_preprocess(text)
        self.assertEqual(cleaned, "하루 1000ml 수유, 2시간 낮잠")

    def test_metadata_extraction_months(self):
        text = "생후 4개월부터 6개월까지의 발달"
        metadata = MetadataExtractor.extract(text)
        self.assertEqual(metadata["target_month_start"], 4)
        self.assertEqual(metadata["target_month_end"], 6)

        no_month_text = "일반적인 육아 정보입니다."
        metadata_none = MetadataExtractor.extract(no_month_text)
        self.assertEqual(metadata_none["target_month_start"], "UNTAGGED")
        self.assertEqual(metadata_none["target_month_end"], "UNTAGGED")

    def test_metadata_extraction_category(self):
        text = "아기가 밤에 잠을 안 자요. 수면 교육이 필요합니다."
        metadata = MetadataExtractor.extract(text)
        self.assertEqual(metadata["category"], "sleep")

    def test_safety_filter(self):
        unsafe_text = "이 약은 기적의 특효약입니다. 절대 믿으세요."
        self.assertFalse(SafetyFilter.is_safe(unsafe_text))
        
        safe_text = "병원에 방문하여 의사의 처방을 받으세요."
        self.assertTrue(SafetyFilter.is_safe(safe_text))

if __name__ == '__main__':
    unittest.main()
