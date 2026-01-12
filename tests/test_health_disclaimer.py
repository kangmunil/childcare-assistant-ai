import unittest
from langchain_core.documents import Document
from src.rag.knowledge_base import ChildcareKnowledgeBase

class TestHealthDisclaimer(unittest.TestCase):
    def setUp(self):
        self.kb = ChildcareKnowledgeBase()

    def test_health_disclaimer_appended(self):
        # Create a mock document tagged as 'health'
        doc = Document(
            page_content="아기 열이 39도입니다.",
            metadata={
                "target_month_start": 6, 
                "target_month_end": 12, 
                "category": "health"
            }
        )
        
        # Run step 3 (context aware chunking)
        # Accessing private method for testing purpose
        chunks = self.kb._context_aware_chunking([doc])
        
        # Verify disclaimer is present
        self.assertTrue(len(chunks) > 0)
        # Note: the actual text in knowledge_base.py is "[주의: 의학적 조언이 아닙니다. 전문의와 상담하세요.]"
        self.assertIn("[주의: 의학적 조언이 아닙니다", chunks[0].page_content)
        self.assertIn("[6~12개월 health 정보]", chunks[0].page_content)

    def test_no_disclaimer_for_general(self):
        # Create a mock document tagged as 'general'
        doc = Document(
            page_content="아기가 잘 놀아요.",
            metadata={
                "target_month_start": 6, 
                "target_month_end": 12, 
                "category": "general"
            }
        )
        
        # Run step 3
        chunks = self.kb._context_aware_chunking([doc])
        
        # Verify disclaimer is NOT present
        self.assertTrue(len(chunks) > 0)
        self.assertNotIn("[주의: 의학적 조언이 아닙니다", chunks[0].page_content)

if __name__ == '__main__':
    unittest.main()
