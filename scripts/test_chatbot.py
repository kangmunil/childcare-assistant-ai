from src.rag.childcare_agent import ChildcareAgent
from loguru import logger
from src.core.config import settings
import os
import sys

def main():
    logger.info("ChatBot 종합 테스트를 시작합니다.")
    logger.info(f"Using LLM_MODEL from settings: {settings.LLM_MODEL}")
    logger.info(f"Using EMBEDDING_MODEL from settings: {settings.EMBEDDING_MODEL}")
    
    # 에이전트 초기화
    try:
        agent = ChildcareAgent()
    except Exception as e:
        logger.error(f"에이전트 초기화 실패: {e}")
        return

    # 테스트 질문 리스트
    test_cases = [
        {
            "category": "성장 분석 (Tool 사용)",
            "question": "2024년 12월 30일생 남아인데 오늘 키를 재보니 76cm이고 몸무게는 10kg이야. 우리 아이 잘 크고 있는 거야?"
        },
        {
            "category": "발달 지식 (RAG 사용)",
            "question": "아기가 보통 옹알이는 언제 시작하고, 뒤집기는 언제쯤 하는 게 정상이야?"
        },
        {
            "category": "복합 질문 (Tool + RAG)",
            "question": "현재 12개월 남아 평균 몸무게가 얼마인지 알려주고, 이 시기 아기들에게 해주면 좋은 놀이도 추천해줘."
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"테스트 {i}: [{case['category']}]")
        print(f"질문: {case['question']}")
        print(f"{'-'*50}")
        
        try:
            response = agent.chat(case['question'])
            print(f"AI 응답:\n{response}")
        except Exception as e:
            print(f"에러 발생: {e}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
