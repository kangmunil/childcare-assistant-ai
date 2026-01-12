"""
육아 헬퍼 AI 챗봇 사용 예시

이 파일은 에이전틱 RAG 챗봇의 사용 방법을 보여줍니다.
"""

import sys
from pathlib import Path
import requests
import json

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.childcare_agent import ChildcareAgent
from src.rag.document_processor import DocumentProcessor
from loguru import logger


def example_1_document_embedding():
    """
    예제 1: 문서 임베딩 및 벡터 DB 저장
    """
    logger.info("=" * 80)
    logger.info("예제 1: 문서 임베딩 및 벡터 DB 저장")
    logger.info("=" * 80)

    processor = DocumentProcessor()

    # docs 디렉토리의 Markdown 파일 로드
    docs_dir = "./docs"

    logger.info(f"문서 로드 중: {docs_dir}")

    try:
        # 1. 문서 로드
        documents = processor.load_markdown_files(docs_dir)

        if not documents:
            logger.warning("로드된 문서가 없습니다.")
            return

        # 2. 청킹
        chunks = processor.split_documents(documents)

        # 3. 벡터 DB 저장
        vectorstore = processor.create_vectorstore(
            documents=chunks,
            collection_name="childcare_knowledge"
        )

        logger.success(f"벡터 DB 생성 완료: {vectorstore._collection.count()}개 벡터")

        # 4. 테스트 검색
        test_queries = [
            "달빛어린이병원이 뭐야?",
            "성장도표는 어떻게 계산해?",
            "예방접종은 언제 하나요?"
        ]

        for query in test_queries:
            logger.info(f"\n[검색] {query}")
            results = processor.search_similar_documents(
                query=query,
                collection_name="childcare_knowledge",
                k=2
            )

            for i, doc in enumerate(results, 1):
                logger.info(f"  [{i}] {doc.page_content[:150]}...")

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")


def example_2_chatbot_cli():
    """
    예제 2: CLI 챗봇 (명령줄 대화)
    """
    logger.info("\n" + "=" * 80)
    logger.info("예제 2: CLI 챗봇 - 육아 헬퍼 AI와 대화하기")
    logger.info("=" * 80)
    logger.info("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")

    try:
        # 에이전트 초기화
        agent = ChildcareAgent()

        # 대화 히스토리
        chat_history = []

        while True:
            # 사용자 입력
            user_input = input("\n[사용자] ")

            if user_input.lower() in ['quit', 'exit', '종료']:
                logger.info("챗봇을 종료합니다.")
                break

            if not user_input.strip():
                continue

            # AI 응답
            try:
                response = agent.chat(user_input, chat_history)
                print(f"\n[AI 비서] {response}")

                # 히스토리 저장
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response})

            except Exception as e:
                logger.error(f"오류: {str(e)}")
                print(f"\n[오류] 응답 생성 중 오류가 발생했습니다: {str(e)}")

    except KeyboardInterrupt:
        logger.info("\n\n챗봇을 종료합니다.")


def example_3_test_questions():
    """
    예제 3: 미리 정의된 질문으로 챗봇 테스트
    """
    logger.info("\n" + "=" * 80)
    logger.info("예제 3: 테스트 질문으로 챗봇 기능 검증")
    logger.info("=" * 80)

    # 에이전트 초기화
    agent = ChildcareAgent()

    # 테스트 질문들
    test_cases = [
        {
            "category": "지식 검색 (RAG)",
            "question": "달빛어린이병원이 뭐야?"
        },
        {
            "category": "Function Calling - 어린이집 검색",
            "question": "서울특별시 강남구 어린이집 찾아줘"
        },
        {
            "category": "Function Calling - 성장 분석",
            "question": "12개월 남자아이 몸무게가 10.5kg인데 정상이야?"
        },
        {
            "category": "복합 질문",
            "question": "아기가 밤에 열이 나는데 어떻게 해야 해? 그리고 근처 병원도 찾아줘."
        }
    ]

    for i, test in enumerate(test_cases, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"테스트 {i}: {test['category']}")
        logger.info(f"{'=' * 80}")
        logger.info(f"[질문] {test['question']}")

        try:
            answer = agent.chat(test['question'])
            logger.info(f"\n[답변]\n{answer}\n")

        except Exception as e:
            logger.error(f"오류: {str(e)}")


def example_4_api_client():
    """
    예제 4: FastAPI 서버와 통신 (HTTP 클라이언트)
    """
    logger.info("\n" + "=" * 80)
    logger.info("예제 4: FastAPI 챗봇 API 호출")
    logger.info("=" * 80)
    logger.info("먼저 API 서버를 실행하세요:")
    logger.info("  python src/api/chatbot_api.py")
    logger.info("=" * 80 + "\n")

    base_url = "http://localhost:8000"

    # 1. 헬스 체크
    try:
        response = requests.get(f"{base_url}/health")

        if response.status_code == 200:
            logger.success("API 서버 연결 성공!")
            logger.info(f"응답: {response.json()}")
        else:
            logger.error(f"API 서버 연결 실패: {response.status_code}")
            return

    except requests.exceptions.ConnectionError:
        logger.error("API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        return

    # 2. 챗봇 대화
    session_id = None

    test_messages = [
        "안녕? 너는 뭐하는 AI야?",
        "달빛어린이병원이 뭔지 알려줘",
        "서울 강남구 어린이집 찾아줘"
    ]

    for message in test_messages:
        logger.info(f"\n[사용자] {message}")

        payload = {
            "message": message,
            "session_id": session_id
        }

        response = requests.post(
            f"{base_url}/chat",
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            session_id = result["session_id"]

            logger.info(f"[AI 비서] {result['reply']}")
            logger.debug(f"세션 ID: {session_id}")
        else:
            logger.error(f"오류: {response.status_code} - {response.text}")


def main():
    """
    모든 예제 실행 (선택적)
    """
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("육아 헬퍼 AI 챗봇 - 사용 예시")
    logger.info("=" * 80)
    logger.info("\n실행할 예제를 선택하세요:")
    logger.info("  1. 문서 임베딩 및 벡터 DB 저장")
    logger.info("  2. CLI 챗봇 (대화형)")
    logger.info("  3. 테스트 질문으로 검증")
    logger.info("  4. FastAPI 클라이언트")
    logger.info("  0. 종료")

    while True:
        choice = input("\n선택 (0-4): ").strip()

        if choice == "1":
            example_1_document_embedding()
        elif choice == "2":
            example_2_chatbot_cli()
        elif choice == "3":
            example_3_test_questions()
        elif choice == "4":
            example_4_api_client()
        elif choice == "0":
            logger.info("프로그램을 종료합니다.")
            break
        else:
            logger.warning("잘못된 입력입니다. 0-4 중에서 선택하세요.")


if __name__ == "__main__":
    main()
