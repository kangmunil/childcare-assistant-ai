"""
[Childcare Assistant AI - Pipeline Entry Point]
RAG 파이프라인을 실행하여 데이터를 적재하고 검색 테스트를 수행합니다.
"""

import os
from src.rag.pipeline import DataPipeline
from langchain_core.documents import Document

def create_mock_data(path: str):
    """
    테스트를 위한 가상 데이터 생성
    """
    os.makedirs(path, exist_ok=True)
    
    sample_texts = [
        """
        [육아 가이드] 생후 3개월 수유 가이드
        생후 3개월 아기의 수유량은 1회 160cc에서 200cc 정도가 적당합니다.
        수유 텀은 4시간 간격으로 유지하는 것이 좋습니다.
        밤수(밤중 수유)는 서서히 줄여나갈 준비를 해야 합니다.
        """,
        """
        [할머니의 지혜] 아기 배앓이에는 기적의 꿀물?
        옛날에는 아기가 배가 아프면 꿀물을 타 먹이곤 했습니다.
        이것은 민간요법으로 아주 효과가 좋습니다. 꼭 해보세요.
        """,
        """
        [발달 정보] 6개월 아기 발달
        이 시기의 아기는 뒤집기를 자유자재로 할 수 있습니다.
        빠른 아기는 배밀이를 시작하기도 합니다.
        낯가림이 시작될 수 있으니 주의해주세요.
        """,
        """
        [건강] 아기 열 날 때 대처법
        38도 이상의 고열이 나면 해열제를 먹여야 합니다.
        아세트아미노펜 계열은 생후 4개월부터, 이부프로펜은 6개월부터 가능합니다.
        """
    ]
    
    for i, text in enumerate(sample_texts):
        with open(f"{path}/doc_{i}.txt", "w", encoding="utf-8") as f:
            f.write(text.strip())

def main():
    # 데이터 경로 설정
    raw_data_dir = "./data/raw_samples"
    
    # 0. 가상 데이터 생성 (실제 운영 시에는 이 단계 생략)
    create_mock_data(raw_data_dir)
    
    # 파이프라인 초기화
    pipeline = DataPipeline()
    
    # 파이프라인 실행
    print(">>> Starting Childcare RAG Pipeline...")
    retriever = pipeline.run_pipeline(raw_data_dir)
    
    if retriever:
        print("\n>>> Pipeline execution successful. Testing Hybrid Search...")
        
        # 검색 테스트
        query = "6개월 아기 해열제 뭐 먹여?"
        print(f"\nQuery: {query}")
        
        results = retriever.invoke(query)
        
        for i, doc in enumerate(results):
            print(f"\n[Result {i+1}]")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
    else:
        print("Pipeline failed to produce a retriever.")

if __name__ == "__main__":
    main()
