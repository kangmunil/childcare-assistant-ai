from src.rag.pipeline import DataPipeline
import os
from loguru import logger
from dotenv import load_dotenv

def main():
    # 환경 변수 강제 로드
    load_dotenv(override=True)
    
    # API 키 확인 (로컬 임베딩 사용 시 필수 아님)
    # if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
    #     logger.warning("API Key is not set. Local embeddings will be used if configured.")

    pipeline = DataPipeline()
    # processed 폴더를 대상으로 실행
    pipeline.run_pipeline("data/processed")
    
if __name__ == "__main__":
    main()
