import os
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일을 가장 먼저 명시적으로 로드
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
load_dotenv(dotenv_path=env_path, override=True)

class Settings(BaseSettings):
    """
    애플리케이션 전체 설정 관리 (Pydantic 기반)
    .env 파일에서 환경 변수를 로드하고 유효성을 검증합니다.
    """
    
    # === 1. 기본 설정 ===
    APP_NAME: str = "Baby-Bot"
    ENV: Literal["development", "production", "testing"] = "development"
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "./logs/app.log"

    # === 2. LLM & Embedding 설정 ===
    # LLM
    LLM_MODEL: str = Field(default="google/gemini-2.0-flash-exp:free", description="메인 LLM 모델명")
    LLM_TEMPERATURE: float = 0.7
    
    # Embedding
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="임베딩 모델명")
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, description="OpenRouter API Key")
    OPENAI_API_BASE: Optional[str] = Field(default=None, description="Custom API Base URL")

    # === 3. Database & Storage ===
    # Vector Store
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    
    # RDB (SQLite/Supabase)
    SQLITE_DB_PATH: str = "data/childcare.db"
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None

    # === 4. Computed Properties (동적 설정) ===
    @computed_field
    def EFFECTIVE_API_KEY(self) -> str:
        """사용 가능한 API Key를 반환 (OpenRouter 우선)"""
        key = self.OPENROUTER_API_KEY or self.OPENAI_API_KEY
        if not key:
            # 임베딩이 로컬 모델이 아닌 경우에만 경고/에러가 필요하지만, 
            # 일단 LLM 구동을 위해 키가 없으면 경고 값을 반환 (Log에서 확인 가능)
            return "MISSING_API_KEY"
        return key

    @computed_field
    def EFFECTIVE_API_BASE(self) -> str:
        """API Base URL 결정"""
        if self.OPENAI_API_BASE:
            return self.OPENAI_API_BASE
        
        if self.OPENROUTER_API_KEY:
            return "https://openrouter.ai/api/v1"
        
        return "https://api.openai.com/v1" # Default OpenAI

    @computed_field
    def IS_LOCAL_EMBEDDING(self) -> bool:
        """로컬 임베딩 모델 사용 여부 판단"""
        # OpenAI/OpenRouter/Google API를 사용하는 모델들을 필터링
        api_keywords = ["text-embedding", "openai", "google", "gemini"]
        return not any(kw in self.EMBEDDING_MODEL.lower() for kw in api_keywords)

    model_config = SettingsConfigDict(
        # 현재 파일(src/core/config.py) 기준 상위상위 폴더가 프로젝트 루트
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore" # .env에 정의되지 않은 변수는 무시
    )

# 싱글톤 인스턴스 생성
settings = Settings()

# 디렉토리 자동 생성
os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH), exist_ok=True)
