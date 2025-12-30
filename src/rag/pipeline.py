"""
[Childcare Assistant AI - RAG Data Pipeline]
이 모듈은 'Baby-Bot' 프로젝트의 핵심 데이터 처리 파이프라인을 구현합니다.
Raw Data를 수집하여 세척, 메타데이터 추출, 청킹, 임베딩을 거쳐 Vector DB에 적재합니다.

작성자: Senior AI Engineer (Baby-Bot Architect)
작성일: 2025-12-30
"""

import re
import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SafetyFilter:
    """
    [Safety Guardrail]
    부적절하거나 위험한 콘텐츠(민간요법, 검증되지 않은 정보)를 필터링합니다.
    """
    
    UNRELIABLE_KEYWORDS = [
        "민간요법", "카더라", "할머니가 그러는데", "옛날에는", 
        "과학적 근거는 없지만", "특효약", "기적의", "절대", "무조건"
    ]

    @staticmethod
    def is_safe(text: str) -> bool:
        """
        텍스트에 신뢰할 수 없는 키워드가 포함되어 있는지 검사합니다.
        """
        for keyword in SafetyFilter.UNRELIABLE_KEYWORDS:
            if keyword in text:
                logger.warning(f"Safety Filter Triggered: Found '{keyword}'")
                return False
        return True

class MetadataExtractor:
    """
    [Step 2: Semantic Metadata Extraction]
    텍스트 내용을 분석하여 타겟 월령, 카테고리, 신뢰도 등의 메타데이터를 추출합니다.
    보수적인 Rule-based 로직을 우선 사용합니다.
    """

    CATEGORIES = ["feeding", "sleep", "development", "health"]
    
    # 간단한 키워드 매핑 (실제 운영 시에는 더 정교한 LLM 분류 권장)
    KEYWORD_MAP = {
        "feeding": ["수유", "분유", "모유", "이유식", "젖병"],
        "sleep": ["수면", "잠투정", "낮잠", "통잠", "수면교육"],
        "development": ["발달", "뒤집기", "기어가기", "걸음마", "옹알이"],
        "health": ["열", "체온", "예방접종", "병원", "응급", "약"]
    }

    @staticmethod
    def extract(text: str, source_type: str = "medium") -> Dict[str, Any]:
        """
        텍스트에서 메타데이터를 추출합니다.
        
        Args:
            text (str): 분석할 텍스트
            source_type (str): 데이터 소스 유형 (high/medium/low) - 기본값 medium (서적 등)
        
        Returns:
            Dict: 추출된 메타데이터
        """
        metadata = {
            "target_month_start": "UNTAGGED",
            "target_month_end": "UNTAGGED",
            "category": "general",
            "source_reliability": source_type,
            "processed_at": datetime.now().isoformat()
        }

        # 1. 월령 추출 (Regex: "N개월", "생후 N개월")
        # 예: "생후 3개월 아기는..." -> start:3, end:3
        month_matches = re.findall(r'(\d+)\s*개월', text)
        if month_matches:
            months = [int(m) for m in month_matches if 0 <= int(m) <= 24] # 0~24개월 필터링
            if months:
                metadata["target_month_start"] = min(months)
                metadata["target_month_end"] = max(months)
        
        # 2. 카테고리 분류 (Keyword Counting)
        counts = {cat: 0 for cat in MetadataExtractor.CATEGORIES}
        for cat, keywords in MetadataExtractor.KEYWORD_MAP.items():
            for kw in keywords:
                if kw in text:
                    counts[cat] += 1
        
        # 가장 많이 매칭된 카테고리 선정
        best_category = max(counts, key=counts.get)
        if counts[best_category] > 0:
            metadata["category"] = best_category

        return metadata

class DataPipeline:
    """
    [Project 'Baby-Bot' RAG Pipeline]
    Raw Data -> Cleaning -> Metadata -> Chunking -> Embedding -> VectorStore
    """

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self.persist_dir = persist_dir
        # [Step 4: Embedding Model]
        
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        logger.info(f"Using Embedding Model: {embedding_model_name}")

        # HuggingFace 로컬 모델인지 확인 (예: intfloat/...)
        if "text-embedding-3" in embedding_model_name:
            # OpenAI / OpenRouter
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE")
            
            if os.getenv("OPENROUTER_API_KEY") and not base_url:
                base_url = "https://openrouter.ai/api/v1"

            if not api_key:
                logger.warning("API Key not found for OpenAI embeddings.")

            self.embeddings = OpenAIEmbeddings(
                model=embedding_model_name,
                openai_api_key=api_key,
                base_url=base_url
            )
        else:
            # HuggingFace Local (OpenSource)
            logger.info("Initializing Local HuggingFace Embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': 'cpu'}, # GPU가 있으면 'cuda'
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # [Step 3: Context-Aware Chunking Strategy]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 300~500 tokens recommendation
            chunk_overlap=75, # ~15% overlap
            separators=["\n\n", "\n", ".", "!", "?", " "]
        )

    def step1_preprocess(self, text: str) -> str:
        """
        [Step 1: Preprocessing & Cleaning]
        HTML 태그 제거, 특수문자 정리, PII 제거, 단위 표준화
        """
        # 0. HTML 태그 제거
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

        # 1. PII 제거 (주민번호, 전화번호 등 - 간단한 예시)
        text = re.sub(r'\d{6}-\d{7}', '[RESIDENT_ID_REMOVED]', text)
        text = re.sub(r'010-\d{4}-\d{4}', '[PHONE_REMOVED]', text)

        # 2. 단위 표준화
        text = text.replace("cc", "ml").replace("CC", "ml") # 수유량 단위 통일
        text = text.replace("hr", "시간").replace("hour", "시간")
        
        # 3. 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def step2_enrich_metadata(self, docs: List[Document]) -> List[Document]:
        """
        [Step 2: Semantic Metadata Extraction]
        각 문서에 메타데이터를 주입합니다.
        """
        enriched_docs = []
        for doc in docs:
            # 안전성 검사
            if not SafetyFilter.is_safe(doc.page_content):
                continue

            # 메타데이터 추출
            meta = MetadataExtractor.extract(doc.page_content)
            doc.metadata.update(meta)
            enriched_docs.append(doc)
        
        return enriched_docs

    def step3_context_aware_chunking(self, docs: List[Document]) -> List[Document]:
        """
        [Step 3: Context-Aware Chunking]
        문서를 청킹하고, 각 청크의 앞부분에 문맥(Context)을 덧붙입니다.
        [Requirement] 'Health' category must have a disclaimer.
        [Requirement] Handle 'UNTAGGED' months.
        """
        chunks = self.text_splitter.split_documents(docs)
        
        enriched_chunks = []
        for chunk in chunks:
            # Context Prepending: 
            # 메타데이터를 활용하여 청크 앞단에 문맥 정보를 강제로 주입합니다.
            start_m = chunk.metadata.get('target_month_start')
            end_m = chunk.metadata.get('target_month_end')
            category = chunk.metadata.get('category', 'general')
            
            # 월령 정보 포맷팅
            if start_m == "UNTAGGED" or end_m == "UNTAGGED":
                month_info = "월령미상"
            else:
                month_info = f"{start_m}~{end_m}개월"

            context_header = f"[{month_info} {category} 정보] "
            
            # Medical Disclaimer 추가
            disclaimer = ""
            if category == "health":
                disclaimer = "\n[주의: 이 정보는 의학적 진단을 대신할 수 없습니다. 증상이 심하면 즉시 병원을 방문하세요.]"

            # 원본 내용을 수정하여 context와 disclaimer를 포함시킴
            chunk.page_content = context_header + chunk.page_content + disclaimer
            enriched_chunks.append(chunk)
            
        return enriched_chunks

    def step4_indexing(self, chunks: List[Document]):
        """
        [Step 4: Embedding & Vector Storage]
        처리된 청크를 ChromaDB에 저장합니다.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return None

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="childcare_knowledge"
        )
        logger.info(f"Indexed {len(chunks)} chunks into ChromaDB at {self.persist_dir}")
        return vectorstore

    def step5_create_retriever(self, vectorstore) -> EnsembleRetriever:
        """
        [Step 5: Hybrid Search Retriever]
        BM25(키워드) + Vector(의미) 검색을 결합합니다.
        """
        # 1. Vector Retriever (Semantic) - 70% Weight
        # k=5: 상위 5개 추출
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 2. BM25 Retriever (Keyword) - 30% Weight
        # Chroma vectorstore의 docstore를 활용하거나 원본 청크 리스트가 필요함.
        # 여기서는 vectorstore에 저장된 문서를 다시 가져올 수 없으므로(일반적으로), 
        # 파이프라인 상에서 chunks를 메모리에 들고 있다가 넘겨줘야 함.
        # *주의: 실제 구현 시에는 별도 인덱싱 필요. 여기선 데모용으로 바로 생성.
        # 이 메서드는 chunks 리스트를 인자로 받아야 정확함.
        # 편의상 vectorstore만 받으면 BM25 구성이 어려우므로 구조를 조정할 수 있음.
        pass 
    
    def run_pipeline(self, raw_docs_path: str):
        """
        전체 파이프라인 실행 함수
        """
        logger.info("Pipeline Started...")
        
        # 0. Load Raw Data (Support TXT and PDF)
        raw_docs = []
        
        # Load TXT
        try:
            txt_loader = DirectoryLoader(raw_docs_path, glob="**/*.txt", loader_cls=TextLoader)
            raw_docs.extend(txt_loader.load())
        except Exception as e:
            logger.warning(f"TXT loading warning: {e}")

        # Load PDF
        try:
            pdf_loader = DirectoryLoader(raw_docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
            raw_docs.extend(pdf_loader.load())
        except Exception as e:
            logger.warning(f"PDF loading warning: {e}")

        logger.info(f"Loaded {len(raw_docs)} raw documents.")

        # 1. Preprocessing
        cleaned_docs = []
        for doc in raw_docs:
            doc.page_content = self.step1_preprocess(doc.page_content)
            cleaned_docs.append(doc)
        logger.info("Step 1 (Preprocessing) Complete.")

        # 2. Metadata Extraction & Safety Filter
        tagged_docs = self.step2_enrich_metadata(cleaned_docs)
        logger.info(f"Step 2 (Metadata) Complete. {len(tagged_docs)} valid docs remaining.")

        # 3. Chunking
        final_chunks = self.step3_context_aware_chunking(tagged_docs)
        logger.info(f"Step 3 (Chunking) Complete. Generated {len(final_chunks)} chunks.")

        # 4. Indexing (Vector Store)
        vectorstore = self.step4_indexing(final_chunks)
        logger.info("Step 4 (Indexing) Complete.")

        # 5. Return Retriever (Hybrid)
        if final_chunks:
            bm25_retriever = BM25Retriever.from_documents(final_chunks)
            bm25_retriever.k = 5
            
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.3, 0.7] # 30% Keyword, 70% Semantic
            )
            logger.info("Step 5 (Hybrid Retriever) Configured.")
            return ensemble_retriever
        
        return None
