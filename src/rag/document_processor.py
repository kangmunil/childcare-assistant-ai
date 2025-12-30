"""
문서 임베딩 및 벡터 DB 저장 모듈

육아 가이드 문서, 크롤링 데이터 등을 처리하여
Vector Database에 저장하고 검색 가능하게 만듭니다.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

load_dotenv()


class DocumentProcessor:
    """
    문서 처리 및 벡터 DB 저장 클래스

    주요 기능:
    1. 문서 로드 (Markdown, Text 등)
    2. 청킹 (Chunking) - 의미 단위로 분할
    3. 임베딩 (Embedding) - 텍스트를 벡터로 변환
    4. 벡터 DB 저장 (ChromaDB)
    """

    def __init__(
        self,
        embedding_model: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: str = None
    ):
        """
        Args:
            embedding_model: OpenAI 임베딩 모델명
            chunk_size: 청크 크기 (문자 수)
            chunk_overlap: 청크 간 중복 크기
            persist_directory: ChromaDB 저장 경로
        """
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL",
            "text-embedding-3-small"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            "./data/chroma_db"
        )

        # OpenRouter 지원
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        
        if os.getenv("OPENROUTER_API_KEY") and not base_url:
            base_url = "https://openrouter.ai/api/v1"

        # 임베딩 모델 초기화
        if "text-embedding-3" in self.embedding_model:
            # OpenAI / OpenRouter
            if not api_key:
                logger.warning("API Key not found for embeddings.")

            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=api_key,
                base_url=base_url
            )
        else:
            # HuggingFace Local (OpenSource)
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                logger.info(f"Initializing Local HuggingFace Embeddings: {self.embedding_model}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except ImportError:
                logger.error("HuggingFaceEmbeddings requires sentence-transformers. Please install it.")
                raise

        # 텍스트 스플리터 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        logger.info(f"DocumentProcessor 초기화 완료")
        logger.info(f"  - 임베딩 모델: {self.embedding_model}")
        logger.info(f"  - 청크 크기: {self.chunk_size}")
        logger.info(f"  - 저장 경로: {self.persist_directory}")

    def load_markdown_files(self, directory: str) -> List[Document]:
        """
        디렉토리에서 Markdown 파일을 로드합니다.

        Args:
            directory: 문서 디렉토리 경로

        Returns:
            LangChain Document 객체 리스트
        """
        logger.info(f"Markdown 파일 로드 시작: {directory}")

        # DirectoryLoader를 사용한 일괄 로드
        loader = DirectoryLoader(
            directory,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True
        )

        documents = loader.load()
        logger.info(f"로드 완료: {len(documents)}개 문서")

        return documents

    def load_single_file(self, file_path: str) -> List[Document]:
        """
        단일 파일을 로드합니다.

        Args:
            file_path: 파일 경로

        Returns:
            LangChain Document 객체 리스트
        """
        logger.info(f"파일 로드: {file_path}")

        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        documents = loader.load()
        logger.info(f"로드 완료: {len(documents)}개 문서")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서를 청크로 분할합니다.

        Args:
            documents: LangChain Document 리스트

        Returns:
            분할된 Document 리스트
        """
        logger.info(f"문서 분할 시작: {len(documents)}개 문서")

        chunks = self.text_splitter.split_documents(documents)

        logger.info(f"분할 완료: {len(chunks)}개 청크 생성")

        return chunks

    def create_vectorstore(
        self,
        documents: List[Document],
        collection_name: str = "childcare_docs"
    ) -> Chroma:
        """
        문서를 임베딩하여 Vector DB에 저장합니다.

        Args:
            documents: Document 리스트
            collection_name: 컬렉션 이름

        Returns:
            Chroma vectorstore 객체
        """
        logger.info(f"벡터 DB 생성 시작: {len(documents)}개 문서")
        logger.info(f"  - 컬렉션 이름: {collection_name}")

        # 기존 컬렉션이 있으면 로드, 없으면 생성
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.persist_directory
        )

        logger.success(f"벡터 DB 생성 완료: {vectorstore._collection.count()}개 벡터")

        return vectorstore

    def load_vectorstore(
        self,
        collection_name: str = "childcare_docs"
    ) -> Chroma:
        """
        기존 Vector DB를 로드합니다.

        Args:
            collection_name: 컬렉션 이름

        Returns:
            Chroma vectorstore 객체
        """
        logger.info(f"벡터 DB 로드 시작: {collection_name}")

        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        logger.info(f"벡터 DB 로드 완료: {vectorstore._collection.count()}개 벡터")

        return vectorstore

    def add_documents_to_existing(
        self,
        documents: List[Document],
        collection_name: str = "childcare_docs"
    ):
        """
        기존 Vector DB에 문서를 추가합니다.

        Args:
            documents: 추가할 Document 리스트
            collection_name: 컬렉션 이름
        """
        logger.info(f"문서 추가 시작: {len(documents)}개")

        vectorstore = self.load_vectorstore(collection_name)
        vectorstore.add_documents(documents)

        logger.success(f"문서 추가 완료")

    def search_similar_documents(
        self,
        query: str,
        collection_name: str = "childcare_docs",
        k: int = 5
    ) -> List[Document]:
        """
        유사 문서를 검색합니다.

        Args:
            query: 검색 쿼리
            collection_name: 컬렉션 이름
            k: 반환할 문서 개수

        Returns:
            유사도 높은 Document 리스트
        """
        vectorstore = self.load_vectorstore(collection_name)

        results = vectorstore.similarity_search(query, k=k)

        logger.info(f"검색 완료: '{query}' → {len(results)}개 결과")

        return results


def process_childcare_documents():
    """
    육아 문서를 처리하여 Vector DB에 저장하는 예제 함수
    """
    processor = DocumentProcessor()

    # 1. docs 디렉토리의 모든 Markdown 파일 로드
    docs_dir = "./docs"

    if not os.path.exists(docs_dir):
        logger.warning(f"문서 디렉토리가 없습니다: {docs_dir}")
        return

    documents = processor.load_markdown_files(docs_dir)

    if not documents:
        logger.warning("로드된 문서가 없습니다.")
        return

    # 2. 문서 분할 (청킹)
    chunks = processor.split_documents(documents)

    # 3. Vector DB 생성 및 저장
    vectorstore = processor.create_vectorstore(
        documents=chunks,
        collection_name="childcare_knowledge"
    )

    # 4. 테스트 검색
    test_query = "달빛어린이병원이 뭐야?"
    results = processor.search_similar_documents(
        query=test_query,
        collection_name="childcare_knowledge",
        k=3
    )

    logger.info(f"\n[테스트 검색] '{test_query}'")
    for i, doc in enumerate(results, 1):
        logger.info(f"\n[결과 {i}]")
        logger.info(f"내용: {doc.page_content[:200]}...")
        logger.info(f"메타데이터: {doc.metadata}")


if __name__ == "__main__":
    # 육아 문서 처리 실행
    process_childcare_documents()
