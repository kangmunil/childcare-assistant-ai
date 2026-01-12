"""
[Childcare Assistant AI - Unified Knowledge Base]
This module consolidates the RAG pipeline, combining advanced ingestion logic (cleaning, metadata, context-aware chunking)
with practical document loading and retrieval capabilities.

It replaces the legacy `ingestion.py` and `document_processor.py`.
"""

import os
import re
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

# LangChain imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader,
    UnstructuredMarkdownLoader
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain.retrievers import EnsembleRetriever # Pending implementation for persisted stores
from src.rag.embeddings import OpenRouterEmbeddings
from src.core.config import settings
from src.safety import safety_manager

# Logging setup
logger = logging.getLogger(__name__)

class MetadataExtractor:
    """
    Analyzes text to extract metadata: target age (months), category, etc.
    """

    CATEGORIES = ["feeding", "sleep", "development", "health"]
    
    KEYWORD_MAP = {
        "feeding": ["수유", "분유", "모유", "이유식", "젖병", "먹이"],
        "sleep": ["수면", "잠투정", "낮잠", "통잠", "수면교육", "재우"],
        "development": ["발달", "뒤집기", "기어가기", "걸음마", "옹알이", "장난감"],
        "health": ["열", "체온", "예방접종", "병원", "응급", "약", "아파", "진료"]
    }

    @staticmethod
    def extract(text: str, source_type: str = "medium") -> Dict[str, Any]:
        metadata = {
            "target_month_start": "UNTAGGED",
            "target_month_end": "UNTAGGED",
            "category": "general",
            "source_reliability": source_type,
            "processed_at": datetime.now().isoformat()
        }

        # 1. Month Extraction (Regex: "N개월", "생후 N개월" etc)
        # matches 0-24 months
        month_matches = re.findall(r'(\d+)\s*개월', text)
        if month_matches:
            months = [int(m) for m in month_matches if 0 <= int(m) <= 36]
            if months:
                metadata["target_month_start"] = min(months)
                metadata["target_month_end"] = max(months)
        
        # 2. Category Classification (Keyword Counting)
        counts = {cat: 0 for cat in MetadataExtractor.CATEGORIES}
        for cat, keywords in MetadataExtractor.KEYWORD_MAP.items():
            for kw in keywords:
                if kw in text:
                    counts[cat] += 1
        
        best_category = max(counts, key=counts.get)
        if counts[best_category] > 0:
            metadata["category"] = best_category

        return metadata


class ChildcareKnowledgeBase:
    """
    Unified class for Document Loading, Ingestion (Cleaning->Metadata->Chunking), and Retrieval.
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIRECTORY
        
        # 1. Setup Embeddings
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self._setup_embeddings()

        # 2. Setup Splitter (Context-Aware Base)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", " "]
        )
        
        logger.info(f"KnowledgeBase Initialized using {self.embedding_model_name}")

    def _setup_embeddings(self):
        if not settings.IS_LOCAL_EMBEDDING:
            if settings.EFFECTIVE_API_KEY == "MISSING_API_KEY":
                logger.warning("API Key not found for embeddings.")

            if "openrouter.ai" in settings.EFFECTIVE_API_BASE:
                logger.info("Using OpenRouter Embeddings")
                self.embeddings = OpenRouterEmbeddings(
                    model=self.embedding_model_name,
                    api_key=settings.EFFECTIVE_API_KEY,
                    base_url=settings.EFFECTIVE_API_BASE
                )
            else:
                logger.info("Using OpenAI Embeddings")
                self.embeddings = OpenAIEmbeddings(
                    model=self.embedding_model_name,
                    openai_api_key=settings.EFFECTIVE_API_KEY,
                    base_url=settings.EFFECTIVE_API_BASE
                )
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("Using Local HuggingFace Embeddings")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

    # =========================================================================
    # Step 1: Document Loading
    # =========================================================================
    def load_documents(self, directory: str) -> List[Document]:
        """Loads PDF, MD, TXT files from a directory."""
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return []

        documents = []
        
        # PDF
        try:
            pdf_loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents.extend(pdf_loader.load())
        except Exception:
            pass # Ignore errors for now or log debug

        # Markdown
        try:
            md_loader = DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
            documents.extend(md_loader.load())
        except Exception:
            pass

        # Text
        try:
            txt_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
            documents.extend(txt_loader.load())
        except Exception:
            pass
            
        logger.info(f"Loaded {len(documents)} raw documents from {directory}")
        return documents

    # =========================================================================
    # Step 2: Processing Pipeline (Cleaning -> Metadata -> Chunking)
    # =========================================================================
    def _preprocess_text(self, text: str) -> str:
        # Remove HTML
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        # Basic PII scrubbing (Simple example)
        text = re.sub(r'\d{6}-\d{7}', '[ID_REMOVED]', text)
        text = re.sub(r'010-\d{4}-\d{4}', '[PHONE_REMOVED]', text)
        # Normalize units
        text = text.replace("cc", "ml").replace("CC", "ml")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _enrich_metadata(self, docs: List[Document]) -> List[Document]:
        enriched = []
        for doc in docs:
            # Safety Filter
            if not safety_manager.content.is_reliable(doc.page_content):
                continue
            
            clean_text = self._preprocess_text(doc.page_content)
            doc.page_content = clean_text # Update content to be clean
            
            meta = MetadataExtractor.extract(clean_text)
            doc.metadata.update(meta)
            enriched.append(doc)
        return enriched

    def _context_aware_chunking(self, docs: List[Document]) -> List[Document]:
        # Split first
        chunks = self.text_splitter.split_documents(docs)
        
        enriched_chunks = []
        for chunk in chunks:
            # Prepend Context from Metadata
            start_m = chunk.metadata.get('target_month_start')
            end_m = chunk.metadata.get('target_month_end')
            category = chunk.metadata.get('category', 'general')
            
            month_str = "월령미상"
            if start_m != "UNTAGGED" and end_m != "UNTAGGED":
                month_str = f"{start_m}~{end_m}개월"
            
            header = f"[{month_str} {category} 정보] "
            
            disclaimer = ""
            if category == "health":
                disclaimer = "\n[주의: 의학적 조언이 아닙니다. 전문의와 상담하세요.]"
            
            chunk.page_content = header + chunk.page_content + disclaimer
            enriched_chunks.append(chunk)
            
        return enriched_chunks

    def ingest_documents(self, directory: str) -> Optional[Chroma]:
        """Runs the full pipeline: Load -> Clean -> Tag -> Chunk -> Index"""
        
        # 1. Load
        raw_docs = self.load_documents(directory)
        if not raw_docs:
            return None

        # 2. Process (Clean & Tag)
        tagged_docs = self._enrich_metadata(raw_docs)
        
        # 3. Chunk with Context
        final_chunks = self._context_aware_chunking(tagged_docs)
        if not final_chunks:
            logger.warning("No valid chunks generated after processing.")
            return None

        # 4. Index
        vectorstore = Chroma.from_documents(
            documents=final_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="childcare_knowledge"
        )
        logger.info(f"Successfully indexed {len(final_chunks)} chunks.")
        return vectorstore

    # =========================================================================
    # Step 3: Retrieval
    # =========================================================================
    def get_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name="childcare_knowledge",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Simple semantic search"""
        try:
            vectorstore = self.get_vectorstore()
            results = vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
