"""
육아 헬퍼 AI 에이전트 (Agentic RAG)

이 에이전트는 다음 기능을 수행합니다:
1. RAG 검색: 육아 지식 베이스에서 정보 검색
2. Function Calling: 공공 API 호출 (어린이집, 병원, 예방접종)
3. 계산: 성장도표 백분위수 계산
4. 대화: 사용자와 자연스러운 대화
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool, StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field

# 프로젝트 모듈
from src.core.config import settings
from src.prompts.templates import prompt_loader
from src.rag.document_processor import DocumentProcessor
from src.collectors.public_api_collector import (
    ChildcareAPICollector,
    MoonlightHospitalCollector,
    VaccineAPICollector
)
from src.analysis.growth_analyzer import GrowthAnalyzer
from datetime import date, datetime
import json

# load_dotenv() # config.py에서 처리하므로 제거

# ========================================
# Tool 입력 스키마 정의
# ========================================

class ChildcareSearchInput(BaseModel):
    """어린이집 검색 도구 입력"""
    sido: str = Field(description="시도명 (예: 서울특별시)")
    sigungu: str = Field(description="시군구명 (예: 강남구)")


class MoonlightHospitalInput(BaseModel):
    """달빛어린이병원 검색 도구 입력"""
    sido: str = Field(description="시도명 (예: 서울특별시)")
    sigungu: Optional[str] = Field(default=None, description="시군구명 (예: 강남구)")


class GrowthAnalysisInput(BaseModel):
    """성장 분석 도구 입력"""
    gender: str = Field(description="성별 (M: 남성, F: 여성)")
    birth_date: str = Field(description="아이의 생년월일 (YYYY-MM-DD)")
    height: Optional[float] = Field(default=None, description="키 (cm)")
    weight: Optional[float] = Field(default=None, description="몸무게 (kg)")
    head_circ: Optional[float] = Field(default=None, description="머리둘레 (cm)")


# ========================================
# ChildcareAgent 클래스
# ========================================

class ChildcareAgent:
    """
    육아 헬퍼 AI 에이전트

    RAG + Function Calling을 결합한 지능형 육아 비서
    """

    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        vector_collection: str = "childcare_knowledge"
    ):
        """
        Args:
            model_name: LLM 모델명
            temperature: 온도 (0.0~1.0)
            vector_collection: 벡터 DB 컬렉션 이름
        """
        self.model_name = model_name or settings.LLM_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.vector_collection = vector_collection

        # API Key 유효성 검사
        if settings.EFFECTIVE_API_KEY == "MISSING_API_KEY":
            logger.warning("API Key not found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY.")

        # LLM 초기화
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=settings.EFFECTIVE_API_KEY,
            base_url=settings.EFFECTIVE_API_BASE
        )

        # Document Processor 초기화
        self.doc_processor = DocumentProcessor()

        # 공공 API Collectors 초기화
        self.childcare_collector = ChildcareAPICollector()
        self.moonlight_collector = MoonlightHospitalCollector()
        self.vaccine_collector = VaccineAPICollector()

        # Tools 등록
        self.tools = self._create_tools()

        # 에이전트 생성
        self.agent = self._create_agent()

        logger.info(f"ChildcareAgent 초기화 완료")
        logger.info(f"  - 모델: {self.model_name}")
        logger.info(f"  - 도구 개수: {len(self.tools)}개")

    def _create_tools(self) -> List[Tool]:
        """
        에이전트가 사용할 도구들을 생성합니다.

        Returns:
            Tool 리스트
        """
        tools = []

        # 1. RAG 검색 도구
        def search_knowledge_base(query: str) -> str:
            """
            육아 지식 베이스를 검색합니다.

            Args:
                query: 검색 쿼리

            Returns:
                검색 결과 텍스트
            """
            try:
                results = self.doc_processor.search_similar_documents(
                    query=query,
                    collection_name=self.vector_collection,
                    k=3
                )

                if not results:
                    return "관련 정보를 찾을 수 없습니다."

                # 검색 결과를 텍스트로 변환
                answer = "\n\n".join([
                    f"[문서 {i+1}]\n{doc.page_content}"
                    for i, doc in enumerate(results)
                ])

                return answer

            except Exception as e:
                logger.error(f"지식 베이스 검색 오류: {str(e)}")
                return f"검색 중 오류가 발생했습니다: {str(e)}"

        tools.append(Tool(
            name="search_knowledge_base",
            description=(
                "육아 가이드, 병원 정보, 예방접종 정보 등 육아 관련 지식을 검색합니다. "
                "사용자가 '~이 뭐야?', '~에 대해 알려줘' 등 정보를 물어볼 때 사용합니다."
            ),
            func=search_knowledge_base
        ))

        # 2. 어린이집 검색 도구
        def search_childcare_centers(sido: str, sigungu: str) -> str:
            """어린이집을 검색합니다."""
            try:
                centers = self.childcare_collector.fetch_childcare_centers(
                    sido=sido,
                    sigungu=sigungu,
                    num_of_rows=5
                )

                if not centers:
                    return f"{sido} {sigungu}에서 어린이집 정보를 찾을 수 없습니다."

                result = f"{sido} {sigungu}의 어린이집 {len(centers)}곳:\n\n"

                for i, center in enumerate(centers, 1):
                    result += f"{i}. {center.get('crname', 'N/A')}\n"
                    result += f"   - 유형: {center.get('crgbname', 'N/A')}\n"
                    result += f"   - 주소: {center.get('craddr', 'N/A')}\n"
                    result += f"   - 정원: {center.get('chcrtescnt', 'N/A')}명\n"
                    result += f"   - 전화: {center.get('telno', 'N/A')}\n\n"

                return result

            except Exception as e:
                logger.error(f"어린이집 검색 오류: {str(e)}")
                return f"어린이집 검색 중 오류가 발생했습니다: {str(e)}"

        tools.append(StructuredTool.from_function(
            func=search_childcare_centers,
            name="search_childcare_centers",
            description=(
                "특정 지역의 어린이집을 검색합니다. "
                "사용자가 '강남구 어린이집 찾아줘', '우리 동네 어린이집 있어?' 등 질문할 때 사용합니다."
            ),
            args_schema=ChildcareSearchInput
        ))

        # 3. 달빛어린이병원 검색 도구
        def search_moonlight_hospitals(sido: str, sigungu: Optional[str] = None) -> str:
            """달빛어린이병원을 검색합니다."""
            try:
                qt = datetime.now().isoweekday()  # 현재 요일 (1=월, 7=일)

                hospitals = self.moonlight_collector.fetch_moonlight_hospitals(
                    q0=sido,
                    q1=sigungu,
                    qt=qt,
                    num_of_rows=5
                )

                if not hospitals:
                    return f"{sido}에서 달빛어린이병원 정보를 찾을 수 없습니다."

                result = f"{sido}의 달빛어린이병원 {len(hospitals)}곳:\n\n"

                for i, hospital in enumerate(hospitals, 1):
                    result += f"{i}. {hospital.get('dutyName', 'N/A')}\n"
                    result += f"   - 주소: {hospital.get('dutyAddr', 'N/A')}\n"
                    result += f"   - 전화: {hospital.get('dutyTel1', 'N/A')}\n\n"

                return result

            except Exception as e:
                logger.error(f"달빛어린이병원 검색 오류: {str(e)}")
                return f"달빛어린이병원 검색 중 오류가 발생했습니다: {str(e)}"

        tools.append(StructuredTool.from_function(
            func=search_moonlight_hospitals,
            name="search_moonlight_hospitals",
            description=(
                "야간/휴일에 진료하는 달빛어린이병원을 검색합니다. "
                "사용자가 '밤에 아이가 아픈데 병원 있어?', '주말에 문 연 병원 찾아줘' 등 질문할 때 사용합니다."
            ),
            args_schema=MoonlightHospitalInput
        ))

        # 4. 성장 분석 도구
        def analyze_growth(
            gender: str, 
            birth_date: str, 
            height: Optional[float] = None, 
            weight: Optional[float] = None, 
            head_circ: Optional[float] = None
        ) -> str:
            """아이의 성장 상태를 분석합니다."""
            try:
                # 성별 변환 (M/F -> 1/2)
                gender_int = 1 if gender.upper() == 'M' else 2
                
                # 날짜 변환
                b_date = date.fromisoformat(birth_date)
                
                analyzer = GrowthAnalyzer()
                result = analyzer.assess_growth(
                    gender=gender_int,
                    birth_date=b_date,
                    height=height,
                    weight=weight,
                    head_circ=head_circ
                )

                if result["status"] != "success":
                    return "성장 분석 중 오류가 발생했습니다."

                analysis = result["analysis"]
                response = f"분석 결과 (월령: {result['age_months']}개월):\n"
                
                if "height" in analysis:
                    h = analysis["height"]
                    response += f"- 키: {h['value']}cm (백분위: {h['percentile']}%, {h['status']})\n"
                
                if "weight" in analysis:
                    w = analysis["weight"]
                    response += f"- 몸무게: {w['value']}kg (백분위: {w['percentile']}%, {w['status']})\n"
                
                if "weight_for_height" in analysis:
                    wh = analysis["weight_for_height"]
                    response += f"- 비만도(신장별 체중): {wh['status']} (백분위: {wh['percentile']}%)\n"
                
                if "head_circumference" in analysis:
                    hc = analysis["head_circumference"]
                    response += f"- 머리둘레: {hc['value']}cm (백분위: {hc['percentile']}%, {hc['status']})\n"

                return response

            except Exception as e:
                logger.error(f"성장 분석 오류: {str(e)}")
                return f"성장 분석 중 오류가 발생했습니다: {str(e)}"

        tools.append(StructuredTool.from_function(
            func=analyze_growth,
            name="analyze_growth",
            description=(
                "아이의 신체 계측치(키, 몸무게, 머리둘레)를 바탕으로 성장 상태를 분석합니다. "
                "백분위수와 표준 범위를 제공합니다. "
                "입력 시 성별(M/F)과 생년월일(YYYY-MM-DD)이 필수입니다."
            ),
            args_schema=GrowthAnalysisInput
        ))

        logger.info(f"도구 생성 완료: {len(tools)}개")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description[:50]}...")

        return tools

    def _create_agent(self) -> AgentExecutor:
        """
        LangChain 에이전트를 생성합니다.

        Returns:
            AgentExecutor
        """
        # 시스템 프롬프트 로드 (YAML)
        system_prompt = prompt_loader.get_system_prompt("default")

        # 프롬프트 템플릿 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # 에이전트 생성
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # AgentExecutor 생성
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5
        )

        return agent_executor

    def chat(self, user_input: str, chat_history: List = None) -> str:
        """
        사용자와 대화합니다.

        Args:
            user_input: 사용자 입력
            chat_history: 대화 히스토리 (선택사항)

        Returns:
            AI 응답
        """
        try:
            response = self.agent.invoke({
                "input": user_input,
                "chat_history": chat_history or []
            })

            return response["output"]

        except Exception as e:
            logger.error(f"에이전트 실행 오류: {str(e)}")
            return f"죄송합니다. 오류가 발생했습니다: {str(e)}"


# ========================================
# 사용 예시
# ========================================

if __name__ == "__main__":
    # 에이전트 초기화
    agent = ChildcareAgent()

    # 테스트 질문들
    test_questions = [
        "달빛어린이병원이 뭐야?",
        "서울특별시 강남구 어린이집 찾아줘",
        "12개월 남자아이 몸무게가 10.5kg인데 정상이야?",
    ]

    for question in test_questions:
        logger.info(f"\n[질문] {question}")
        answer = agent.chat(question)
        logger.info(f"[답변] {answer}\n")
        logger.info("=" * 80)
