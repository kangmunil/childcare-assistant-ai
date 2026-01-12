"""
Supabase 클라이언트 연결 및 데이터베이스 작업 모듈

이 모듈은 Supabase와의 연결을 관리하고,
데이터 CRUD 작업을 위한 헬퍼 함수를 제공합니다.
"""

import os
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from dotenv import load_dotenv
from loguru import logger
import sys

# 환경 변수 로드
load_dotenv()

# 로거 설정
logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
logger.add(
    os.getenv("LOG_FILE_PATH", "./logs/app.log"),
    rotation="500 MB",
    retention="10 days",
    level="DEBUG"
)


class SupabaseClient:
    """
    Supabase 클라이언트 싱글톤 클래스

    Environment Variables:
        SUPABASE_URL: Supabase 프로젝트 URL
        SUPABASE_KEY: Supabase Anon Key (클라이언트용)
        SUPABASE_SERVICE_ROLE_KEY: Service Role Key (서버 사이드용)
    """

    _instance: Optional['SupabaseClient'] = None
    _client: Optional[Client] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, use_service_role: bool = False):
        """
        Args:
            use_service_role: True면 Service Role Key 사용 (RLS 우회, 서버 전용)
        """
        if self._client is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") if use_service_role else os.getenv("SUPABASE_KEY")

            if not url or not key:
                raise ValueError(
                    "SUPABASE_URL과 SUPABASE_KEY 환경 변수가 설정되지 않았습니다. "
                    ".env 파일을 확인하세요."
                )

            self._client = create_client(url, key)
            logger.info(f"Supabase 클라이언트 초기화 완료 (Service Role: {use_service_role})")

    @property
    def client(self) -> Client:
        """Supabase 클라이언트 인스턴스 반환"""
        if self._client is None:
            raise RuntimeError("Supabase 클라이언트가 초기화되지 않았습니다.")
        return self._client

    # ========================================
    # Helper Methods
    # ========================================

    def insert_data(self, table_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 레코드 삽입

        Args:
            table_name: 테이블명
            data: 삽입할 데이터 딕셔너리

        Returns:
            삽입된 레코드
        """
        try:
            response = self.client.table(table_name).insert(data).execute()
            logger.info(f"[INSERT] {table_name} 테이블에 데이터 삽입 완료")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"[INSERT ERROR] {table_name}: {str(e)}")
            raise

    def insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        대량 레코드 삽입 (배치)

        Args:
            table_name: 테이블명
            data_list: 삽입할 데이터 리스트

        Returns:
            삽입된 레코드 리스트
        """
        try:
            response = self.client.table(table_name).insert(data_list).execute()
            logger.info(f"[BATCH INSERT] {table_name} 테이블에 {len(data_list)}건 삽입 완료")
            return response.data
        except Exception as e:
            logger.error(f"[BATCH INSERT ERROR] {table_name}: {str(e)}")
            raise

    def select_data(
        self,
        table_name: str,
        columns: str = "*",
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        데이터 조회

        Args:
            table_name: 테이블명
            columns: 조회할 컬럼 (기본값: "*")
            filters: 필터 조건 딕셔너리 (예: {"gender": "M", "month_age": 12})
            limit: 조회 제한 개수
            order_by: 정렬 기준 컬럼

        Returns:
            조회된 레코드 리스트
        """
        try:
            query = self.client.table(table_name).select(columns)

            # 필터 적용
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)

            # 정렬
            if order_by:
                query = query.order(order_by)

            # 제한
            if limit:
                query = query.limit(limit)

            response = query.execute()
            logger.debug(f"[SELECT] {table_name}: {len(response.data)}건 조회")
            return response.data
        except Exception as e:
            logger.error(f"[SELECT ERROR] {table_name}: {str(e)}")
            raise

    def update_data(
        self,
        table_name: str,
        filters: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        데이터 업데이트

        Args:
            table_name: 테이블명
            filters: 업데이트 대상 필터 조건
            updates: 업데이트할 데이터

        Returns:
            업데이트된 레코드 리스트
        """
        try:
            query = self.client.table(table_name).update(updates)

            for key, value in filters.items():
                query = query.eq(key, value)

            response = query.execute()
            logger.info(f"[UPDATE] {table_name}: {len(response.data)}건 업데이트 완료")
            return response.data
        except Exception as e:
            logger.error(f"[UPDATE ERROR] {table_name}: {str(e)}")
            raise

    def delete_data(self, table_name: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        데이터 삭제

        Args:
            table_name: 테이블명
            filters: 삭제 대상 필터 조건

        Returns:
            삭제된 레코드 리스트
        """
        try:
            query = self.client.table(table_name).delete()

            for key, value in filters.items():
                query = query.eq(key, value)

            response = query.execute()
            logger.warning(f"[DELETE] {table_name}: {len(response.data)}건 삭제 완료")
            return response.data
        except Exception as e:
            logger.error(f"[DELETE ERROR] {table_name}: {str(e)}")
            raise

    def upsert_data(self, table_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upsert (존재하면 업데이트, 없으면 삽입)

        Args:
            table_name: 테이블명
            data: 삽입/업데이트할 데이터

        Returns:
            결과 레코드
        """
        try:
            response = self.client.table(table_name).upsert(data).execute()
            logger.info(f"[UPSERT] {table_name} 테이블 작업 완료")
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"[UPSERT ERROR] {table_name}: {str(e)}")
            raise

    def execute_rpc(self, function_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Supabase RPC (원격 프로시저 호출) 실행

        Args:
            function_name: PostgreSQL 함수명
            params: 함수 파라미터

        Returns:
            함수 실행 결과
        """
        try:
            response = self.client.rpc(function_name, params or {}).execute()
            logger.info(f"[RPC] {function_name} 함수 실행 완료")
            return response.data
        except Exception as e:
            logger.error(f"[RPC ERROR] {function_name}: {str(e)}")
            raise


# 싱글톤 인스턴스 생성 헬퍼
def get_supabase_client(use_service_role: bool = False) -> SupabaseClient:
    """
    Supabase 클라이언트 인스턴스를 반환합니다.

    Args:
        use_service_role: Service Role Key 사용 여부

    Returns:
        SupabaseClient 인스턴스
    """
    return SupabaseClient(use_service_role=use_service_role)


if __name__ == "__main__":
    # 연결 테스트
    try:
        client = get_supabase_client()
        logger.info("Supabase 연결 테스트 성공!")

        # 테이블 리스트 확인 (예시)
        # tables = client.client.table("information_schema.tables").select("*").execute()
        # logger.info(f"사용 가능한 테이블: {len(tables.data)}개")

    except Exception as e:
        logger.error(f"Supabase 연결 실패: {str(e)}")
        logger.info("환경 변수 (.env 파일)를 확인하세요:")
        logger.info("  - SUPABASE_URL")
        logger.info("  - SUPABASE_KEY")
