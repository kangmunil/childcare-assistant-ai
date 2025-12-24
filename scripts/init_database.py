"""
데이터베이스 초기화 및 데이터 수집 메인 스크립트

이 스크립트는 다음 작업을 수행합니다:
1. Supabase 데이터베이스 연결 확인
2. 필요한 테이블 스키마 검증
3. 공공 API 데이터 수집 및 저장
4. 성장도표 LMS 데이터 로드 및 저장

실행 방법:
    python scripts/init_database.py --all
    python scripts/init_database.py --growth-chart-only
    python scripts/init_database.py --public-api-only
"""

import sys
import os
import argparse
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python Path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.database.supabase_client import get_supabase_client
from src.collectors.public_api_collector import DataCollectionPipeline
from src.collectors.growth_chart_parser import GrowthChartParser


class DatabaseInitializer:
    """
    데이터베이스 초기화 및 데이터 수집 관리 클래스
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("육아 헬퍼 프로그램 - 데이터베이스 초기화")
        logger.info("=" * 60)

        try:
            self.supabase = get_supabase_client(use_service_role=True)
            logger.success("[성공] Supabase 연결 성공")
        except Exception as e:
            logger.error(f"[실패] Supabase 연결 실패: {str(e)}")
            logger.info("\n환경 변수를 확인하세요:")
            logger.info("  1. .env 파일이 프로젝트 루트에 있는지 확인")
            logger.info("  2. SUPABASE_URL과 SUPABASE_SERVICE_ROLE_KEY가 설정되어 있는지 확인")
            logger.info("  3. .env.example 파일을 참고하여 .env 파일을 생성하세요")
            raise

    def check_tables(self):
        """
        필요한 테이블이 존재하는지 확인합니다.

        참고: Supabase의 경우 SQL 편집기에서 테이블을 미리 생성해야 합니다.
        """
        logger.info("\n[확인] 데이터베이스 테이블 확인 중...")

        required_tables = [
            "growth_standards",      # 성장도표 LMS 데이터
            "childcare_centers",     # 어린이집 정보
            "moonlight_hospitals",   # 달빛어린이병원
            "vaccine_organizations", # 예방접종 위탁의료기관
            "babies",                # 아기 프로필
            "growth_records",        # 성장 기록
            "daily_logs"             # 일상 로그
        ]

        # 테이블 존재 확인 (간단한 SELECT 쿼리로)
        for table_name in required_tables:
            try:
                result = self.supabase.client.table(table_name).select("*").limit(1).execute()
                logger.success(f"  [확인] {table_name} 테이블 확인")
            except Exception as e:
                logger.warning(f"  [주의] {table_name} 테이블이 없거나 접근할 수 없습니다: {str(e)}")
                logger.info(f"     Supabase 대시보드에서 {table_name} 테이블을 생성하세요.")

    def load_growth_chart_data(self):
        """
        성장도표 LMS 데이터를 로드하여 Supabase에 저장합니다.
        """
        logger.info("\n[데이터] 성장도표 LMS 데이터 로딩 시작...")

        try:
            parser = GrowthChartParser()
            parser.load_lms_data()

            # Supabase 포맷으로 변환
            records = parser.export_to_supabase_format()

            # 기존 데이터 삭제 (선택사항)
            logger.info("  기존 성장도표 데이터를 삭제합니다...")
            try:
                # Supabase에서는 DELETE ALL이 직접 지원되지 않으므로 RPC 함수 사용 권장
                # 또는 특정 조건으로 삭제
                pass
            except Exception as e:
                logger.warning(f"  기존 데이터 삭제 실패 (무시): {str(e)}")

            # 배치 삽입
            batch_size = 500
            total_inserted = 0

            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                try:
                    self.supabase.insert_many("growth_standards", batch)
                    total_inserted += len(batch)
                    logger.info(f"  진행: {total_inserted}/{len(records)}건")
                except Exception as e:
                    logger.error(f"  배치 삽입 실패: {str(e)}")

            logger.success(f"[성공] 성장도표 데이터 로딩 완료: {total_inserted}건")

        except FileNotFoundError as e:
            logger.error(f"[실패] {str(e)}")
            logger.info("\n성장도표 파일 다운로드 방법:")
            logger.info("  1. 공공데이터포털 접속: https://www.data.go.kr/")
            logger.info("  2. '질병관리청 소아청소년 성장도표' 검색")
            logger.info("  3. 엑셀 파일 다운로드 후 data/growth_standards/ 디렉토리에 저장")
            logger.info("  4. .env 파일에 GROWTH_CHART_FILE_PATH 경로 설정")

        except Exception as e:
            logger.error(f"[실패] 성장도표 데이터 로딩 실패: {str(e)}")

    def collect_public_api_data(self):
        """
        공공 API 데이터를 수집하여 Supabase에 저장합니다.
        """
        logger.info("\n[API] 공공 API 데이터 수집 시작...")

        try:
            pipeline = DataCollectionPipeline(self.supabase)

            # 전체 수집 실행
            pipeline.run_full_collection()

            logger.success("[성공] 공공 API 데이터 수집 완료")

        except Exception as e:
            logger.error(f"[실패] 공공 API 데이터 수집 실패: {str(e)}")
            logger.info("\nAPI 키 확인 방법:")
            logger.info("  1. 공공데이터포털 가입: https://www.data.go.kr/")
            logger.info("  2. 필요한 API 활용 신청")
            logger.info("  3. 발급받은 인증키를 .env 파일에 설정")

    def run_full_initialization(self):
        """
        전체 초기화 프로세스를 실행합니다.
        """
        logger.info("\n[시작] 전체 데이터베이스 초기화 시작\n")

        # 1. 테이블 확인
        self.check_tables()

        # 2. 성장도표 데이터 로드
        self.load_growth_chart_data()

        # 3. 공공 API 데이터 수집
        self.collect_public_api_data()

        logger.info("\n" + "=" * 60)
        logger.success("[완료] 데이터베이스 초기화 완료!")
        logger.info("=" * 60)


def main():
    """
    메인 실행 함수
    """
    parser = argparse.ArgumentParser(
        description="육아 헬퍼 프로그램 - 데이터베이스 초기화 및 데이터 수집"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="전체 초기화 실행 (성장도표 + 공공 API)"
    )

    parser.add_argument(
        "--growth-chart-only",
        action="store_true",
        help="성장도표 데이터만 로드"
    )

    parser.add_argument(
        "--public-api-only",
        action="store_true",
        help="공공 API 데이터만 수집"
    )

    parser.add_argument(
        "--check-tables",
        action="store_true",
        help="테이블 존재 여부만 확인"
    )

    args = parser.parse_args()

    # 인자가 없으면 도움말 출력
    if not any(vars(args).values()):
        parser.print_help()
        return

    try:
        initializer = DatabaseInitializer()

        if args.check_tables:
            initializer.check_tables()

        elif args.growth_chart_only:
            initializer.load_growth_chart_data()

        elif args.public_api_only:
            initializer.collect_public_api_data()

        elif args.all:
            initializer.run_full_initialization()

    except KeyboardInterrupt:
        logger.warning("\n\n[중단] 사용자에 의해 중단되었습니다.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n\n[오류] 오류 발생: {str(e)}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
