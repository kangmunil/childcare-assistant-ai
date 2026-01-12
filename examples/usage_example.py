"""
육아 헬퍼 프로그램 - 사용 예시

이 파일은 주요 기능들의 사용 방법을 보여줍니다.
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.supabase_client import get_supabase_client
from src.collectors.growth_chart_parser import GrowthChartParser, calculate_age_in_months
from src.collectors.public_api_collector import ChildcareAPICollector
from loguru import logger


def example_1_supabase_connection():
    """예제 1: Supabase 연결 테스트"""
    logger.info("=" * 60)
    logger.info("예제 1: Supabase 연결 테스트")
    logger.info("=" * 60)

    try:
        client = get_supabase_client()
        logger.success("[성공] Supabase 연결 성공!")

        # 간단한 데이터 조회 테스트
        result = client.select_data("babies", limit=5)
        logger.info(f"babies 테이블 레코드 수: {len(result)}개")

    except Exception as e:
        logger.error(f"[실패] 연결 실패: {str(e)}")


def example_2_add_baby():
    """예제 2: 아기 프로필 추가"""
    logger.info("\n" + "=" * 60)
    logger.info("예제 2: 아기 프로필 추가")
    logger.info("=" * 60)

    try:
        client = get_supabase_client()

        # 아기 정보
        baby_data = {
            "name": "지안",
            "birth_date": "2024-01-15",
            "gender": "M",
            "birth_height": 52.0,
            "birth_weight": 3.5
        }

        # 삽입
        result = client.insert_data("babies", baby_data)
        logger.success(f"[성공] 아기 프로필 생성 완료: {result.get('name')}")

        return result

    except Exception as e:
        logger.error(f"[오류] 오류: {str(e)}")
        return None


def example_3_add_growth_record(baby_id: str):
    """예제 3: 성장 기록 추가 및 분석"""
    logger.info("\n" + "=" * 60)
    logger.info("예제 3: 성장 기록 추가 및 분석")
    logger.info("=" * 60)

    try:
        client = get_supabase_client()
        parser = GrowthChartParser()

        # 성장 데이터
        birth_date = date(2024, 1, 15)
        current_date = date.today()
        age_months = int(calculate_age_in_months(birth_date, current_date))

        height = 75.5  # cm
        weight = 10.2  # kg

        logger.info(f"아기 월령: {age_months}개월")
        logger.info(f"측정값 - 키: {height}cm, 몸무게: {weight}kg")

        # 성장 분석 (몸무게)
        weight_analysis = parser.assess_growth(
            value=weight,
            gender="M",
            age_months=age_months,
            measure_type="weight"
        )

        if "error" not in weight_analysis:
            logger.info(f"\n{weight_analysis['message']}")

            # Supabase에 저장
            growth_record = {
                "baby_id": baby_id,
                "measured_date": current_date.isoformat(),
                "height": height,
                "weight": weight,
                "weight_percentile": weight_analysis['percentile'],
                "weight_z_score": weight_analysis['z_score']
            }

            result = client.insert_data("growth_records", growth_record)
            logger.success("[성공] 성장 기록 저장 완료")
        else:
            logger.warning(weight_analysis['error'])

    except Exception as e:
        logger.error(f"[오류] 오류: {str(e)}")


def example_4_add_daily_log(baby_id: str):
    """예제 4: 일상 로그 추가 (수유 기록)"""
    logger.info("\n" + "=" * 60)
    logger.info("예제 4: 일상 로그 추가 - 수유 기록")
    logger.info("=" * 60)

    try:
        client = get_supabase_client()

        # 수유 로그
        feeding_log = {
            "baby_id": baby_id,
            "log_type": "feeding",
            "recorded_at": datetime.now().isoformat(),
            "details": {
                "type": "breast_milk",
                "duration": 15,  # 분
                "side": "left"
            },
            "memo": "잘 먹었음"
        }

        result = client.insert_data("daily_logs", feeding_log)
        logger.success("[성공] 수유 기록 저장 완료")

        # 수면 로그
        sleep_log = {
            "baby_id": baby_id,
            "log_type": "sleep",
            "recorded_at": (datetime.now() - timedelta(hours=8)).isoformat(),
            "details": {
                "start_time": (datetime.now() - timedelta(hours=8)).isoformat(),
                "end_time": datetime.now().isoformat(),
                "quality": "good"
            },
            "memo": "숙면"
        }

        result = client.insert_data("daily_logs", sleep_log)
        logger.success("[성공] 수면 기록 저장 완료")

    except Exception as e:
        logger.error(f"[오류] 오류: {str(e)}")


def example_5_query_childcare_centers():
    """예제 5: 어린이집 정보 조회"""
    logger.info("\n" + "=" * 60)
    logger.info("예제 5: 공공 API - 어린이집 정보 조회")
    logger.info("=" * 60)

    try:
        collector = ChildcareAPICollector()

        # 서울 강남구 어린이집 조회
        centers = collector.fetch_childcare_centers(
            sido="서울특별시",
            sigungu="강남구",
            num_of_rows=5
        )

        if centers:
            logger.success(f"[성공] 조회 성공: {len(centers)}개 어린이집")

            for i, center in enumerate(centers[:3], 1):
                logger.info(f"\n[{i}] {center.get('crname', 'N/A')}")
                logger.info(f"    주소: {center.get('craddr', 'N/A')}")
                logger.info(f"    유형: {center.get('crgbname', 'N/A')}")
                logger.info(f"    정원: {center.get('chcrtescnt', 'N/A')}명")
        else:
            logger.warning("[주의] 데이터를 가져오지 못했습니다. API 키를 확인하세요.")

    except Exception as e:
        logger.error(f"[오류] 오류: {str(e)}")


def example_6_query_logs():
    """예제 6: 일상 로그 조회 및 분석"""
    logger.info("\n" + "=" * 60)
    logger.info("예제 6: 일상 로그 조회 및 패턴 분석")
    logger.info("=" * 60)

    try:
        client = get_supabase_client()

        # 최근 수유 기록 조회
        feeding_logs = client.select_data(
            table_name="daily_logs",
            columns="*",
            filters={"log_type": "feeding"},
            limit=10,
            order_by="recorded_at.desc"
        )

        logger.info(f"최근 수유 기록: {len(feeding_logs)}건")

        if feeding_logs:
            for log in feeding_logs[:3]:
                logger.info(f"  - {log.get('recorded_at')}: {log.get('details', {})}")

    except Exception as e:
        logger.error(f"[오류] 오류: {str(e)}")


def main():
    """
    모든 예제 실행
    """
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("육아 헬퍼 프로그램 - 사용 예시 시작")
    logger.info("=" * 60)

    # 예제 1: Supabase 연결
    example_1_supabase_connection()

    # 예제 2: 아기 프로필 추가
    # baby = example_2_add_baby()

    # if baby and baby.get('id'):
    #     baby_id = baby['id']

    #     # 예제 3: 성장 기록
    #     example_3_add_growth_record(baby_id)

    #     # 예제 4: 일상 로그
    #     example_4_add_daily_log(baby_id)

    #     # 예제 6: 로그 조회
    #     example_6_query_logs()

    # 예제 5: 공공 API (독립적)
    # example_5_query_childcare_centers()

    logger.info("\n" + "=" * 60)
    logger.success("[완료] 모든 예제 실행 완료!")
    logger.info("=" * 60)
    logger.info("\n주의: 예제 2~4는 실제 데이터를 생성하므로 주석 처리되어 있습니다.")
    logger.info("      사용하려면 주석을 해제하세요.")


if __name__ == "__main__":
    main()
