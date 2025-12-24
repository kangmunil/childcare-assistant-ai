"""
공공데이터포털 오픈 API 데이터 수집 모듈

이 모듈은 다음 공공 API로부터 데이터를 수집합니다:
1. 전국 어린이집 정보 조회 서비스 (한국사회보장정보원)
2. 달빛어린이병원 정보 (국립중앙의료원)
3. 예방접종 위탁의료기관 정보 (질병관리청)

참고: 문서의 2장 "공공데이터포털 오픈 API 확보 및 정밀 연동 전략" 참조
"""

import os
import time
from typing import List, Dict, Any, Optional
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class PublicAPICollector:
    """
    공공 API 데이터 수집기 베이스 클래스

    주요 기능:
    - API 요청 및 응답 처리
    - XML/JSON 파싱
    - 재시도 로직
    - Rate Limiting (서버 부하 방지)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DATA_GO_KR_API_KEY")
        self.request_delay = float(os.getenv("API_REQUEST_DELAY", "1.0"))
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ChildcareAssistant-DataCollector/1.0'
        })

    def _request_api(
        self,
        url: str,
        params: Dict[str, Any],
        method: str = "GET",
        max_retries: int = 3
    ) -> requests.Response:
        """
        API 요청 실행 (재시도 로직 포함)

        Args:
            url: API 엔드포인트 URL
            params: 요청 파라미터
            method: HTTP 메소드
            max_retries: 최대 재시도 횟수

        Returns:
            Response 객체
        """
        params['serviceKey'] = self.api_key

        for attempt in range(max_retries):
            try:
                logger.debug(f"API 요청 시도 {attempt + 1}/{max_retries}: {url}")

                if method == "GET":
                    response = self.session.get(url, params=params, timeout=30)
                else:
                    response = self.session.post(url, data=params, timeout=30)

                response.raise_for_status()

                # Rate Limiting
                time.sleep(self.request_delay)

                return response

            except requests.exceptions.RequestException as e:
                logger.warning(f"API 요청 실패 (시도 {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"최대 재시도 횟수 초과: {url}")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def _parse_xml_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        XML 응답을 파싱하여 딕셔너리로 변환

        Args:
            response: API 응답 객체

        Returns:
            파싱된 데이터 딕셔너리
        """
        try:
            root = ET.fromstring(response.content)

            # 일반적인 공공 API XML 구조
            # <response>
            #   <header>
            #     <resultCode>00</resultCode>
            #   </header>
            #   <body>
            #     <items>
            #       <item>...</item>
            #     </items>
            #   </body>
            # </response>

            result_code = root.find('.//resultCode')
            if result_code is not None and result_code.text != '00':
                result_msg = root.find('.//resultMsg')
                error_msg = result_msg.text if result_msg is not None else "Unknown error"
                raise ValueError(f"API 오류 응답: {error_msg}")

            items = []
            for item in root.findall('.//item'):
                item_dict = {}
                for child in item:
                    item_dict[child.tag] = child.text
                items.append(item_dict)

            return {
                'result_code': result_code.text if result_code is not None else '00',
                'total_count': len(items),
                'items': items
            }

        except ET.ParseError as e:
            logger.error(f"XML 파싱 오류: {str(e)}")
            logger.debug(f"응답 내용: {response.text[:500]}")
            raise


class ChildcareAPICollector(PublicAPICollector):
    """
    전국 어린이집 정보 조회 서비스

    문서 참조: 2.1. 전국 어린이집 정보 조회 서비스
    API: https://www.data.go.kr/data/15101155/openapi.do
    """

    BASE_URL = "http://apis.data.go.kr/B551011/ChildcareCenterInfo"

    def __init__(self):
        api_key = os.getenv("CHILDCARE_API_KEY") or os.getenv("DATA_GO_KR_API_KEY")
        super().__init__(api_key)

    def fetch_childcare_centers(
        self,
        sido: Optional[str] = None,
        sigungu: Optional[str] = None,
        page_no: int = 1,
        num_of_rows: int = 100
    ) -> List[Dict[str, Any]]:
        """
        어린이집 정보를 조회합니다.

        Args:
            sido: 시도명 (예: "서울특별시")
            sigungu: 시군구명 (예: "강남구")
            page_no: 페이지 번호
            num_of_rows: 한 페이지 결과 수

        Returns:
            어린이집 정보 리스트

        필드 설명 (문서 표 참조):
            - crname: 어린이집 명
            - craddr: 상세 주소
            - crgbname: 어린이집 유형 (국공립/민간/가정/직장)
            - chcrtescnt: 정원
            - crcapacity: 현원
            - telno: 연락처
            - la, lo: 위도, 경도
        """
        params = {
            'pageNo': page_no,
            'numOfRows': num_of_rows,
            'MobileOS': 'ETC',
            'MobileApp': 'ChildcareAssistant',
            '_type': 'xml'
        }

        if sido:
            params['sido'] = sido
        if sigungu:
            params['sigungu'] = sigungu

        try:
            response = self._request_api(
                f"{self.BASE_URL}/getChildcareCenterList",
                params
            )

            parsed_data = self._parse_xml_response(response)

            logger.info(
                f"어린이집 정보 {parsed_data['total_count']}건 수집 완료 "
                f"(시도: {sido or '전체'}, 시군구: {sigungu or '전체'})"
            )

            return parsed_data['items']

        except Exception as e:
            logger.error(f"어린이집 정보 조회 실패: {str(e)}")
            return []

    def fetch_all_childcare_centers(
        self,
        sido: Optional[str] = None,
        sigungu: Optional[str] = None,
        max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """
        페이징을 순회하며 모든 어린이집 정보를 수집합니다.

        Args:
            sido: 시도명
            sigungu: 시군구명
            max_pages: 최대 페이지 수

        Returns:
            전체 어린이집 정보 리스트
        """
        all_centers = []

        for page in range(1, max_pages + 1):
            centers = self.fetch_childcare_centers(sido, sigungu, page)

            if not centers:
                logger.info(f"페이지 {page}에서 데이터 없음. 수집 종료.")
                break

            all_centers.extend(centers)
            logger.info(f"누적 수집: {len(all_centers)}건")

        return all_centers


class MoonlightHospitalCollector(PublicAPICollector):
    """
    달빛어린이병원 정보 조회 서비스

    문서 참조: 2.2. 달빛어린이병원 및 야간/휴일 진료 정보
    API: https://www.data.go.kr/data/15000736/openapi.do
    """

    BASE_URL = "http://apis.data.go.kr/B552657/ErmctInfoInqireService"

    def __init__(self):
        api_key = os.getenv("MOONLIGHT_HOSPITAL_API_KEY") or os.getenv("DATA_GO_KR_API_KEY")
        super().__init__(api_key)

    def fetch_moonlight_hospitals(
        self,
        q0: Optional[str] = None,  # 시도 (예: "서울특별시")
        q1: Optional[str] = None,  # 시군구 (예: "강남구")
        qt: Optional[int] = None,  # 진료요일/시간 코드 (1~8: 월~일/공휴일)
        page_no: int = 1,
        num_of_rows: int = 10
    ) -> List[Dict[str, Any]]:
        """
        달빛어린이병원 정보를 조회합니다.

        Args:
            q0: 주소(시도)
            q1: 주소(시군구)
            qt: 진료요일/시간 (1~8: 월~일/공휴일)
            page_no: 페이지 번호
            num_of_rows: 한 페이지 결과 수

        Returns:
            달빛어린이병원 정보 리스트

        현재 요일 코드 계산 예시:
            qt = datetime.now().isoweekday()  # 1(월)~7(일)
        """
        params = {
            'pageNo': page_no,
            'numOfRows': num_of_rows
        }

        if q0:
            params['Q0'] = q0
        if q1:
            params['Q1'] = q1
        if qt:
            params['QT'] = qt

        try:
            response = self._request_api(
                f"{self.BASE_URL}/getMoonlightListInfo",
                params
            )

            parsed_data = self._parse_xml_response(response)

            logger.info(f"달빛어린이병원 {parsed_data['total_count']}건 수집 완료")

            return parsed_data['items']

        except Exception as e:
            logger.error(f"달빛어린이병원 정보 조회 실패: {str(e)}")
            return []


class VaccineAPICollector(PublicAPICollector):
    """
    예방접종 위탁의료기관 정보 조회 서비스

    문서 참조: 2.3. 예방접종 도우미 및 위탁의료기관 정보
    API: https://www.data.go.kr/data/15084303/openapi.do
    """

    BASE_URL = "http://apis.data.go.kr/1790387/orglist3"

    def __init__(self):
        api_key = os.getenv("KDCA_VACCINE_API_KEY") or os.getenv("DATA_GO_KR_API_KEY")
        super().__init__(api_key)

    def fetch_vaccine_organizations(
        self,
        sido_cd: Optional[str] = None,
        sgg_cd: Optional[str] = None,
        page_no: int = 1,
        num_of_rows: int = 10
    ) -> List[Dict[str, Any]]:
        """
        예방접종 위탁의료기관 목록을 조회합니다.

        Args:
            sido_cd: 시도 코드
            sgg_cd: 시군구 코드
            page_no: 페이지 번호
            num_of_rows: 한 페이지 결과 수

        Returns:
            위탁의료기관 정보 리스트
        """
        params = {
            'pageNo': page_no,
            'numOfRows': num_of_rows
        }

        if sido_cd:
            params['sidoCd'] = sido_cd
        if sgg_cd:
            params['sggCd'] = sgg_cd

        try:
            response = self._request_api(
                f"{self.BASE_URL}/getOrgList3",
                params
            )

            parsed_data = self._parse_xml_response(response)

            logger.info(f"예방접종 위탁의료기관 {parsed_data['total_count']}건 수집 완료")

            return parsed_data['items']

        except Exception as e:
            logger.error(f"예방접종 위탁의료기관 조회 실패: {str(e)}")
            return []


# ========================================
# 통합 수집 파이프라인
# ========================================

class DataCollectionPipeline:
    """
    모든 공공 API 데이터를 수집하고 Supabase에 저장하는 통합 파이프라인
    """

    def __init__(self, supabase_client):
        """
        Args:
            supabase_client: SupabaseClient 인스턴스
        """
        self.supabase = supabase_client
        self.childcare_collector = ChildcareAPICollector()
        self.moonlight_collector = MoonlightHospitalCollector()
        self.vaccine_collector = VaccineAPICollector()

    def collect_and_store_childcare_centers(self, sido: str = None, sigungu: str = None):
        """어린이집 정보 수집 및 저장"""
        logger.info("=== 어린이집 정보 수집 시작 ===")

        centers = self.childcare_collector.fetch_all_childcare_centers(sido, sigungu)

        if centers:
            # Supabase에 저장
            batch_size = int(os.getenv("BATCH_SIZE", "100"))
            for i in range(0, len(centers), batch_size):
                batch = centers[i:i + batch_size]
                try:
                    self.supabase.insert_many("childcare_centers", batch)
                    logger.info(f"배치 {i//batch_size + 1} 저장 완료 ({len(batch)}건)")
                except Exception as e:
                    logger.error(f"배치 저장 실패: {str(e)}")

        logger.info(f"=== 어린이집 정보 수집 완료: 총 {len(centers)}건 ===")
        return centers

    def collect_and_store_moonlight_hospitals(self):
        """달빛어린이병원 정보 수집 및 저장"""
        logger.info("=== 달빛어린이병원 정보 수집 시작 ===")

        hospitals = self.moonlight_collector.fetch_moonlight_hospitals()

        if hospitals:
            try:
                self.supabase.insert_many("moonlight_hospitals", hospitals)
                logger.info(f"달빛어린이병원 정보 저장 완료 ({len(hospitals)}건)")
            except Exception as e:
                logger.error(f"저장 실패: {str(e)}")

        logger.info(f"=== 달빛어린이병원 정보 수집 완료: 총 {len(hospitals)}건 ===")
        return hospitals

    def collect_and_store_vaccine_organizations(self):
        """예방접종 위탁의료기관 정보 수집 및 저장"""
        logger.info("=== 예방접종 위탁의료기관 정보 수집 시작 ===")

        orgs = self.vaccine_collector.fetch_vaccine_organizations()

        if orgs:
            try:
                self.supabase.insert_many("vaccine_organizations", orgs)
                logger.info(f"예방접종 위탁의료기관 정보 저장 완료 ({len(orgs)}건)")
            except Exception as e:
                logger.error(f"저장 실패: {str(e)}")

        logger.info(f"=== 예방접종 위탁의료기관 정보 수집 완료: 총 {len(orgs)}건 ===")
        return orgs

    def run_full_collection(self):
        """전체 데이터 수집 실행"""
        logger.info("=" * 50)
        logger.info("전체 공공 데이터 수집 파이프라인 시작")
        logger.info("=" * 50)

        start_time = time.time()

        # 1. 어린이집 정보
        self.collect_and_store_childcare_centers()

        # 2. 달빛어린이병원
        self.collect_and_store_moonlight_hospitals()

        # 3. 예방접종 위탁의료기관
        self.collect_and_store_vaccine_organizations()

        elapsed_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"전체 데이터 수집 완료 (소요 시간: {elapsed_time:.2f}초)")
        logger.info("=" * 50)


if __name__ == "__main__":
    # 테스트 실행
    logger.info("공공 API 수집기 테스트 시작")

    # 개별 테스트
    childcare = ChildcareAPICollector()
    centers = childcare.fetch_childcare_centers(sido="서울특별시", sigungu="강남구", num_of_rows=5)

    if centers:
        logger.info(f"테스트 결과: {len(centers)}건 수집 성공")
        logger.info(f"첫 번째 어린이집: {centers[0].get('crname', 'N/A')}")
    else:
        logger.warning("테스트 결과: 데이터 수집 실패. API 키를 확인하세요.")
