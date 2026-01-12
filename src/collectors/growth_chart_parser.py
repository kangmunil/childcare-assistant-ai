"""
질병관리청 성장도표 LMS 데이터 파서

문서 참조: 3. 질병관리청 성장도표(LMS) 데이터의 확보 및 정밀 알고리즘 구현

LMS (Lambda-Mu-Sigma) 방법:
- L (Lambda): Box-Cox 변환 계수 (분포의 왜도 보정)
- M (Mu): 중앙값 (50th percentile)
- S (Sigma): 변동계수 (Coefficient of Variation)

Z-Score 계산 공식:
    Z = ((X/M)^L - 1) / (L × S)  (L ≠ 0)
    Z = ln(X/M) / S              (L = 0)

백분위수(Percentile) 변환:
    Z-Score를 표준정규분포의 누적확률분포(CDF)로 변환
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class GrowthChartParser:
    """
    질병관리청 성장도표 LMS 데이터 파서 및 계산기

    데이터 소스:
    - 질병관리청 '2017 소아청소년 성장도표'
    - 공공데이터포털: https://www.data.go.kr/data/15076588/fileData.do
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Args:
            file_path: 성장도표 엑셀 파일 경로 (.xlsx)
        """
        self.file_path = file_path or os.getenv(
            "GROWTH_CHART_FILE_PATH",
            "./data/growth_standards/kdca_growth_chart_2017.xlsx"
        )
        self.lms_data: Optional[pd.DataFrame] = None

    def load_lms_data(self) -> pd.DataFrame:
        """
        LMS 데이터를 엑셀 파일에서 로드합니다.

        Returns:
            LMS 데이터프레임

        예상 컬럼 구조:
            - 성별코드 (Sex): 1 (남아), 2 (여아)
            - 개월수구분코드 (Age_Months): 0 ~ 228 (18세)
            - 영유아성장종류코드 (Measure_Type): 01(몸무게), 02(신장), 03(머리둘레), 04(BMI)
            - L값 (L_value): Lambda
            - M값 (M_value): Mu (중앙값)
            - S값 (S_value): Sigma (변동계수)
        """
        if not os.path.exists(self.file_path):
            logger.error(f"성장도표 파일을 찾을 수 없습니다: {self.file_path}")
            raise FileNotFoundError(
                f"성장도표 파일이 없습니다. 다음 경로를 확인하세요: {self.file_path}\n"
                "공공데이터포털에서 '질병관리청 소아청소년 성장도표' 파일을 다운로드하세요."
            )

        try:
            # 엑셀 파일 로드 (실제 컬럼명은 파일에 따라 다를 수 있음)
            df = pd.read_excel(self.file_path)

            # 컬럼명 표준화 (실제 파일에 맞게 수정 필요)
            # 예시: 한글 컬럼명을 영어로 변환
            column_mapping = {
                '성별코드': 'sex',
                '개월수구분코드': 'age_months',
                '영유아성장종류코드': 'measure_type',
                'L값': 'L',
                'M값': 'M',
                'S값': 'S',
                # 영문 컬럼인 경우
                'Sex': 'sex',
                'Age_Months': 'age_months',
                'Measure_Type': 'measure_type',
                'L_value': 'L',
                'M_value': 'M',
                'S_value': 'S'
            }

            # 컬럼명 변경 (존재하는 것만)
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)

            # 필수 컬럼 확인
            required_cols = ['sex', 'age_months', 'measure_type', 'L', 'M', 'S']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(f"필수 컬럼 누락: {missing_cols}")
                logger.info(f"현재 컬럼: {df.columns.tolist()}")
                logger.info("컬럼명을 확인하고 column_mapping을 수정하세요.")

            self.lms_data = df
            logger.info(f"성장도표 LMS 데이터 로드 완료: {len(df)}건")

            return df

        except Exception as e:
            logger.error(f"LMS 데이터 로딩 실패: {str(e)}")
            raise

    def get_lms_params(
        self,
        gender: str,
        age_months: int,
        measure_type: str = "weight"
    ) -> Optional[Dict[str, float]]:
        """
        특정 성별, 월령, 측정 유형에 대한 LMS 파라미터를 조회합니다.

        Args:
            gender: "M" (남아) 또는 "F" (여아)
            age_months: 월령 (개월 수)
            measure_type: 측정 유형
                - "weight" (01): 몸무게
                - "height" (02): 신장/키
                - "head_circ" (03): 머리둘레
                - "bmi" (04): 체질량지수

        Returns:
            LMS 파라미터 딕셔너리 {"L": float, "M": float, "S": float}
            또는 데이터가 없으면 None
        """
        if self.lms_data is None:
            self.load_lms_data()

        # 측정 유형 코드 매핑
        measure_type_code = {
            "weight": "01",
            "height": "02",
            "head_circ": "03",
            "bmi": "04"
        }.get(measure_type, measure_type)

        # 성별 코드 변환
        sex_code = "1" if gender == "M" else "2"

        # 데이터 조회
        query = (
            (self.lms_data['sex'] == int(sex_code)) &
            (self.lms_data['age_months'] == age_months) &
            (self.lms_data['measure_type'] == measure_type_code)
        )

        result = self.lms_data[query]

        if result.empty:
            logger.warning(
                f"LMS 데이터를 찾을 수 없음: "
                f"성별={gender}, 월령={age_months}, 유형={measure_type}"
            )
            return None

        row = result.iloc[0]
        return {
            "L": float(row['L']),
            "M": float(row['M']),
            "S": float(row['S'])
        }

    def calculate_z_score(
        self,
        value: float,
        L: float,
        M: float,
        S: float
    ) -> float:
        """
        Z-Score (표준점수)를 계산합니다.

        공식:
            Z = ((X/M)^L - 1) / (L × S)  (L ≠ 0)
            Z = ln(X/M) / S              (L = 0)

        Args:
            value: 실제 측정값 (X)
            L: Lambda (Box-Cox Power)
            M: Mu (중앙값)
            S: Sigma (변동계수)

        Returns:
            Z-Score
        """
        if L != 0:
            z_score = ((value / M) ** L - 1) / (L * S)
        else:
            z_score = np.log(value / M) / S

        return z_score

    def z_score_to_percentile(self, z_score: float) -> float:
        """
        Z-Score를 백분위수(Percentile)로 변환합니다.

        Args:
            z_score: Z-Score 값

        Returns:
            백분위수 (0 ~ 100)

        예시:
            Z = 0   → 50th percentile (중앙값)
            Z = 1.88 → 97th percentile (상위 3%)
            Z = -1.88 → 3rd percentile (하위 3%)
        """
        # 표준정규분포의 누적확률분포(CDF)
        percentile = stats.norm.cdf(z_score) * 100
        return percentile

    def assess_growth(
        self,
        value: float,
        gender: str,
        age_months: int,
        measure_type: str = "weight"
    ) -> Dict[str, Any]:
        """
        성장 상태를 평가합니다.

        Args:
            value: 실제 측정값 (예: 몸무게 10.5kg)
            gender: "M" 또는 "F"
            age_months: 월령
            measure_type: 측정 유형

        Returns:
            평가 결과 딕셔너리
            {
                "value": 측정값,
                "z_score": Z-Score,
                "percentile": 백분위수,
                "status": 상태 ("저체중", "정상", "과체중" 등),
                "message": 해석 메시지
            }
        """
        # LMS 파라미터 조회
        lms_params = self.get_lms_params(gender, age_months, measure_type)

        if lms_params is None:
            return {
                "error": "해당 월령의 성장 기준 데이터를 찾을 수 없습니다.",
                "value": value
            }

        # Z-Score 계산
        z_score = self.calculate_z_score(
            value,
            lms_params['L'],
            lms_params['M'],
            lms_params['S']
        )

        # 백분위수 변환
        percentile = self.z_score_to_percentile(z_score)

        # 상태 판정 (WHO 기준)
        if measure_type == "weight":
            if percentile < 3:
                status = "저체중"
                status_prefix = "[주의]"
            elif percentile < 15:
                status = "정상 하한"
                status_prefix = "[낮음]"
            elif percentile <= 85:
                status = "정상"
                status_prefix = "[정상]"
            elif percentile <= 97:
                status = "정상 상한"
                status_prefix = "[높음]"
            else:
                status = "과체중"
                status_prefix = "[주의]"
        elif measure_type == "height":
            if percentile < 3:
                status = "저신장"
                status_prefix = "[주의]"
            elif percentile <= 97:
                status = "정상"
                status_prefix = "[정상]"
            else:
                status = "고신장"
                status_prefix = "[높음]"
        else:
            status = "정상" if 3 <= percentile <= 97 else "주의"
            status_prefix = "[정상]" if status == "정상" else "[주의]"

        # 메시지 생성
        measure_name = {
            "weight": "몸무게",
            "height": "키",
            "head_circ": "머리둘레",
            "bmi": "BMI"
        }.get(measure_type, measure_type)

        message = (
            f"{status_prefix} {measure_name} {value} (월령: {age_months}개월)\n"
            f"백분위수: {percentile:.1f}% (상위 {100 - percentile:.1f}%)\n"
            f"또래 중앙값: {lms_params['M']:.2f}\n"
            f"상태: {status}"
        )

        return {
            "value": value,
            "median": lms_params['M'],
            "z_score": round(z_score, 2),
            "percentile": round(percentile, 1),
            "status": status,
            "message": message,
            "lms_params": lms_params
        }

    def export_to_supabase_format(self) -> List[Dict[str, Any]]:
        """
        LMS 데이터를 Supabase 저장용 포맷으로 변환합니다.

        Returns:
            Supabase 삽입용 레코드 리스트
        """
        if self.lms_data is None:
            self.load_lms_data()

        records = []

        for _, row in self.lms_data.iterrows():
            record = {
                "gender": "M" if row['sex'] == 1 else "F",
                "month_age": int(row['age_months']),
                "measure_type": row['measure_type'],
                "l_value": float(row['L']),
                "m_value": float(row['M']),
                "s_value": float(row['S']),
            }
            records.append(record)

        logger.info(f"Supabase 포맷 변환 완료: {len(records)}건")
        return records


# ========================================
# 월령 계산 유틸리티
# ========================================

def calculate_age_in_months(birth_date, measurement_date=None) -> float:
    """
    만 나이(개월 수)를 정밀하게 계산합니다.

    공식 (질병관리청 지침):
        만 나이(개월) = (측정년도 - 출생년도) × 12
                      + (측정월 - 출생월)
                      + (측정일 - 출생일) / 30.4

    Args:
        birth_date: 생년월일 (datetime.date 또는 datetime.datetime)
        measurement_date: 측정일 (기본값: 오늘)

    Returns:
        만 나이 (개월 수, 소수점 포함)
    """
    from datetime import date, datetime

    if measurement_date is None:
        measurement_date = date.today()

    # datetime을 date로 변환
    if isinstance(birth_date, datetime):
        birth_date = birth_date.date()
    if isinstance(measurement_date, datetime):
        measurement_date = measurement_date.date()

    year_diff = measurement_date.year - birth_date.year
    month_diff = measurement_date.month - birth_date.month
    day_diff = measurement_date.day - birth_date.day

    age_months = year_diff * 12 + month_diff + day_diff / 30.4

    return age_months


if __name__ == "__main__":
    # 테스트 실행
    from datetime import date

    logger.info("성장도표 파서 테스트 시작")

    # 샘플 데이터로 테스트 (실제 파일이 없을 경우 더미 데이터 생성)
    try:
        parser = GrowthChartParser()
        parser.load_lms_data()

        # 예시: 12개월 남아, 몸무게 10.5kg
        result = parser.assess_growth(
            value=10.5,
            gender="M",
            age_months=12,
            measure_type="weight"
        )

        logger.info(f"평가 결과:\n{result['message']}")

    except FileNotFoundError as e:
        logger.warning(str(e))
        logger.info("테스트를 건너뜁니다. 실제 데이터 파일을 준비하세요.")

    # 월령 계산 테스트
    birth = date(2023, 1, 15)
    current = date(2024, 2, 20)
    age_months = calculate_age_in_months(birth, current)
    logger.info(f"월령 계산: {age_months:.2f}개월")
