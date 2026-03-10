import sqlite3
import numpy as np
from scipy.stats import norm
from typing import Dict, Any, Optional
from datetime import date
from pathlib import Path
from loguru import logger

class GrowthAnalyzer:
    _GROWTH_STANDARDS_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS growth_standards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chart_type TEXT NOT NULL,
        gender INTEGER NOT NULL,
        age_months REAL,
        height_cm REAL,
        l REAL,
        m REAL,
        s REAL,
        p1 REAL,
        p3 REAL,
        p5 REAL,
        p10 REAL,
        p15 REAL,
        p25 REAL,
        p50 REAL,
        p75 REAL,
        p85 REAL,
        p90 REAL,
        p95 REAL,
        p97 REAL,
        p99 REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_growth_standards_type_gender
        ON growth_standards(chart_type, gender);
    """

    def __init__(self, db_path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[2]
        resolved = Path(db_path) if db_path else (project_root / "data" / "childcare.db")
        if not resolved.is_absolute():
            resolved = project_root / resolved

        self.db_path = str(resolved)
        self._growth_table_checked = False
        self._growth_table_has_data = False

    def _get_conn(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _normalize_gender(self, gender: Any) -> Any:
        """
        int(1/2), str(M/F), Enum(Gender.MALE/FEMALE) 입력을 모두 지원한다.
        """
        if isinstance(gender, int):
            return gender

        value = getattr(gender, "value", gender)
        if isinstance(value, str):
            upper = value.upper()
            if upper == "M":
                return 1
            if upper == "F":
                return 2

        return gender

    def _ensure_growth_standards_ready(self) -> bool:
        """
        성장 기준 테이블 스키마를 보장하고, 실제 데이터 적재 여부를 캐시한다.
        """
        if self._growth_table_checked:
            return self._growth_table_has_data

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.executescript(self._GROWTH_STANDARDS_SCHEMA_SQL)
            conn.commit()

            cursor.execute("SELECT 1 FROM growth_standards LIMIT 1")
            self._growth_table_has_data = cursor.fetchone() is not None
            self._growth_table_checked = True

            if not self._growth_table_has_data:
                logger.warning(
                    "growth_standards table is present but empty. "
                    "Load growth standards data before running growth analysis. "
                    f"(db_path={self.db_path})"
                )
        except Exception as e:
            self._growth_table_checked = True
            self._growth_table_has_data = False
            logger.error(f"Error initializing growth_standards table: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

        return self._growth_table_has_data

    def calculate_age_in_months(self, birth_date: date, measured_date: date = None) -> float:
        """
        정확한 개월 수 계산 (일 단위 소수점 포함)
        """
        if measured_date is None:
            measured_date = date.today()
        
        diff = measured_date - birth_date
        return round(diff.days / 30.4375, 2) # 평균 한 달 일수 사용

    def get_lms_parameters(self, chart_type: str, gender: int, age_months: float = None, height_cm: float = None) -> Optional[Dict[str, float]]:
        """
        DB에서 가장 가까운 월령/신장의 LMS 파라미터를 조회합니다.
        """
        if not self._ensure_growth_standards_ready():
            return None

        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            if chart_type == 'weight_for_height':
                # 신장별 체중은 가장 가까운 키(height_cm)를 찾음
                query = """
                    SELECT l, m, s, p3, p50, p97 
                    FROM growth_standards 
                    WHERE chart_type = ? AND gender = ? 
                    ORDER BY ABS(height_cm - ?) ASC LIMIT 1
                """
                cursor.execute(query, (chart_type, gender, height_cm))
            else:
                # 연령별 차트는 가장 가까운 월령(age_months)을 찾음
                query = """
                    SELECT l, m, s, p3, p50, p97 
                    FROM growth_standards 
                    WHERE chart_type = ? AND gender = ? 
                    ORDER BY ABS(age_months - ?) ASC LIMIT 1
                """
                cursor.execute(query, (chart_type, gender, age_months))
            
            row = cursor.fetchone()
            if row:
                return {
                    'l': row[0], 'm': row[1], 's': row[2],
                    'p3': row[3], 'p50': row[4], 'p97': row[5]
                }
        except Exception as e:
            logger.error(f"Error fetching LMS parameters: {e}")
        finally:
            conn.close()
        return None

    def calculate_z_score(self, value: float, l: float, m: float, s: float) -> float:
        """
        LMS 공식을 사용하여 Z-Score 계산
        Z = ((X/M)^L - 1) / (L*S)
        """
        if l == 0:
            return np.log(value / m) / s
        else:
            return (pow(value / m, l) - 1) / (l * s)

    def calculate_percentile(self, z_score: float) -> float:
        """
        Z-Score를 백분위수(0~100)로 변환
        """
        if not np.isfinite(z_score):
            return 0.0

        percentile = norm.cdf(z_score) * 100
        # 극단적으로 큰 값이 100.0으로 고정되는 것을 방지해 실측 오차/입력 오류를 완화
        return float(min(max(percentile, 0.0), 99.9))

    def _build_growth_warning(self, metric_name: str, value: float, z_score: float) -> str | None:
        """
        성장치가 표준편차 범위를 크게 벗어나는 경우 사용자 확인 필요 메시지 생성
        """
        if not np.isfinite(z_score):
            return f"{metric_name}의 Z-score 계산이 불안정합니다. 입력값 또는 단위를 확인해주세요."

        if abs(z_score) >= 6.0:
            return (
                f"{metric_name} 값({value})이(가) 성장표준 범위를 많이 벗어납니다. "
                f"입력 단위(키 cm, 몸무게 kg)가 맞는지 다시 확인해주세요."
            )

        if abs(z_score) >= 4.0:
            return (
                f"{metric_name} 값({value})이(가) 표준범위에서 다소 벗어납니다. "
                f"정상 범위인지 다시 확인해주세요."
            )

        return None

    def assess_growth(self, gender: int, birth_date: date, measured_date: date = None, 
                      height: float = None, weight: float = None, head_circ: float = None) -> Dict[str, Any]:
        """
        종합 성장 분석 수행
        """
        gender = self._normalize_gender(gender)
        warnings = []
        if measured_date is None:
            measured_date = date.today()
            
        age_months = self.calculate_age_in_months(birth_date, measured_date)
        results = {
            "age_months": age_months,
            "measured_date": measured_date.isoformat(),
            "analysis": {},
            "warnings": warnings,
            "status": "success"
        }

        if not self._ensure_growth_standards_ready():
            results["status"] = "error"
            results["error_code"] = "growth_standards_not_loaded"
            results["message"] = (
                "성장 기준 데이터가 아직 초기화되지 않았습니다. "
                "childcare-assistant-ai/scripts/init_database.py로 growth_standards 데이터를 먼저 적재해주세요."
            )
            return results

        # 1. 연령별 신장
        if height:
            lms = self.get_lms_parameters('height_for_age', gender, age_months=age_months)
            if lms:
                z = self.calculate_z_score(height, lms['l'], lms['m'], lms['s'])
                p = self.calculate_percentile(z)
                warning = self._build_growth_warning("키", height, z)
                if warning:
                    warnings.append(warning)
                results["analysis"]["height"] = {
                    "value": height,
                    "percentile": round(p, 1),
                    "z_score": round(z, 2),
                    "median": lms['p50'],
                    "status": self._get_status_label(p)
                }

        # 2. 연령별 체중
        if weight:
            lms = self.get_lms_parameters('weight_for_age', gender, age_months=age_months)
            if lms:
                z = self.calculate_z_score(weight, lms['l'], lms['m'], lms['s'])
                p = self.calculate_percentile(z)
                warning = self._build_growth_warning("몸무게", weight, z)
                if warning:
                    warnings.append(warning)
                results["analysis"]["weight"] = {
                    "value": weight,
                    "percentile": round(p, 1),
                    "z_score": round(z, 2),
                    "median": lms['p50'],
                    "status": self._get_status_label(p)
                }

        # 3. 신장별 체중 (영유아 비만도 측정용)
        if height and weight and age_months <= 24: # 보통 24개월 미만에서 중요
            lms = self.get_lms_parameters('weight_for_height', gender, height_cm=height)
            if lms:
                z = self.calculate_z_score(weight, lms['l'], lms['m'], lms['s'])
                p = self.calculate_percentile(z)
                warning = self._build_growth_warning("신장별 체중(몸무게)", weight, z)
                if warning:
                    warnings.append(warning)
                results["analysis"]["weight_for_height"] = {
                    "percentile": round(p, 1),
                    "status": self._get_obesity_label(p)
                }

        # 4. 연령별 머리둘레
        if head_circ:
            lms = self.get_lms_parameters('head_circumference_for_age', gender, age_months=age_months)
            if lms:
                z = self.calculate_z_score(head_circ, lms['l'], lms['m'], lms['s'])
                p = self.calculate_percentile(z)
                warning = self._build_growth_warning("머리둘레", head_circ, z)
                if warning:
                    warnings.append(warning)
                results["analysis"]["head_circumference"] = {
                    "value": head_circ,
                    "percentile": round(p, 1),
                    "status": self._get_head_status_label(p)
                }

        return results

    def _get_status_label(self, percentile: float) -> str:
        if percentile < 3: return "매우 작음 (정밀검사 권고)"
        if percentile < 15: return "작은 편"
        if percentile > 97: return "매우 큼"
        if percentile > 85: return "큰 편"
        return "정상 (보통)"

    def _get_obesity_label(self, percentile: float) -> str:
        if percentile < 5: return "저체중"
        if percentile > 95: return "비만"
        if percentile > 85: return "과체중"
        return "정상"

    def _get_head_status_label(self, percentile: float) -> str:
        if percentile < 3: return "소두증 의심"
        if percentile > 97: return "대두증 의심"
        return "정상"
