from sqlalchemy.orm import Session
from src.models.domain import GrowthStandard, Gender
from datetime import date
import json

class GrowthAnalyzer:
    def __init__(self, db: Session):
        self.db = db

    def calculate_age_in_months(self, birth_date: date, current_date: date = date.today()) -> int:
        """
        만 나이(개월 수) 계산
        """
        return (current_date.year - birth_date.year) * 12 + (current_date.month - birth_date.month)

    def assess_growth(self, gender: Gender, birth_date: date, height: float = None, weight: float = None):
        """
        신체 계측치를 받아 표준 성장표와 비교 분석합니다.
        """
        months = self.calculate_age_in_months(birth_date)
        
        # 해당 월령/성별의 표준 데이터 조회
        standard = self.db.query(GrowthStandard).filter(
            GrowthStandard.gender == gender,
            GrowthStandard.month_age == months
        ).first()

        if not standard:
            return {"status": "error", "message": "해당 월령의 표준 데이터를 찾을 수 없습니다."}

        result = {
            "month_age": months,
            "analysis": {}
        }

        # 키 분석
        if height:
            diff = height - standard.height_median
            status = "평균"
            if diff > 2.0: status = "큰 편"
            elif diff < -2.0: status = "작은 편"
            
            result["analysis"]["height"] = {
                "value": height,
                "median": standard.height_median,
                "diff": round(diff, 1),
                "status": status,
                "message": f"또래 중앙값({standard.height_median}cm)보다 {abs(round(diff, 1))}cm {'큽니다' if diff > 0 else '작습니다'}."
            }

        # 몸무게 분석
        if weight:
            diff = weight - standard.weight_median
            result["analysis"]["weight"] = {
                "value": weight,
                "median": standard.weight_median,
                "diff": round(diff, 1),
                "message": f"또래 중앙값({standard.weight_median}kg)보다 {abs(round(diff, 1))}kg {'무겁습니다' if diff > 0 else '가볍습니다'}."
            }

        return result

    def check_red_flags(self, log_type: str, value: any):
        """
        위험 징후(Red Flags) 감지 - 룰 베이스
        """
        warnings = []
        
        if log_type == "temperature" and float(value) >= 38.0:
            warnings.append("체온이 38도 이상입니다. 해열제 복용이나 미온수 마사지가 필요할 수 있습니다.")
            
        if log_type == "excretion" and value == "white":
             warnings.append("회색 변(담도폐쇄 의심)은 즉시 진료가 필요합니다.")

        if log_type == "excretion" and value == "red":
             warnings.append("혈변이 의심됩니다. 장중첩증 등의 가능성이 있으니 병원에 방문하세요.")
             
        return warnings
