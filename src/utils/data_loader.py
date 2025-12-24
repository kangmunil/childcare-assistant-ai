import pandas as pd
import json
import os
from sqlalchemy.orm import Session
from src.models.domain import GrowthStandard, VaccineSchedule, Gender
# 실제 환경에서는 db_session을 가져오는 코드가 필요합니다.
# from src.database.connection import get_db

def load_growth_standards(file_path: str, db: Session):
    """
    질병관리청 성장도표 엑셀 파일을 읽어 DB에 적재합니다.
    """
    if not os.path.exists(file_path):
        print(f"[Warning] 파일이 없습니다: {file_path}")
        return

    print(f"Loading Growth Standards from {file_path}...")
    
    # 엑셀 구조에 따라 파라미터 조정 필요 (예시 로직)
    # 보통 질병관리청 데이터는 '개월수', 'L', 'M', 'S' 또는 백분위수 컬럼으로 되어 있음
    try:
        df = pd.read_excel(file_path)
        
        standards = []
        for index, row in df.iterrows():
            # 데이터 전처리 (이 부분은 실제 엑셀 컬럼명에 맞춰 수정되어야 함)
            # 예: row['성별']이 1이면 MALE, 2면 FEMALE 등
            gender_val = Gender.MALE if row.get('Sex') == 1 else Gender.FEMALE
            
            standard = GrowthStandard(
                gender=gender_val,
                month_age=row.get('Age_Months'),
                height_median=row.get('Height_50th'), # 50백분위수
                weight_median=row.get('Weight_50th'),
                head_circ_median=row.get('HeadCirc_50th'),
                data_json=json.dumps(row.to_dict()) # 원본 데이터 전체 보존
            )
            standards.append(standard)
        
        db.add_all(standards)
        db.commit()
        print(f"Successfully loaded {len(standards)} growth standards.")
        
    except Exception as e:
        print(f"[Error] 성장도표 로딩 중 오류 발생: {e}")
        db.rollback()

def load_vaccine_schedule(file_path: str, db: Session):
    """
    예방접종 CSV 파일을 읽어 DB에 적재합니다.
    """
    if not os.path.exists(file_path):
        # 파일이 없으면 기본 데이터를 생성하는 로직으로 대체 가능
        print(f"[Info] 파일이 없어 기본 백신 데이터를 생성합니다.")
        create_default_vaccines(db)
        return

    # CSV 로딩 로직... (생략)

def create_default_vaccines(db: Session):
    """
    파일이 없을 경우를 대비한 하드코딩된 필수 백신 데이터 (샘플)
    """
    defaults = [
        VaccineSchedule(disease_name="B형간염", vaccine_name="HepB", dose_number=1, start_month=0, end_month=0, description="출생 직후 접종"),
        VaccineSchedule(disease_name="B형간염", vaccine_name="HepB", dose_number=2, start_month=1, end_month=1, description="1개월 권장"),
        VaccineSchedule(disease_name="결핵", vaccine_name="BCG", dose_number=1, start_month=0, end_month=1, description="생후 4주 이내"),
        # ... 추가 데이터
    ]
    db.add_all(defaults)
    db.commit()
    print("Default vaccine schedule loaded.")
