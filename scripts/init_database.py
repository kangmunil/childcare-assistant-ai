import pandas as pd
import sqlite3
import os

DB_PATH = "data/childcare.db"
EXCEL_PATH = "babyData/성장도표+데이터+테이블.xls"
SCHEMA_PATH = "scripts/create_tables.sql"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, 'r') as f:
        conn.executescript(f.read())
    return conn

def clean_df(df, chart_type):
    # The first row often contains percentile labels (1st, 3rd, etc.)
    # We should skip it if it's not data
    if pd.isna(df.iloc[0]['성별']):
        df = df.iloc[1:].copy()
    
    # Drop rows where essential data is missing
    df = df.dropna(subset=['성별', 'L', 'M', 'S'])
    
    return df

def map_and_insert(conn, df, chart_type, column_mapping):
    cursor = conn.cursor()
    
    for _, row in df.iterrows():
        data = {
            'chart_type': chart_type,
            'gender': int(row['성별']),
            'l': float(row['L']),
            'm': float(row['M']),
            's': float(row['S']),
        }
        
        # Age or Height mapping
        if 'age_months' in column_mapping:
            data['age_months'] = float(row[column_mapping['age_months']])
        if 'height_cm' in column_mapping:
            data['height_cm'] = float(row[column_mapping['height_cm']])
            
        # Percentile mapping
        percentiles = [1, 3, 5, 10, 15, 25, 50, 75, 85, 90, 95, 97, 99]
        # We need to find which column corresponds to which percentile
        # Based on inspection, percentiles start from index 6 (or similar)
        # It's better to use column names or positions if they are consistent
        
        # Let's use positions for percentiles as names might be Unnamed
        # Sheet structure: 성별, 나이..., L, M, S, P1, P3, P5...
        # For age-based: 0:성별, 1:나이(세), 2:나이(개월), 3:L, 4:M, 5:S, 6:P1...
        # For height-based: 0:성별, 1:키, 2:L, 3:M, 4:S, 5:P1...
        
        start_idx = 6 if 'age_months' in column_mapping and '만나이(세)' in df.columns else 5
        if chart_type == 'weight_for_height':
            start_idx = 5
            
        for i, p in enumerate(percentiles):
            col_idx = start_idx + i
            if col_idx < len(row):
                data[f'p{p}'] = float(row.iloc[col_idx])
        
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        query = f"INSERT INTO growth_standards ({columns}) VALUES ({placeholders})"
        cursor.execute(query, list(data.values()))
    
    conn.commit()

def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"Excel file not found: {EXCEL_PATH}")
        return

    conn = init_db()
    xl = pd.ExcelFile(EXCEL_PATH)
    
    sheet_configs = {
        '연령별 신장': {'type': 'height_for_age', 'mapping': {'age_months': '만나이(개월)'}},
        '연령별 체중': {'type': 'weight_for_age', 'mapping': {'age_months': '만나이(개월)'}},
        '연령별 체질량지수': {'type': 'bmi_for_age', 'mapping': {'age_months': '만나이(개월)'}},
        ' 신장별 체중(2세미만)': {'type': 'weight_for_height', 'mapping': {'height_cm': '누운키(cm)'}},
        '신장별 체중(2-3세미만)': {'type': 'weight_for_height', 'mapping': {'height_cm': '선키(cm)'}},
        '신장별 체중(3세 이상)': {'type': 'weight_for_height', 'mapping': {'height_cm': '선키(cm)'}},
        '연령별 머리둘레': {'type': 'head_circumference_for_age', 'mapping': {'age_months': '만나이(개월)'}},
    }
    
    # Create a mapping of stripped sheet names to actual sheet names
    actual_sheets = {str(s).strip(): s for s in xl.sheet_names}
    
    for sheet_key, config in sheet_configs.items():
        stripped_key = sheet_key.strip()
        if stripped_key in actual_sheets:
            sheet_name = actual_sheets[stripped_key]
            print(f"Processing sheet: {sheet_name}")
            df = xl.parse(sheet_name)
            df = clean_df(df, config['type'])
            map_and_insert(conn, df, config['type'], config['mapping'])
        else:
            print(f"Sheet not found: {sheet_key}")
            
    conn.close()
    print("Database initialization complete.")

if __name__ == "__main__":
    main()