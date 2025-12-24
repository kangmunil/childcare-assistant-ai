-- ========================================
-- 육아 헬퍼 프로그램 - Supabase 테이블 생성 스크립트
-- ========================================
--
-- 사용 방법:
-- 1. Supabase 대시보드 접속 (https://app.supabase.com/)
-- 2. SQL Editor 메뉴로 이동
-- 3. 이 SQL 스크립트 복사 후 실행
--
-- 참고: 문서의 4장 및 5장 데이터 구조 참조
-- ========================================

-- ========================================
-- 1. Enum Types (열거형 타입)
-- ========================================

-- 성별
CREATE TYPE gender_enum AS ENUM ('M', 'F');

-- 수유 타입
CREATE TYPE feeding_type_enum AS ENUM ('breast_milk', 'formula', 'baby_food');

-- 대변 색상
CREATE TYPE stool_color_enum AS ENUM ('golden', 'green', 'black', 'red', 'white');

-- 로그 타입
CREATE TYPE log_type_enum AS ENUM ('feeding', 'sleep', 'excretion', 'temperature', 'memo');


-- ========================================
-- 2. Static Data Tables (기준 정보)
-- ========================================

-- 2.1. 성장도표 LMS 데이터
CREATE TABLE IF NOT EXISTS growth_standards (
    id BIGSERIAL PRIMARY KEY,
    gender gender_enum NOT NULL,
    month_age INTEGER NOT NULL,  -- 월령 (0~228)
    measure_type VARCHAR(10) NOT NULL,  -- '01': 몸무게, '02': 신장, '03': 머리둘레, '04': BMI

    -- LMS 파라미터
    l_value NUMERIC(10, 6) NOT NULL,  -- Lambda (Box-Cox Power)
    m_value NUMERIC(10, 4) NOT NULL,  -- Mu (중앙값)
    s_value NUMERIC(10, 6) NOT NULL,  -- Sigma (변동계수)

    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- 인덱스 (빠른 조회를 위해)
    CONSTRAINT unique_growth_standard UNIQUE (gender, month_age, measure_type)
);

CREATE INDEX idx_growth_standards_lookup ON growth_standards(gender, month_age, measure_type);


-- 2.2. 예방접종 일정
CREATE TABLE IF NOT EXISTS vaccine_schedules (
    id BIGSERIAL PRIMARY KEY,
    disease_name VARCHAR(100) NOT NULL,  -- 대상 감염병 (예: B형간염)
    vaccine_name VARCHAR(100) NOT NULL,  -- 백신명
    dose_number INTEGER NOT NULL,        -- 접종 차수 (1차, 2차, 3차...)
    start_month INTEGER NOT NULL,        -- 권장 접종 시작 월령
    end_month INTEGER NOT NULL,          -- 권장 접종 종료 월령
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- 2.3. 발달 이정표 (K-DST)
CREATE TABLE IF NOT EXISTS development_milestones (
    id BIGSERIAL PRIMARY KEY,
    min_month INTEGER NOT NULL,     -- 검사 권장 시작 월령
    max_month INTEGER NOT NULL,     -- 검사 권장 종료 월령
    category VARCHAR(50) NOT NULL,  -- 대근육, 소근육, 인지, 언어 등
    question TEXT NOT NULL,         -- 질문 내용
    importance_level INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ========================================
-- 3. External Data Tables (공공 API 수집 데이터)
-- ========================================

-- 3.1. 전국 어린이집 정보
CREATE TABLE IF NOT EXISTS childcare_centers (
    id BIGSERIAL PRIMARY KEY,
    crname VARCHAR(200),          -- 어린이집 명
    craddr TEXT,                  -- 상세 주소
    crgbname VARCHAR(50),         -- 어린이집 유형 (국공립/민간/가정/직장)
    chcrtescnt INTEGER,           -- 정원
    crcapacity INTEGER,           -- 현원
    telno VARCHAR(50),            -- 연락처
    faxno VARCHAR(50),            -- 팩스
    la NUMERIC(10, 7),            -- 위도
    lo NUMERIC(10, 7),            -- 경도

    -- 메타데이터
    collected_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- 추가 데이터 (JSON으로 유연하게 저장)
    extra_data JSONB
);

CREATE INDEX idx_childcare_centers_location ON childcare_centers(la, lo);
CREATE INDEX idx_childcare_centers_name ON childcare_centers(crname);


-- 3.2. 달빛어린이병원
CREATE TABLE IF NOT EXISTS moonlight_hospitals (
    id BIGSERIAL PRIMARY KEY,
    dutyName VARCHAR(200),        -- 병원명
    dutyAddr TEXT,                -- 주소
    dutyTel1 VARCHAR(50),         -- 대표전화
    dutyTime1s VARCHAR(10),       -- 월요일 시작 시간
    dutyTime1c VARCHAR(10),       -- 월요일 종료 시간
    -- 화~일 시간 (dutyTime2s ~ dutyTime8c)
    wgs84Lat NUMERIC(10, 7),      -- 위도
    wgs84Lon NUMERIC(10, 7),      -- 경도

    collected_at TIMESTAMPTZ DEFAULT NOW(),
    extra_data JSONB
);

CREATE INDEX idx_moonlight_hospitals_location ON moonlight_hospitals(wgs84Lat, wgs84Lon);


-- 3.3. 예방접종 위탁의료기관
CREATE TABLE IF NOT EXISTS vaccine_organizations (
    id BIGSERIAL PRIMARY KEY,
    orgnm VARCHAR(200),           -- 기관명
    orgAddr TEXT,                 -- 주소
    orgTlno VARCHAR(50),          -- 전화번호

    collected_at TIMESTAMPTZ DEFAULT NOW(),
    extra_data JSONB
);


-- ========================================
-- 4. User Data Tables (사용자 입력 데이터)
-- ========================================

-- 4.1. 아기 프로필
CREATE TABLE IF NOT EXISTS babies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,  -- Supabase Auth 연동

    name VARCHAR(100) NOT NULL,
    birth_date DATE NOT NULL,
    gender gender_enum NOT NULL,

    -- 출생 정보
    birth_height NUMERIC(5, 2),     -- 출생 시 키 (cm)
    birth_weight NUMERIC(5, 3),     -- 출생 시 몸무게 (kg)
    premature_birth BOOLEAN DEFAULT FALSE,  -- 미숙아 여부

    photo_url TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_babies_user_id ON babies(user_id);


-- 4.2. 성장 기록
CREATE TABLE IF NOT EXISTS growth_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    baby_id UUID REFERENCES babies(id) ON DELETE CASCADE,

    measured_date DATE DEFAULT CURRENT_DATE,

    height NUMERIC(5, 2),          -- 키 (cm)
    weight NUMERIC(5, 3),          -- 몸무게 (kg)
    head_circ NUMERIC(5, 2),       -- 머리둘레 (cm)

    -- 분석 결과 캐싱 (계산 부하 감소)
    height_percentile NUMERIC(5, 2),
    weight_percentile NUMERIC(5, 2),
    height_z_score NUMERIC(5, 2),
    weight_z_score NUMERIC(5, 2),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_growth_records_baby_id ON growth_records(baby_id);
CREATE INDEX idx_growth_records_date ON growth_records(measured_date DESC);


-- 4.3. 일상 로그 (수유, 수면, 배변 등)
CREATE TABLE IF NOT EXISTS daily_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    baby_id UUID REFERENCES babies(id) ON DELETE CASCADE,

    log_type log_type_enum NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT NOW(),

    -- 타입별 상세 정보 (JSON으로 유연하게 저장)
    -- feeding: {"type": "breast_milk", "amount": 120, "duration": 15, "side": "left"}
    -- sleep: {"start_time": "2024-01-01 20:00", "end_time": "2024-01-02 06:00", "quality": "good"}
    -- excretion: {"type": "wet", "color": "golden", "consistency": "soft"}
    details JSONB,

    memo TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_daily_logs_baby_id ON daily_logs(baby_id);
CREATE INDEX idx_daily_logs_recorded_at ON daily_logs(recorded_at DESC);
CREATE INDEX idx_daily_logs_type ON daily_logs(log_type);


-- ========================================
-- 5. Row Level Security (RLS) 설정
-- ========================================

-- RLS 활성화
ALTER TABLE babies ENABLE ROW LEVEL SECURITY;
ALTER TABLE growth_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_logs ENABLE ROW LEVEL SECURITY;

-- 정책: 사용자는 자신의 데이터만 조회/수정/삭제 가능
CREATE POLICY "Users can view their own babies"
    ON babies FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own babies"
    ON babies FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own babies"
    ON babies FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own babies"
    ON babies FOR DELETE
    USING (auth.uid() = user_id);

-- Growth Records 정책
CREATE POLICY "Users can view their baby's growth records"
    ON growth_records FOR SELECT
    USING (baby_id IN (SELECT id FROM babies WHERE user_id = auth.uid()));

CREATE POLICY "Users can insert their baby's growth records"
    ON growth_records FOR INSERT
    WITH CHECK (baby_id IN (SELECT id FROM babies WHERE user_id = auth.uid()));

-- Daily Logs 정책
CREATE POLICY "Users can view their baby's logs"
    ON daily_logs FOR SELECT
    USING (baby_id IN (SELECT id FROM babies WHERE user_id = auth.uid()));

CREATE POLICY "Users can insert their baby's logs"
    ON daily_logs FOR INSERT
    WITH CHECK (baby_id IN (SELECT id FROM babies WHERE user_id = auth.uid()));


-- ========================================
-- 6. Functions (헬퍼 함수)
-- ========================================

-- 6.1. 성장 백분위수 계산 함수
CREATE OR REPLACE FUNCTION calculate_growth_percentile(
    p_value NUMERIC,
    p_gender gender_enum,
    p_age_months INTEGER,
    p_measure_type VARCHAR
) RETURNS NUMERIC AS $$
DECLARE
    v_l NUMERIC;
    v_m NUMERIC;
    v_s NUMERIC;
    v_z_score NUMERIC;
    v_percentile NUMERIC;
BEGIN
    -- LMS 파라미터 조회
    SELECT l_value, m_value, s_value
    INTO v_l, v_m, v_s
    FROM growth_standards
    WHERE gender = p_gender
      AND month_age = p_age_months
      AND measure_type = p_measure_type
    LIMIT 1;

    IF v_m IS NULL THEN
        RETURN NULL;  -- 데이터 없음
    END IF;

    -- Z-Score 계산
    IF v_l != 0 THEN
        v_z_score := (POWER(p_value / v_m, v_l) - 1) / (v_l * v_s);
    ELSE
        v_z_score := LN(p_value / v_m) / v_s;
    END IF;

    -- 백분위수 변환 (간단한 근사)
    -- 실제로는 표준정규분포 CDF 사용 (PostgreSQL에서는 별도 함수 필요)
    v_percentile := 50 + (v_z_score * 15);  -- 간단한 선형 근사

    RETURN ROUND(v_percentile, 2);
END;
$$ LANGUAGE plpgsql;


-- 6.2. 월령 계산 함수
CREATE OR REPLACE FUNCTION calculate_age_in_months(
    p_birth_date DATE,
    p_current_date DATE DEFAULT CURRENT_DATE
) RETURNS NUMERIC AS $$
DECLARE
    v_year_diff INTEGER;
    v_month_diff INTEGER;
    v_day_diff INTEGER;
    v_age_months NUMERIC;
BEGIN
    v_year_diff := EXTRACT(YEAR FROM p_current_date) - EXTRACT(YEAR FROM p_birth_date);
    v_month_diff := EXTRACT(MONTH FROM p_current_date) - EXTRACT(MONTH FROM p_birth_date);
    v_day_diff := EXTRACT(DAY FROM p_current_date) - EXTRACT(DAY FROM p_birth_date);

    v_age_months := v_year_diff * 12 + v_month_diff + (v_day_diff / 30.4);

    RETURN v_age_months;
END;
$$ LANGUAGE plpgsql;


-- ========================================
-- 완료
-- ========================================

-- 확인: 생성된 테이블 목록
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_type = 'BASE TABLE'
ORDER BY table_name;
