CREATE TABLE IF NOT EXISTS growth_standards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chart_type TEXT NOT NULL, -- 'height_for_age', 'weight_for_age', 'bmi_for_age', 'weight_for_height', 'head_circumference_for_age'
    gender INTEGER NOT NULL,    -- 1: Male, 2: Female
    age_months REAL,
    height_cm REAL,           -- Used for weight_for_height
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

CREATE INDEX IF NOT EXISTS idx_growth_standards_type_gender ON growth_standards(chart_type, gender);