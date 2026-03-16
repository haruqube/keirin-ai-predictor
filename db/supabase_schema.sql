-- Supabase (PostgreSQL) スキーマ
-- Supabaseダッシュボードの SQL Editor で実行してください

-- レース情報
CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    velodrome TEXT NOT NULL,
    race_number INTEGER NOT NULL,
    race_name TEXT,
    grade TEXT,
    round TEXT,
    bank_length INTEGER,
    weather TEXT,
    track_condition TEXT,
    rider_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 選手マスタ
CREATE TABLE IF NOT EXISTS riders (
    rider_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prefecture TEXT,
    class TEXT,
    period INTEGER,
    birth_year INTEGER,
    gender TEXT DEFAULT 'M',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- レース結果
CREATE TABLE IF NOT EXISTS race_results (
    id BIGSERIAL PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    rider_id TEXT NOT NULL REFERENCES riders(rider_id),
    finish_position INTEGER,
    frame_number INTEGER,
    bike_number INTEGER,
    rider_name TEXT,
    class TEXT,
    prefecture TEXT,
    gear_ratio REAL,
    finish_time TEXT,
    margin TEXT,
    last_1lap REAL,
    backstretching TEXT,
    remarks TEXT,
    odds REAL,
    popularity INTEGER,
    line_group TEXT,
    line_role TEXT,
    UNIQUE(race_id, rider_id)
);

-- 出走表（レース前）
CREATE TABLE IF NOT EXISTS entries (
    id BIGSERIAL PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    rider_id TEXT NOT NULL,
    frame_number INTEGER,
    bike_number INTEGER,
    rider_name TEXT,
    class TEXT,
    prefecture TEXT,
    gear_ratio REAL,
    win_rate REAL,
    place_rate REAL,
    avg_competition_score REAL,
    line_group TEXT,
    line_role TEXT,
    odds REAL,
    popularity INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(race_id, rider_id)
);

-- 選手成績サマリ（期別）
CREATE TABLE IF NOT EXISTS rider_stats (
    id BIGSERIAL PRIMARY KEY,
    rider_id TEXT NOT NULL REFERENCES riders(rider_id),
    period INTEGER NOT NULL,
    class TEXT,
    win_count INTEGER DEFAULT 0,
    race_count INTEGER DEFAULT 0,
    top2_count INTEGER DEFAULT 0,
    top3_count INTEGER DEFAULT 0,
    avg_score REAL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(rider_id, period)
);

-- 予測
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    rider_id TEXT NOT NULL,
    predicted_score REAL,
    predicted_rank INTEGER,
    mark TEXT,
    confidence REAL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(race_id, rider_id)
);

-- 予測結果
CREATE TABLE IF NOT EXISTS prediction_results (
    id BIGSERIAL PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    predicted_top1 TEXT,
    predicted_top3 TEXT,
    actual_top1 TEXT,
    actual_top3 TEXT,
    top1_hit INTEGER DEFAULT 0,
    top3_hit INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(race_id)
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_results_race ON race_results(race_id);
CREATE INDEX IF NOT EXISTS idx_results_rider ON race_results(rider_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
CREATE INDEX IF NOT EXISTS idx_entries_race ON entries(race_id);
CREATE INDEX IF NOT EXISTS idx_predictions_race ON predictions(race_id);
CREATE INDEX IF NOT EXISTS idx_rider_stats_rider ON rider_stats(rider_id);

-- Row Level Security (RLS) - 読み取り専用で公開
ALTER TABLE races ENABLE ROW LEVEL SECURITY;
ALTER TABLE riders ENABLE ROW LEVEL SECURITY;
ALTER TABLE race_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE rider_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_results ENABLE ROW LEVEL SECURITY;

-- anon/authenticated ユーザーに読み取り許可
CREATE POLICY "Allow read access" ON races FOR SELECT USING (true);
CREATE POLICY "Allow read access" ON riders FOR SELECT USING (true);
CREATE POLICY "Allow read access" ON race_results FOR SELECT USING (true);
CREATE POLICY "Allow read access" ON entries FOR SELECT USING (true);
CREATE POLICY "Allow read access" ON rider_stats FOR SELECT USING (true);
CREATE POLICY "Allow read access" ON predictions FOR SELECT USING (true);
CREATE POLICY "Allow read access" ON prediction_results FOR SELECT USING (true);

-- service_role キーでの書き込み許可（スクリプトからの書き込み用）
CREATE POLICY "Allow insert with service role" ON races FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow insert with service role" ON riders FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow insert with service role" ON race_results FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow insert with service role" ON entries FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow insert with service role" ON rider_stats FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow insert with service role" ON predictions FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow insert with service role" ON prediction_results FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow update with service role" ON races FOR UPDATE USING (true);
CREATE POLICY "Allow update with service role" ON riders FOR UPDATE USING (true);
CREATE POLICY "Allow update with service role" ON race_results FOR UPDATE USING (true);
CREATE POLICY "Allow update with service role" ON entries FOR UPDATE USING (true);
CREATE POLICY "Allow update with service role" ON rider_stats FOR UPDATE USING (true);
CREATE POLICY "Allow update with service role" ON predictions FOR UPDATE USING (true);
CREATE POLICY "Allow update with service role" ON prediction_results FOR UPDATE USING (true);
