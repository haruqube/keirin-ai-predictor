-- ============================================================
-- keirin-ai-predictor: Supabase 初期スキーマ
-- SQLite 7テーブル → PostgreSQL + RLS + prediction_detail ビュー
-- ============================================================

-- ── レース情報 ──
CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,
    date DATE NOT NULL,
    velodrome TEXT NOT NULL,
    velodrome_code TEXT GENERATED ALWAYS AS (substring(race_id from 9 for 2)) STORED,
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

CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
CREATE INDEX IF NOT EXISTS idx_races_date_id ON races(date, race_id);
CREATE INDEX IF NOT EXISTS idx_races_velodrome_code ON races(velodrome_code);

-- ── 選手マスタ ──
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

-- ── レース結果 ──
CREATE TABLE IF NOT EXISTS race_results (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    rider_id TEXT NOT NULL REFERENCES riders(rider_id),
    finish_position INTEGER,
    frame_number INTEGER,
    bike_number INTEGER,
    rider_name TEXT,
    class TEXT,
    prefecture TEXT,
    gear_ratio DOUBLE PRECISION,
    finish_time TEXT,
    margin TEXT,
    last_1lap DOUBLE PRECISION,
    backstretching TEXT,
    remarks TEXT,
    odds DOUBLE PRECISION,
    popularity INTEGER,
    line_group TEXT,
    line_role TEXT,
    UNIQUE(race_id, rider_id)
);

CREATE INDEX IF NOT EXISTS idx_results_race ON race_results(race_id);
CREATE INDEX IF NOT EXISTS idx_results_rider ON race_results(rider_id);
CREATE INDEX IF NOT EXISTS idx_results_rider_race ON race_results(rider_id, race_id);

-- ── 出走表（レース前） ──
CREATE TABLE IF NOT EXISTS entries (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    rider_id TEXT NOT NULL,
    frame_number INTEGER,
    bike_number INTEGER,
    rider_name TEXT,
    class TEXT,
    prefecture TEXT,
    gear_ratio DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    place_rate DOUBLE PRECISION,
    avg_competition_score DOUBLE PRECISION,
    line_group TEXT,
    line_role TEXT,
    odds DOUBLE PRECISION,
    popularity INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(race_id, rider_id)
);

CREATE INDEX IF NOT EXISTS idx_entries_race ON entries(race_id);
CREATE INDEX IF NOT EXISTS idx_entries_race_rider ON entries(race_id, rider_id);

-- ── 選手成績サマリ（期別） ──
CREATE TABLE IF NOT EXISTS rider_stats (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rider_id TEXT NOT NULL REFERENCES riders(rider_id),
    period INTEGER NOT NULL,
    class TEXT,
    win_count INTEGER DEFAULT 0,
    race_count INTEGER DEFAULT 0,
    top2_count INTEGER DEFAULT 0,
    top3_count INTEGER DEFAULT 0,
    avg_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(rider_id, period)
);

CREATE INDEX IF NOT EXISTS idx_rider_stats_rider ON rider_stats(rider_id);

-- ── 予測 ──
CREATE TABLE IF NOT EXISTS predictions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    rider_id TEXT NOT NULL,
    predicted_score DOUBLE PRECISION,
    predicted_rank INTEGER,
    mark TEXT,
    confidence DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(race_id, rider_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_race ON predictions(race_id);

-- ── 予測結果 ──
CREATE TABLE IF NOT EXISTS prediction_results (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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

-- ============================================================
-- RLS (Row Level Security)
-- 全テーブル: anon ユーザーに SELECT のみ許可
-- ============================================================

ALTER TABLE races ENABLE ROW LEVEL SECURITY;
ALTER TABLE riders ENABLE ROW LEVEL SECURITY;
ALTER TABLE race_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE rider_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_select_races" ON races FOR SELECT TO anon USING (true);
CREATE POLICY "anon_select_riders" ON riders FOR SELECT TO anon USING (true);
CREATE POLICY "anon_select_race_results" ON race_results FOR SELECT TO anon USING (true);
CREATE POLICY "anon_select_entries" ON entries FOR SELECT TO anon USING (true);
CREATE POLICY "anon_select_rider_stats" ON rider_stats FOR SELECT TO anon USING (true);
CREATE POLICY "anon_select_predictions" ON predictions FOR SELECT TO anon USING (true);
CREATE POLICY "anon_select_prediction_results" ON prediction_results FOR SELECT TO anon USING (true);

-- service_role は全操作可能（デフォルト）

-- ============================================================
-- prediction_detail ビュー
-- フロントエンドから1クエリで予測詳細を取得
-- ============================================================

CREATE OR REPLACE VIEW prediction_detail AS
SELECT
    p.race_id,
    r.date AS race_date,
    r.velodrome,
    r.velodrome_code,
    r.race_number,
    r.race_name,
    r.grade,
    r.bank_length,
    r.rider_count,
    p.rider_id,
    COALESCE(e.rider_name, ri.name) AS rider_name,
    COALESCE(e.class, ri.class) AS rider_class,
    ri.prefecture,
    e.frame_number,
    e.bike_number,
    e.avg_competition_score,
    e.win_rate,
    e.place_rate,
    e.gear_ratio,
    p.predicted_score,
    p.predicted_rank,
    p.mark,
    p.confidence,
    rr.finish_position,
    rr.last_1lap,
    rr.odds AS result_odds,
    rr.backstretching
FROM predictions p
JOIN races r ON r.race_id = p.race_id
LEFT JOIN riders ri ON ri.rider_id = p.rider_id
LEFT JOIN entries e ON e.race_id = p.race_id AND e.rider_id = p.rider_id
LEFT JOIN race_results rr ON rr.race_id = p.race_id AND rr.rider_id = p.rider_id
ORDER BY r.date DESC, r.race_number, p.predicted_rank;
