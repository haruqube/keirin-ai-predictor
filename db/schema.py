"""DBスキーマ定義・接続管理

環境変数 SUPABASE_URL / SUPABASE_KEY が設定されていれば Supabase (PostgreSQL)、
なければローカル SQLite に接続する。
"""

import os
import sqlite3
from pathlib import Path
from config import DB_PATH


def use_supabase() -> bool:
    """Supabaseを使用するかどうか"""
    return bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_KEY"))


def get_connection():
    """DB接続を返す（SQLite or Supabase互換ラッパー）"""
    if use_supabase():
        from db.supabase_client import get_supabase_client, SupabaseConnection
        client = get_supabase_client()
        return SupabaseConnection(client)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
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
        created_at TEXT DEFAULT (datetime('now'))
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
        created_at TEXT DEFAULT (datetime('now'))
    );

    -- レース結果
    CREATE TABLE IF NOT EXISTS race_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT NOT NULL,
        rider_id TEXT NOT NULL,
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
        FOREIGN KEY (race_id) REFERENCES races(race_id),
        FOREIGN KEY (rider_id) REFERENCES riders(rider_id),
        UNIQUE(race_id, rider_id)
    );

    -- 出走表（レース前）
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT NOT NULL,
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
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (race_id) REFERENCES races(race_id),
        UNIQUE(race_id, rider_id)
    );

    -- 選手成績サマリ（期別）
    CREATE TABLE IF NOT EXISTS rider_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rider_id TEXT NOT NULL,
        period INTEGER NOT NULL,
        class TEXT,
        win_count INTEGER DEFAULT 0,
        race_count INTEGER DEFAULT 0,
        top2_count INTEGER DEFAULT 0,
        top3_count INTEGER DEFAULT 0,
        avg_score REAL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (rider_id) REFERENCES riders(rider_id),
        UNIQUE(rider_id, period)
    );

    -- 予測
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT NOT NULL,
        rider_id TEXT NOT NULL,
        predicted_score REAL,
        predicted_rank INTEGER,
        mark TEXT,
        confidence REAL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (race_id) REFERENCES races(race_id),
        UNIQUE(race_id, rider_id)
    );

    -- 予測結果
    CREATE TABLE IF NOT EXISTS prediction_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT NOT NULL,
        predicted_top1 TEXT,
        predicted_top3 TEXT,
        actual_top1 TEXT,
        actual_top3 TEXT,
        top1_hit INTEGER DEFAULT 0,
        top3_hit INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (race_id) REFERENCES races(race_id),
        UNIQUE(race_id)
    );

    CREATE INDEX IF NOT EXISTS idx_results_race ON race_results(race_id);
    CREATE INDEX IF NOT EXISTS idx_results_rider ON race_results(rider_id);
    CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
    CREATE INDEX IF NOT EXISTS idx_entries_race ON entries(race_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_race ON predictions(race_id);
    CREATE INDEX IF NOT EXISTS idx_rider_stats_rider ON rider_stats(rider_id);
    """)

    conn.commit()
    conn.close()


def insert_race(conn: sqlite3.Connection, race: dict):
    conn.execute("""
        INSERT OR REPLACE INTO races
        (race_id, date, velodrome, race_number, race_name, grade,
         round, bank_length, weather, track_condition, rider_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        race["race_id"], race["date"], race["velodrome"], race["race_number"],
        race.get("race_name"), race.get("grade"), race.get("round"),
        race.get("bank_length"), race.get("weather"),
        race.get("track_condition"), race.get("rider_count"),
    ))


def insert_rider(conn: sqlite3.Connection, rider: dict):
    conn.execute("""
        INSERT OR IGNORE INTO riders
        (rider_id, name, prefecture, class, period, birth_year, gender)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        rider["rider_id"], rider["name"], rider.get("prefecture"),
        rider.get("class"), rider.get("period"),
        rider.get("birth_year"), rider.get("gender", "M"),
    ))


def insert_result(conn: sqlite3.Connection, result: dict):
    conn.execute("""
        INSERT OR REPLACE INTO race_results
        (race_id, rider_id, finish_position, frame_number, bike_number,
         rider_name, class, prefecture, gear_ratio, finish_time, margin,
         last_1lap, backstretching, remarks, odds, popularity,
         line_group, line_role)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result["race_id"], result["rider_id"], result.get("finish_position"),
        result.get("frame_number"), result.get("bike_number"),
        result.get("rider_name"), result.get("class"),
        result.get("prefecture"), result.get("gear_ratio"),
        result.get("finish_time"), result.get("margin"),
        result.get("last_1lap"), result.get("backstretching"),
        result.get("remarks"), result.get("odds"),
        result.get("popularity"), result.get("line_group"),
        result.get("line_role"),
    ))


def insert_entry(conn: sqlite3.Connection, entry: dict):
    conn.execute("""
        INSERT OR REPLACE INTO entries
        (race_id, rider_id, frame_number, bike_number, rider_name,
         class, prefecture, gear_ratio, win_rate, place_rate,
         avg_competition_score, line_group, line_role, odds, popularity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        entry["race_id"], entry["rider_id"], entry.get("frame_number"),
        entry.get("bike_number"), entry.get("rider_name"),
        entry.get("class"), entry.get("prefecture"),
        entry.get("gear_ratio"), entry.get("win_rate"),
        entry.get("place_rate"), entry.get("avg_competition_score"),
        entry.get("line_group"), entry.get("line_role"),
        entry.get("odds"), entry.get("popularity"),
    ))


def insert_prediction(conn: sqlite3.Connection, pred: dict):
    conn.execute("""
        INSERT OR REPLACE INTO predictions
        (race_id, rider_id, predicted_score, predicted_rank, mark, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        pred["race_id"], pred["rider_id"], pred.get("predicted_score"),
        pred.get("predicted_rank"), pred.get("mark"), pred.get("confidence"),
    ))


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
