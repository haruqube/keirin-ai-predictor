"""SQLite → Supabase 同期スクリプト

Usage:
    python scripts/sync_to_supabase.py --all                    # 全テーブル全件同期
    python scripts/sync_to_supabase.py --since 2025-03-01       # 指定日以降の差分同期
    python scripts/sync_to_supabase.py --predictions-only --date 20250316  # 予測のみ
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY
from supabase import create_client

BATCH_SIZE = 500

# FK依存順（親→子）
TABLE_ORDER = [
    "riders",
    "races",
    "entries",
    "race_results",
    "race_payouts",
    "predictions",
    "prediction_results",
    "rider_stats",
]

# テーブルごとのカラム定義（SQLite → Supabase upsert用）
# id列はSupabase側でGENERATED ALWAYS AS IDENTITYなので除外
TABLE_COLUMNS = {
    "riders": [
        "rider_id", "name", "prefecture", "class", "period",
        "birth_year", "gender",
    ],
    "races": [
        "race_id", "date", "velodrome", "race_number", "race_name",
        "grade", "round", "bank_length", "weather", "track_condition",
        "rider_count",
    ],
    "entries": [
        "race_id", "rider_id", "frame_number", "bike_number", "rider_name",
        "class", "prefecture", "gear_ratio", "win_rate", "place_rate",
        "avg_competition_score", "line_group", "line_role", "odds", "popularity",
    ],
    "race_results": [
        "race_id", "rider_id", "finish_position", "frame_number", "bike_number",
        "rider_name", "class", "prefecture", "gear_ratio", "finish_time",
        "margin", "last_1lap", "backstretching", "remarks", "odds",
        "popularity", "line_group", "line_role",
    ],
    "race_payouts": [
        "race_id", "nisyatan_combo", "nisyatan_payout", "nisyatan_popularity",
        "nishafuku_combo", "nishafuku_payout",
    ],
    "predictions": [
        "race_id", "rider_id", "predicted_score", "predicted_rank",
        "mark", "confidence", "exacta_odds",
    ],
    "prediction_results": [
        "race_id", "predicted_top1", "predicted_top3", "actual_top1",
        "actual_top3", "top1_hit", "top3_hit",
    ],
    "rider_stats": [
        "rider_id", "period", "class", "win_count", "race_count",
        "top2_count", "top3_count", "avg_score",
    ],
}

# upsertのconflictカラム
CONFLICT_COLUMNS = {
    "riders": "rider_id",
    "races": "race_id",
    "entries": "race_id,rider_id",
    "race_results": "race_id,rider_id",
    "race_payouts": "race_id",
    "predictions": "race_id,rider_id",
    "prediction_results": "race_id",
    "rider_stats": "rider_id,period",
}


def get_sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def fetch_rows(conn: sqlite3.Connection, table: str, columns: list[str],
               since: str | None = None) -> list[dict]:
    """SQLiteからデータ取得。sinceが指定された場合は日付フィルタ付き。"""
    cols = ", ".join(columns)
    query = f"SELECT {cols} FROM {table}"

    if since:
        if table == "races":
            query += f" WHERE date >= '{since}'"
        elif table in ("entries", "race_results", "race_payouts", "predictions", "prediction_results"):
            query += f" WHERE race_id IN (SELECT race_id FROM races WHERE date >= '{since}')"
        elif table == "rider_stats":
            # rider_statsは日付フィルタなし（期別データ）
            pass

    rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]


def fetch_predictions_for_date(conn: sqlite3.Connection, date_str: str) -> dict[str, list[dict]]:
    """特定日のレースに関連する予測データのみ取得。"""
    result = {}
    race_ids = conn.execute(
        "SELECT race_id FROM races WHERE date = ?",
        (f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",)
    ).fetchall()
    race_id_list = [r["race_id"] for r in race_ids]

    if not race_id_list:
        print(f"  日付 {date_str} のレースが見つかりません")
        return result

    placeholders = ",".join(["?"] * len(race_id_list))

    for table in ["predictions", "prediction_results"]:
        cols = ", ".join(TABLE_COLUMNS[table])
        query = f"SELECT {cols} FROM {table} WHERE race_id IN ({placeholders})"
        rows = conn.execute(query, race_id_list).fetchall()
        result[table] = [dict(row) for row in rows]

    return result


def upsert_batch(client, table: str, rows: list[dict], conflict: str):
    """Supabaseへバッチupsert。"""
    if not rows:
        return 0

    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        client.table(table).upsert(
            batch, on_conflict=conflict
        ).execute()
        total += len(batch)
        if i + BATCH_SIZE < len(rows):
            time.sleep(0.5)  # レート制限対応
    return total


def sync_all(client, conn: sqlite3.Connection):
    """全テーブル全件同期。"""
    for table in TABLE_ORDER:
        columns = TABLE_COLUMNS[table]
        rows = fetch_rows(conn, table, columns)
        if rows:
            count = upsert_batch(client, table, rows, CONFLICT_COLUMNS[table])
            print(f"  {table}: {count} 件同期完了")
        else:
            print(f"  {table}: データなし")


def sync_since(client, conn: sqlite3.Connection, since: str):
    """指定日以降の差分同期。"""
    for table in TABLE_ORDER:
        columns = TABLE_COLUMNS[table]
        rows = fetch_rows(conn, table, columns, since=since)
        if rows:
            count = upsert_batch(client, table, rows, CONFLICT_COLUMNS[table])
            print(f"  {table}: {count} 件同期完了")
        else:
            print(f"  {table}: 該当データなし")


def sync_predictions_only(client, conn: sqlite3.Connection, date_str: str):
    """特定日の予測データのみ同期。"""
    data = fetch_predictions_for_date(conn, date_str)
    for table, rows in data.items():
        if rows:
            count = upsert_batch(client, table, rows, CONFLICT_COLUMNS[table])
            print(f"  {table}: {count} 件同期完了")
        else:
            print(f"  {table}: 該当データなし")


def main():
    parser = argparse.ArgumentParser(description="SQLite → Supabase 同期")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="全テーブル全件同期")
    group.add_argument("--since", type=str, help="指定日以降の差分同期 (YYYY-MM-DD)")
    group.add_argument("--predictions-only", action="store_true", help="予測データのみ同期")
    parser.add_argument("--date", type=str, help="対象日 (YYYYMMDD) ※--predictions-only時に必須")

    args = parser.parse_args()

    if args.predictions_only and not args.date:
        parser.error("--predictions-only には --date が必須です")

    supabase_key = SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    if not SUPABASE_URL or not supabase_key:
        print("エラー: SUPABASE_URL と SUPABASE_SERVICE_KEY (または SUPABASE_ANON_KEY) を .env に設定してください")
        sys.exit(1)

    print("Supabase に接続中...")
    client = create_client(SUPABASE_URL, supabase_key)
    conn = get_sqlite_conn()

    try:
        if args.all:
            print("全件同期を開始...")
            sync_all(client, conn)
        elif args.since:
            print(f"{args.since} 以降の差分同期を開始...")
            sync_since(client, conn, args.since)
        elif args.predictions_only:
            print(f"{args.date} の予測データ同期を開始...")
            sync_predictions_only(client, conn, args.date)

        print("同期完了!")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
