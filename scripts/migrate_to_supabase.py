"""SQLite → Supabase データ移行スクリプト

自宅PCのSQLiteデータをSupabaseにアップロードする。
バッチ処理で効率的にアップロード。

Usage:
    # .envにSUPABASE_URL, SUPABASE_KEYを設定してから実行
    python scripts/migrate_to_supabase.py

    # 特定テーブルのみ
    python scripts/migrate_to_supabase.py --tables races,riders

    # バッチサイズ変更（デフォルト500行）
    python scripts/migrate_to_supabase.py --batch-size 200
"""

import sys
import logging
import argparse
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH

logger = logging.getLogger(__name__)

# テーブル定義（依存順）
TABLES = [
    "races",
    "riders",
    "race_results",
    "entries",
    "rider_stats",
    "predictions",
    "prediction_results",
]

# テーブルごとのUPSERT用コンフリクトキー
CONFLICT_KEYS = {
    "races": "race_id",
    "riders": "rider_id",
    "race_results": "race_id,rider_id",
    "entries": "race_id,rider_id",
    "rider_stats": "rider_id,period",
    "predictions": "race_id,rider_id",
    "prediction_results": "race_id",
}

# 移行時にスキップするカラム（auto increment）
SKIP_COLUMNS = {"id"}


def get_sqlite_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB not found: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row["name"] for row in cursor.fetchall() if row["name"] not in SKIP_COLUMNS]


def migrate_table(sqlite_conn: sqlite3.Connection, supabase_client, table: str, batch_size: int):
    """1テーブルのデータをSupabaseに移行"""
    columns = get_table_columns(sqlite_conn, table)
    if not columns:
        logger.warning("Table %s: no columns found, skipping", table)
        return 0

    rows = sqlite_conn.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall()
    total = len(rows)
    if total == 0:
        logger.info("Table %s: empty, skipping", table)
        return 0

    logger.info("Table %s: migrating %d rows...", table, total)

    conflict_key = CONFLICT_KEYS.get(table, "id")
    uploaded = 0

    for i in range(0, total, batch_size):
        batch = rows[i:i + batch_size]
        data = []
        for row in batch:
            record = {}
            for col in columns:
                val = row[col]
                record[col] = val
            data.append(record)

        try:
            supabase_client.table(table).upsert(
                data, on_conflict=conflict_key
            ).execute()
            uploaded += len(batch)
            if uploaded % 1000 == 0 or uploaded == total:
                logger.info("  %s: %d/%d uploaded", table, uploaded, total)
        except Exception as e:
            logger.error("  %s batch %d error: %s", table, i // batch_size, e)
            # 1行ずつリトライ
            for record in data:
                try:
                    supabase_client.table(table).upsert(
                        record, on_conflict=conflict_key
                    ).execute()
                    uploaded += 1
                except Exception as e2:
                    logger.warning("    Skip row: %s", e2)

    return uploaded


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="SQLite → Supabase 移行")
    parser.add_argument("--tables", type=str, default=None,
                        help="移行するテーブル(カンマ区切り)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="バッチサイズ (default: 500)")
    args = parser.parse_args()

    # Supabaseクライアント初期化
    from dotenv import load_dotenv
    load_dotenv()

    import os
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL と SUPABASE_KEY を .env に設定してください")
        sys.exit(1)

    from supabase import create_client
    supabase = create_client(url, key)

    # SQLite接続
    sqlite_conn = get_sqlite_connection()

    tables = args.tables.split(",") if args.tables else TABLES

    total_uploaded = 0
    for table in tables:
        count = migrate_table(sqlite_conn, supabase, table, args.batch_size)
        total_uploaded += count

    sqlite_conn.close()
    logger.info("Migration complete: %d total rows uploaded", total_uploaded)


if __name__ == "__main__":
    main()
