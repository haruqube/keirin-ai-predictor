"""entries テーブルの2024-2025年データをバックフィル（6並列版）

出走表ページから競走得点・勝率・連対率・ギア倍率等を取得し、
学習データに新特徴量を追加する。
"""

import sys
import time
import logging
import sqlite3
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, CACHE_DIR
from data.scraper import KeirinScraper
from db.schema import get_connection, insert_entry, insert_rider

logger = logging.getLogger(__name__)

# 並列数とディレイ
NUM_WORKERS = 6
PER_THREAD_DELAY = 1.0  # 秒（6並列×1.0s = 全体6 req/s）

# 進捗管理
progress_lock = threading.Lock()
progress = {"done": 0, "total": 0, "success": 0, "skip": 0, "error": 0}


def get_backfill_race_ids() -> list[str]:
    """バックフィル対象のrace_idリストを取得"""
    conn = get_connection()
    rows = conn.execute("""
        SELECT DISTINCT r.race_id FROM races r
        WHERE r.date >= '2024-01-01' AND r.date < '2026-01-01'
        AND r.race_id NOT IN (SELECT DISTINCT race_id FROM entries)
        ORDER BY r.race_id
    """).fetchall()
    conn.close()
    return [r["race_id"] for r in rows]


def scrape_single_race(race_id: str, scraper: KeirinScraper) -> dict | None:
    """1レースの出走表をスクレイピング（キャッシュ活用）"""
    try:
        entry_data = scraper.scrape_race_entry(race_id)
        entries = entry_data.get("entries", [])
        if not entries:
            return None
        return entry_data
    except Exception as e:
        logger.debug("Error scraping %s: %s", race_id, e)
        return None


def worker(race_ids: list[str], worker_id: int):
    """ワーカースレッド: 割り当てられたレースをスクレイピングしてDBに保存"""
    # スレッド固有のスクレイパーとDB接続
    scraper = KeirinScraper()
    conn = get_connection()

    for i, race_id in enumerate(race_ids):
        try:
            entry_data = scrape_single_race(race_id, scraper)

            if entry_data and entry_data.get("entries"):
                entries = entry_data["entries"]
                for e in entries:
                    e["race_id"] = race_id
                    if e.get("rider_id"):
                        insert_rider(conn, {
                            "rider_id": e["rider_id"],
                            "name": e.get("rider_name", ""),
                            "class": e.get("class"),
                            "prefecture": e.get("prefecture"),
                        })
                        insert_entry(conn, e)

                conn.commit()

                with progress_lock:
                    progress["success"] += 1
            else:
                with progress_lock:
                    progress["skip"] += 1

        except Exception as e:
            logger.debug("Worker %d error on %s: %s", worker_id, race_id, e)
            with progress_lock:
                progress["error"] += 1

        with progress_lock:
            progress["done"] += 1
            done = progress["done"]
            total = progress["total"]
            if done % 100 == 0 or done == total:
                elapsed = time.time() - progress["start_time"]
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate / 60 if rate > 0 else 0
                logger.info(
                    "進捗: %d/%d (%.0f%%) | 成功:%d スキップ:%d エラー:%d | %.1f req/s | 残り%.0f分",
                    done, total, done / total * 100,
                    progress["success"], progress["skip"], progress["error"],
                    rate, eta
                )

    conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="entries バックフィル（並列版）")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help=f"並列数 (default: {NUM_WORKERS})")
    parser.add_argument("--delay", type=float, default=PER_THREAD_DELAY, help=f"スレッドあたりのディレイ秒 (default: {PER_THREAD_DELAY})")
    parser.add_argument("--limit", type=int, default=0, help="処理件数制限 (0=全件)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # ディレイを設定（グローバル変数を上書き）
    import config
    config.SCRAPE_DELAY = args.delay
    logger.info("ディレイ: %.1f秒/スレッド × %d並列 = 実効%.1f req/s",
                args.delay, args.workers, args.workers / args.delay)

    # バックフィル対象取得
    race_ids = get_backfill_race_ids()
    if args.limit > 0:
        race_ids = race_ids[:args.limit]

    if not race_ids:
        logger.info("バックフィル対象なし")
        return

    logger.info("バックフィル対象: %d レース", len(race_ids))
    est_time = len(race_ids) / (args.workers / args.delay) / 60
    logger.info("推定時間: %.0f分", est_time)

    progress["total"] = len(race_ids)
    progress["start_time"] = time.time()

    # レースIDを均等にワーカーに分配
    chunks = [[] for _ in range(args.workers)]
    for i, rid in enumerate(race_ids):
        chunks[i % args.workers].append(rid)

    # 並列実行
    threads = []
    for worker_id, chunk in enumerate(chunks):
        t = threading.Thread(target=worker, args=(chunk, worker_id), daemon=True)
        threads.append(t)
        t.start()
        time.sleep(0.2)  # スレッド間のオフセット

    for t in threads:
        t.join()

    elapsed = time.time() - progress["start_time"]
    logger.info("=" * 50)
    logger.info("完了! %d レース処理 (%.1f分)", progress["done"], elapsed / 60)
    logger.info("成功: %d, スキップ: %d, エラー: %d",
                progress["success"], progress["skip"], progress["error"])

    # 結果確認
    conn = get_connection()
    r = conn.execute("SELECT COUNT(DISTINCT race_id) FROM entries").fetchone()
    logger.info("entries テーブル: %d レース", r[0])
    conn.close()


if __name__ == "__main__":
    main()
