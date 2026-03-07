"""DB初期化 + 過去データ取得

戦略:
1. /db/race_program/?kaisai_group_id=YYYYMMDD+JJ で開催プログラムを取得
   → 全race_id(12桁)をリストアップ
2. /db/result/?race_id=XXXXXXXXXXXX で各レース結果を取得
3. 開催日は日付×場コードの総当たりだが、race_programが空なら即スキップ
   (1リクエストで12レース分の存在確認ができるので効率的)
"""

import sys
import logging
import argparse
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup

from config import NETKEIRIN_BASE_URL, VELODROME_CODES, CACHE_DIR
from db.schema import init_db, get_connection, insert_race, insert_rider, insert_result
from data.scraper import KeirinScraper

logger = logging.getLogger(__name__)


def get_race_ids_from_program(scraper: KeirinScraper, kaisai_group_id: str) -> list[str]:
    """race_programページから全race_idを抽出"""
    cache_key = f"race_program_{kaisai_group_id}"
    cached = scraper._get_json_cache(cache_key)
    if cached is not None:
        return cached

    url = f"{NETKEIRIN_BASE_URL}/db/race_program/?kaisai_group_id={kaisai_group_id}"
    html = scraper._get(url)

    # race_id= の後の12桁数字を全て抽出
    race_ids = sorted(set(re.findall(r"race_id=(\d{12})", html)))

    scraper._set_json_cache(cache_key, race_ids)
    return race_ids


def scrape_db_result(scraper: KeirinScraper, race_id: str) -> dict:
    """netkeirinの /db/result/ ページからレース結果を取得"""
    cache_key = f"db_result_{race_id}"
    cached = scraper._get_json_cache(cache_key)
    if cached:
        return cached

    url = f"{NETKEIRIN_BASE_URL}/db/result/?race_id={race_id}"
    html = scraper._get(url)
    soup = BeautifulSoup(html, "lxml")

    # 1970年表示 = データなし
    title = soup.select_one("title")
    if title and "1970" in title.get_text():
        return {"race_id": race_id, "results": [], "rider_count": 0}

    race_info = scraper._parse_race_info(soup, race_id)
    results = scraper._parse_result_table(soup, race_id)
    race_info["results"] = results
    race_info["rider_count"] = len(results)

    if results:
        scraper._set_json_cache(cache_key, race_info)

    return race_info


def generate_dates(start_year: int, end_year: int):
    """指定年範囲の全日付を生成"""
    d = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    if end > datetime.now():
        end = datetime.now()
    while d <= end:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="DB初期化・過去データ取得")
    parser.add_argument("--start", type=int, default=2024, help="開始年")
    parser.add_argument("--end", type=int, default=2025, help="終了年")
    parser.add_argument("--velodromes", type=str, default="all",
                        help="場コード(カンマ区切り) or 'all' or 'major'")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="中断再開: YYYYMMDD形式の日付から再開")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    init_db()

    # 場コード選択
    if args.velodromes == "all":
        jyo_codes = list(VELODROME_CODES.keys())
    elif args.velodromes == "major":
        jyo_codes = ["22", "25", "27", "28", "31", "33", "34",
                     "41", "53", "55", "74", "81"]
    else:
        jyo_codes = [c.strip() for c in args.velodromes.split(",")]

    scraper = KeirinScraper()
    conn = get_connection()

    total_races = 0
    total_results = 0
    total_programs = 0

    for date in generate_dates(args.start, args.end):
        if args.resume_from and date < args.resume_from:
            continue

        for jyo_cd in jyo_codes:
            kaisai_group_id = f"{date}{jyo_cd}"

            # race_programでこの日・この場の全race_idを取得
            race_ids = get_race_ids_from_program(scraper, kaisai_group_id)

            if not race_ids:
                continue  # この日この場は開催なし

            total_programs += 1
            logger.info("%s %s(%s): %d races",
                        date, VELODROME_CODES.get(jyo_cd, jyo_cd), jyo_cd, len(race_ids))

            for race_id in race_ids:
                # 既にDBにあるかチェック
                existing = conn.execute(
                    "SELECT race_id FROM races WHERE race_id = ?", (race_id,)
                ).fetchone()
                if existing:
                    continue

                try:
                    data = scrape_db_result(scraper, race_id)
                    results = data.get("results", [])
                    if not results:
                        continue

                    race_data = {
                        "race_id": race_id,
                        "date": data["date"],
                        "velodrome": data["velodrome"],
                        "race_number": data["race_number"],
                        "race_name": data.get("race_name"),
                        "grade": data.get("grade"),
                        "rider_count": len(results),
                    }
                    insert_race(conn, race_data)

                    for r in results:
                        if r.get("rider_id"):
                            insert_rider(conn, {
                                "rider_id": r["rider_id"],
                                "name": r.get("rider_name", ""),
                                "class": r.get("class"),
                                "prefecture": r.get("prefecture"),
                            })
                            insert_result(conn, r)

                    total_races += 1
                    total_results += len(results)

                except Exception as e:
                    logger.warning("Error scraping %s: %s", race_id, e)
                    continue

            conn.commit()

        # 日次ログ
        if total_programs > 0 and total_programs % 10 == 0:
            logger.info("Progress: date=%s programs=%d races=%d results=%d",
                        date, total_programs, total_races, total_results)

    conn.close()
    logger.info("Complete: %d programs, %d races, %d results",
                total_programs, total_races, total_results)


if __name__ == "__main__":
    main()
