"""DB初期化 + 過去データ取得（並列版）

戦略:
1. /db/race_program/?kaisai_group_id=YYYYMMDD+JJ で開催プログラムを取得
   → 全race_id(12桁)をリストアップ
2. /db/result/?race_id=XXXXXXXXXXXX で各レース結果を取得
3. 開催日は日付×場コードの総当たりだが、race_programが空なら即スキップ
4. ThreadPoolExecutorでHTTPリクエストを並列化、DB書き込みはLockで直列化
"""

import sys
import logging
import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup

from config import NETKEIRIN_BASE_URL, VELODROME_CODES, CACHE_DIR
from db.schema import init_db, get_connection, insert_race, insert_rider, insert_result
from data.scraper import KeirinScraper

logger = logging.getLogger(__name__)

# DB書き込みロック
db_lock = threading.Lock()


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

    # ライン並び予想をパースして結果に付与
    line_data = scraper._parse_line_formation(soup)
    if line_data:
        bike_to_line = {}
        for line_group_idx, members in enumerate(line_data, 1):
            line_group = str(line_group_idx)
            for pos, member in enumerate(members):
                bike_num = member["bike_number"]
                if pos == 0:
                    role = "自力"
                elif pos == 1:
                    role = "番手"
                else:
                    role = "3番手"
                bike_to_line[bike_num] = {"line_group": line_group, "line_role": role}

        for result in results:
            bn = result.get("bike_number")
            if bn and bn in bike_to_line:
                result["line_group"] = bike_to_line[bn]["line_group"]
                result["line_role"] = bike_to_line[bn]["line_role"]

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


def process_kaisai(date: str, jyo_cd: str, existing_race_ids: set) -> list[dict]:
    """1つの開催(日付×場)を処理し、結果データのリストを返す"""
    scraper = KeirinScraper()
    kaisai_group_id = f"{date}{jyo_cd}"

    race_ids = get_race_ids_from_program(scraper, kaisai_group_id)
    if not race_ids:
        return []

    venue_name = VELODROME_CODES.get(jyo_cd, jyo_cd)
    logger.info("%s %s(%s): %d races", date, venue_name, jyo_cd, len(race_ids))

    results_list = []
    for race_id in race_ids:
        if race_id in existing_race_ids:
            continue

        try:
            data = scrape_db_result(scraper, race_id)
            results = data.get("results", [])
            if not results:
                continue
            results_list.append(data)
        except Exception as e:
            logger.warning("Error scraping %s: %s", race_id, e)
            continue

    return results_list


def save_results_to_db(conn, data_list: list[dict], counters: dict):
    """スクレイピング結果をDBに保存（ロック内で呼ぶ）"""
    for data in data_list:
        race_id = data.get("race_id") or data.get("results", [{}])[0].get("race_id")
        results = data.get("results", [])
        if not results or not race_id:
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

        counters["races"] += 1
        counters["results"] += len(results)


def main():
    parser = argparse.ArgumentParser(description="DB初期化・過去データ取得（並列版）")
    parser.add_argument("--start", type=int, default=2024, help="開始年")
    parser.add_argument("--end", type=int, default=2025, help="終了年")
    parser.add_argument("--velodromes", type=str, default="all",
                        help="場コード(カンマ区切り) or 'all' or 'major'")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="中断再開: YYYYMMDD形式の日付から再開")
    parser.add_argument("--workers", type=int, default=4,
                        help="並列ワーカー数 (default: 4)")
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

    conn = get_connection()

    # 既存race_idを取得してスキップ用setに
    existing = {row[0] for row in conn.execute("SELECT race_id FROM races").fetchall()}
    logger.info("Existing races in DB: %d", len(existing))

    counters = {"races": 0, "results": 0, "programs": 0}

    # 日付ごとにバッチ処理
    for date in generate_dates(args.start, args.end):
        if args.resume_from and date < args.resume_from:
            continue

        # この日付の全場を並列処理
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for jyo_cd in jyo_codes:
                future = executor.submit(process_kaisai, date, jyo_cd, existing)
                futures[future] = jyo_cd

            for future in as_completed(futures):
                jyo_cd = futures[future]
                try:
                    data_list = future.result()
                    if data_list:
                        counters["programs"] += 1
                        with db_lock:
                            save_results_to_db(conn, data_list, counters)
                            # 新規race_idをexistingに追加
                            for d in data_list:
                                rid = d.get("race_id") or d.get("results", [{}])[0].get("race_id")
                                if rid:
                                    existing.add(rid)
                except Exception as e:
                    logger.warning("Error processing %s %s: %s", date, jyo_cd, e)

        conn.commit()

        # 日次ログ
        if counters["programs"] > 0 and counters["programs"] % 20 == 0:
            logger.info("Progress: date=%s programs=%d races=%d results=%d",
                        date, counters["programs"], counters["races"], counters["results"])

    conn.close()
    logger.info("Complete: %d programs, %d races, %d results",
                counters["programs"], counters["races"], counters["results"])


if __name__ == "__main__":
    main()
