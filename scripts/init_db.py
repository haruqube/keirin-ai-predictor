"""DB初期化 + 過去データ取得"""

import sys
import logging
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.schema import init_db, get_connection, insert_race, insert_rider, insert_result
from data.scraper import KeirinScraper
from data.race_calendar import get_kaisai_dates

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DB初期化・過去データ取得")
    parser.add_argument("--start", type=int, default=2023, help="開始年")
    parser.add_argument("--end", type=int, default=2025, help="終了年")
    parser.add_argument("--dry-run", action="store_true", help="DB保存せずに確認のみ")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    init_db()
    logger.info("Database initialized")

    scraper = KeirinScraper()
    conn = get_connection()

    total_races = 0
    total_results = 0

    for year in range(args.start, args.end + 1):
        for month in range(1, 13):
            dates = get_kaisai_dates(year, month)
            logger.info("%d/%02d: %d dates found", year, month, len(dates))

            for date in dates:
                race_ids = scraper.scrape_race_list(date)
                logger.info("  %s: %d races", date, len(race_ids))

                for race_id in race_ids:
                    try:
                        data = scraper.scrape_race_result(race_id)
                        results = data.get("results", [])
                        if not results:
                            continue

                        if not args.dry_run:
                            from config import VELODROME_CODES
                            race_data = {
                                "race_id": race_id,
                                "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                                "velodrome": data.get("velodrome", ""),
                                "race_number": data.get("race_number"),
                                "race_name": data.get("race_name"),
                                "grade": data.get("grade"),
                                "round": data.get("round"),
                                "bank_length": data.get("bank_length"),
                                "weather": data.get("weather"),
                                "track_condition": data.get("track_condition"),
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

                if not args.dry_run:
                    conn.commit()

    conn.close()
    logger.info("Total: %d races, %d results", total_races, total_results)


if __name__ == "__main__":
    main()
