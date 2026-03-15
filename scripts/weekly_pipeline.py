"""週次パイプライン

--predict: 予測 → 記事生成 → X投稿
--result:  結果取得 → 精度評価 → 結果報告X投稿
"""

import sys
import argparse
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def get_weekend_dates() -> list[str]:
    """直近の土日の日付を返す (YYYYMMDD形式)"""
    today = datetime.now()
    dates = []
    # 今日が土日ならその日を含む
    if today.weekday() == 5:  # 土曜
        dates = [today.strftime("%Y%m%d")]
    elif today.weekday() == 6:  # 日曜
        dates = [
            (today - timedelta(days=1)).strftime("%Y%m%d"),
            today.strftime("%Y%m%d"),
        ]
    else:
        # 次の土日
        days_until_sat = (5 - today.weekday()) % 7
        if days_until_sat == 0:
            days_until_sat = 7
        sat = today + timedelta(days=days_until_sat)
        sun = sat + timedelta(days=1)
        dates = [sat.strftime("%Y%m%d"), sun.strftime("%Y%m%d")]
    return dates


def run_predict_pipeline(dates: list[str], velodromes: str = "major"):
    """予測パイプライン: 予測 → 記事生成 → X投稿"""
    from scripts.predict_races import predict_races
    from scripts.generate_article import generate_article
    from publishing.x_poster import XPoster

    poster = XPoster()

    for date in dates:
        print(f"\n{'='*60}")
        print(f"  予測パイプライン: {date}")
        print(f"{'='*60}")

        # 1. レース予測
        print("\n[Step 1] レース予測...")
        predict_races(date, velodromes)

        # 2. 記事生成
        print("\n[Step 2] 記事生成...")
        result = generate_article(date)

        # 3. X投稿
        if result and poster.is_configured:
            print("\n[Step 3] X投稿...")
            poster.post(result["teaser"])
        elif result:
            print("\n[Step 3] X API未設定。投稿スキップ。")
            print(f"投稿内容:\n{result['teaser']}")
        else:
            print("\n[Step 3] 記事生成失敗。X投稿スキップ。")

    print(f"\n{'='*60}")
    print(f"  パイプライン完了")
    print(f"{'='*60}")


def run_result_pipeline(dates: list[str]):
    """結果パイプライン: 結果スクレイピング → 精度評価 → 結果報告"""
    from data.scraper import KeirinScraper
    from db.schema import get_connection
    from backtest.evaluator import evaluate_predictions
    from publishing.x_poster import XPoster

    scraper = KeirinScraper()
    conn = get_connection()
    poster = XPoster()

    for date in dates:
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        print(f"\n{'='*60}")
        print(f"  結果確認: {date}")
        print(f"{'='*60}")

        # 1. 予測済みレースの結果を取得
        pred_races = conn.execute("""
            SELECT DISTINCT p.race_id
            FROM predictions p
            JOIN races r ON p.race_id = r.race_id
            WHERE r.date = ?
        """, (formatted_date,)).fetchall()

        if not pred_races:
            print(f"  {date} の予測データなし")
            continue

        print(f"\n[Step 1] 結果取得 ({len(pred_races)}レース)...")
        for pr in pred_races:
            race_id = pr["race_id"]
            existing = conn.execute(
                "SELECT COUNT(*) as cnt FROM race_results WHERE race_id = ?",
                (race_id,)
            ).fetchone()["cnt"]
            if existing > 0:
                continue
            try:
                result_data = scraper.scrape_race_result(race_id)
                if result_data:
                    from db.schema import insert_result
                    for rr in result_data.get("results", []):
                        rr["race_id"] = race_id
                        insert_result(conn, rr)
                    conn.commit()
            except Exception as e:
                logger.warning("結果取得エラー %s: %s", race_id, e)

        # 2. 精度評価
        print(f"\n[Step 2] 精度評価...")
        stats = evaluate_predictions(formatted_date)

        # 3. 結果報告X投稿
        if stats["total_races"] > 0:
            top1_pct = stats["top1_hits"] / stats["total_races"] * 100
            top3_pct = stats["top3_overlap"] / (stats["total_races"] * 3) * 100
            display_date = f"{int(date[4:6])}/{int(date[6:8])}"

            result_text = (
                f"{display_date} AI競輪予想 結果報告\n\n"
                f"対象: {stats['total_races']}R\n"
                f"◎的中率: {top1_pct:.1f}%\n"
                f"Top3的中: {top3_pct:.1f}%\n\n"
                f"#競輪予想 #AI予想 #競輪"
            )

            if poster.is_configured:
                print(f"\n[Step 3] 結果報告X投稿...")
                poster.post_result(result_text)
            else:
                print(f"\n[Step 3] X API未設定。投稿スキップ。")
                print(f"投稿内容:\n{result_text}")

    conn.close()
    print(f"\n{'='*60}")
    print(f"  結果確認完了")
    print(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="競輪AI予想 週次パイプライン")
    parser.add_argument("--predict", action="store_true", help="予測パイプライン実行")
    parser.add_argument("--result", action="store_true", help="結果確認パイプライン実行")
    parser.add_argument("--date", help="対象日 (YYYYMMDD, カンマ区切りで複数可)")
    parser.add_argument("--velodromes", default="major",
                        help="場コード or 'all' or 'major' (予測時のみ)")
    args = parser.parse_args()

    if args.date:
        dates = [d.strip() for d in args.date.split(",")]
    else:
        dates = get_weekend_dates()

    if not args.predict and not args.result:
        print("--predict または --result を指定してください")
        print("例: python scripts/weekly_pipeline.py --predict --date 20260309")
        sys.exit(1)

    if args.predict:
        run_predict_pipeline(dates, args.velodromes)

    if args.result:
        run_result_pipeline(dates)
