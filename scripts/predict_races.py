"""レース予測スクリプト"""

import sys
import logging
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR
from db.schema import get_connection, insert_prediction
from data.scraper import KeirinScraper
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker

logger = logging.getLogger(__name__)

MARKS = ["◎", "○", "▲", "△", "△"]


def predict_races(date: str):
    logging.basicConfig(level=logging.INFO)

    model_path = str(RESULTS_DIR / "model_lgbm.pkl")
    model = LGBMRanker()
    model.load(model_path)
    logger.info("Model loaded from %s", model_path)

    scraper = KeirinScraper()
    builder = FeatureBuilder()
    conn = get_connection()

    race_ids = scraper.scrape_race_list(date)
    logger.info("%s: %d races found", date, len(race_ids))

    for race_id in race_ids:
        try:
            # 出走表を取得
            entry_data = scraper.scrape_race_entry(race_id)
            entries = entry_data.get("entries", [])
            if not entries:
                continue

            # DBに保存
            from db.schema import insert_race, insert_rider, insert_entry
            formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            race_data = {
                "race_id": race_id,
                "date": formatted_date,
                "velodrome": entry_data.get("velodrome", ""),
                "race_number": entry_data.get("race_number"),
                "race_name": entry_data.get("race_name"),
                "grade": entry_data.get("grade"),
                "round": entry_data.get("round"),
                "bank_length": entry_data.get("bank_length"),
                "weather": entry_data.get("weather"),
                "track_condition": entry_data.get("track_condition"),
                "rider_count": len(entries),
            }
            insert_race(conn, race_data)

            for e in entries:
                if e.get("rider_id"):
                    insert_rider(conn, {
                        "rider_id": e["rider_id"],
                        "name": e.get("rider_name", ""),
                        "class": e.get("class"),
                        "prefecture": e.get("prefecture"),
                    })
                    insert_entry(conn, e)

            conn.commit()

            # 特徴量構築 & 予測
            df = builder.build_race_features(race_id, formatted_date)
            if df.empty:
                continue

            feature_cols = builder.feature_names
            X = df[feature_cols].fillna(0)
            df["pred_score"] = model.predict(X).values
            df = df.sort_values("pred_score", ascending=False).reset_index(drop=True)

            velodrome = entry_data.get("velodrome", "?")
            rnum = entry_data.get("race_number", "?")
            logger.info("--- %s R%s ---", velodrome, rnum)

            for i, row in df.iterrows():
                rank = i + 1
                mark = MARKS[i] if i < len(MARKS) else ""
                rider_name = ""
                for e in entries:
                    if e.get("rider_id") == row["rider_id"]:
                        rider_name = e.get("rider_name", "")
                        break
                logger.info("  #%d %s %s (score=%.3f)", rank, mark, rider_name, row["pred_score"])

                insert_prediction(conn, {
                    "race_id": race_id,
                    "rider_id": row["rider_id"],
                    "predicted_score": row["pred_score"],
                    "predicted_rank": rank,
                    "mark": mark,
                })

            conn.commit()

        except Exception as e:
            logger.warning("Error predicting %s: %s", race_id, e)
            continue

    conn.close()
    logger.info("Predictions complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="予測日 (YYYYMMDD)")
    args = parser.parse_args()
    predict_races(args.date)
