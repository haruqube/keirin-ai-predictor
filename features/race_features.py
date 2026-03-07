"""レース条件特徴量"""

from features.base import BaseFeatureBuilder
from db.schema import get_connection
from config import GRADE_MAP, BANK_LENGTH, DEFAULT_BANK_LENGTH


class RaceFeatureBuilder(BaseFeatureBuilder):
    """レースの条件から特徴量を生成"""

    @property
    def feature_names(self) -> list[str]:
        return [
            "race_grade_num",
            "race_bank_length",
            "race_rider_count",
            "race_number",
            "entry_frame_number",
            "entry_bike_number",
            "entry_gear_ratio",
        ]

    def build(self, race_id: str, rider_id: str, race_date: str) -> dict:
        conn = get_connection()
        try:
            race = conn.execute(
                "SELECT * FROM races WHERE race_id = ?", (race_id,)
            ).fetchone()

            entry = conn.execute(
                "SELECT * FROM entries WHERE race_id = ? AND rider_id = ?",
                (race_id, rider_id)
            ).fetchone()
            if not entry:
                entry = conn.execute(
                    "SELECT * FROM race_results WHERE race_id = ? AND rider_id = ?",
                    (race_id, rider_id)
                ).fetchone()

            feats = {}

            if race:
                feats["race_grade_num"] = GRADE_MAP.get(race["grade"], 6)
                velodrome = race["velodrome"] or ""
                feats["race_bank_length"] = BANK_LENGTH.get(velodrome, DEFAULT_BANK_LENGTH)
                feats["race_rider_count"] = race["rider_count"] or 9
                feats["race_number"] = race["race_number"] or 1
            else:
                feats["race_grade_num"] = 6
                feats["race_bank_length"] = DEFAULT_BANK_LENGTH
                feats["race_rider_count"] = 9
                feats["race_number"] = 1

            if entry:
                feats["entry_frame_number"] = entry["frame_number"] or 1
                feats["entry_bike_number"] = entry["bike_number"] or 1
                feats["entry_gear_ratio"] = entry["gear_ratio"] or 3.93
            else:
                feats["entry_frame_number"] = 1
                feats["entry_bike_number"] = 1
                feats["entry_gear_ratio"] = 3.93

            return feats
        finally:
            conn.close()
