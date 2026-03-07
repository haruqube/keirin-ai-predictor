"""選手成績特徴量"""

from features.base import BaseFeatureBuilder
from db.schema import get_connection
from config import CLASS_MAP


class RiderFeatureBuilder(BaseFeatureBuilder):
    """選手の過去成績から特徴量を生成"""

    @property
    def feature_names(self) -> list[str]:
        return [
            "rider_class_num",
            "rider_win_rate_all",
            "rider_place_rate_all",
            "rider_top3_rate_all",
            "rider_win_rate_recent5",
            "rider_place_rate_recent5",
            "rider_top3_rate_recent5",
            "rider_win_rate_recent10",
            "rider_avg_finish_pos",
            "rider_avg_finish_pos_recent5",
            "rider_race_count",
            "rider_avg_odds",
            "rider_avg_popularity",
        ]

    def build(self, race_id: str, rider_id: str, race_date: str) -> dict:
        conn = get_connection()
        try:
            # 当該レース以前の全成績
            past = conn.execute("""
                SELECT rr.finish_position, rr.odds, rr.popularity, rr.class
                FROM race_results rr
                JOIN races r ON rr.race_id = r.race_id
                WHERE rr.rider_id = ? AND r.date < ?
                  AND rr.finish_position IS NOT NULL
                ORDER BY r.date DESC
            """, (rider_id, race_date)).fetchall()

            feats = {}

            # 級班
            rider = conn.execute(
                "SELECT class FROM riders WHERE rider_id = ?", (rider_id,)
            ).fetchone()
            rider_class = rider["class"] if rider else None
            feats["rider_class_num"] = CLASS_MAP.get(rider_class, 6)

            total = len(past)
            feats["rider_race_count"] = total

            if total == 0:
                for name in self.feature_names:
                    feats.setdefault(name, 0.0)
                return feats

            wins = sum(1 for r in past if r["finish_position"] == 1)
            top2 = sum(1 for r in past if r["finish_position"] <= 2)
            top3 = sum(1 for r in past if r["finish_position"] <= 3)

            feats["rider_win_rate_all"] = wins / total
            feats["rider_place_rate_all"] = top2 / total
            feats["rider_top3_rate_all"] = top3 / total

            # 直近5走
            recent5 = past[:5]
            r5 = len(recent5)
            if r5 > 0:
                feats["rider_win_rate_recent5"] = sum(1 for r in recent5 if r["finish_position"] == 1) / r5
                feats["rider_place_rate_recent5"] = sum(1 for r in recent5 if r["finish_position"] <= 2) / r5
                feats["rider_top3_rate_recent5"] = sum(1 for r in recent5 if r["finish_position"] <= 3) / r5
                feats["rider_avg_finish_pos_recent5"] = sum(r["finish_position"] for r in recent5) / r5
            else:
                feats["rider_win_rate_recent5"] = 0.0
                feats["rider_place_rate_recent5"] = 0.0
                feats["rider_top3_rate_recent5"] = 0.0
                feats["rider_avg_finish_pos_recent5"] = 5.0

            # 直近10走
            recent10 = past[:10]
            r10 = len(recent10)
            feats["rider_win_rate_recent10"] = sum(1 for r in recent10 if r["finish_position"] == 1) / r10

            feats["rider_avg_finish_pos"] = sum(r["finish_position"] for r in past) / total

            # オッズ・人気
            odds_vals = [r["odds"] for r in past if r["odds"]]
            feats["rider_avg_odds"] = sum(odds_vals) / len(odds_vals) if odds_vals else 0.0

            pop_vals = [r["popularity"] for r in past if r["popularity"]]
            feats["rider_avg_popularity"] = sum(pop_vals) / len(pop_vals) if pop_vals else 5.0

            return feats
        finally:
            conn.close()
