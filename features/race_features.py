"""レース条件特徴量（最適化版）

改善点:
- 出走表の競走得点・勝率をそのまま特徴量として活用
- DB接続を外部から受け取れるように変更
"""

import sqlite3
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
            # --- 新規特徴量 ---
            "entry_competition_score",   # 出走表記載の競走得点
            "entry_win_rate",            # 出走表記載の勝率
            "entry_place_rate",          # 出走表記載の連対率
        ]

    def build(self, race_id: str, rider_id: str, race_date: str,
              conn: sqlite3.Connection | None = None) -> dict:
        close_conn = False
        if conn is None:
            conn = get_connection()
            close_conn = True
        try:
            return self._build_impl(conn, race_id, rider_id)
        finally:
            if close_conn:
                conn.close()

    def build_batch(self, race_id: str, rider_ids: list[str], race_date: str,
                    conn: sqlite3.Connection) -> dict[str, dict]:
        """レース内全選手の特徴量を一括計算"""
        race = conn.execute(
            "SELECT * FROM races WHERE race_id = ?", (race_id,)
        ).fetchone()

        entries = conn.execute(
            "SELECT * FROM entries WHERE race_id = ?", (race_id,)
        ).fetchall()
        entry_map = {e["rider_id"]: e for e in entries}

        if not entry_map:
            results = conn.execute(
                "SELECT * FROM race_results WHERE race_id = ?", (race_id,)
            ).fetchall()
            entry_map = {r["rider_id"]: r for r in results}

        results = {}
        for rider_id in rider_ids:
            results[rider_id] = self._build_from_cached(race, entry_map.get(rider_id))
        return results

    def _build_impl(self, conn: sqlite3.Connection, race_id: str, rider_id: str) -> dict:
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

        return self._build_from_cached(race, entry)

    def _build_from_cached(self, race, entry) -> dict:
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
            # 出走表の統計情報（entriesテーブルにある場合のみ）
            feats["entry_competition_score"] = _get_field(entry, "avg_competition_score", 0.0)
            feats["entry_win_rate"] = _get_field(entry, "win_rate", 0.0)
            feats["entry_place_rate"] = _get_field(entry, "place_rate", 0.0)
        else:
            feats["entry_frame_number"] = 1
            feats["entry_bike_number"] = 1
            feats["entry_gear_ratio"] = 3.93
            feats["entry_competition_score"] = 0.0
            feats["entry_win_rate"] = 0.0
            feats["entry_place_rate"] = 0.0

        return feats


def _get_field(row, field: str, default):
    """sqlite3.Row から安全にフィールドを取得"""
    try:
        val = row[field]
        return val if val is not None else default
    except (IndexError, KeyError):
        return default
