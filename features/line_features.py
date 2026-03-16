"""ライン特徴量（最適化版）

改善点:
- DB接続を外部から受け取れるように変更
- バッチ計算対応（レース全選手分を1クエリで取得）
- 役割判定ロジックの堅牢化
"""

import sqlite3
from features.base import BaseFeatureBuilder
from config import CLASS_MAP


class LineFeatureBuilder(BaseFeatureBuilder):
    """ラインの強さ・構成から特徴量を生成"""

    @property
    def feature_names(self) -> list[str]:
        return [
            "line_size",
            "line_avg_class",
            "line_is_jiku",
            "line_is_番手",
            "line_is_3番手",
            "line_strength_score",
            "rider_in_strongest_line",
        ]

    def build(self, race_id: str, rider_id: str, race_date: str,
              conn: sqlite3.Connection | None = None) -> dict:
        close_conn = False
        if conn is None:
            from db.schema import get_connection
            conn = get_connection()
            close_conn = True
        try:
            all_feats = self._build_all(conn, race_id)
            return all_feats.get(rider_id, {name: 0.0 for name in self.feature_names})
        finally:
            if close_conn:
                conn.close()

    def build_batch(self, race_id: str, rider_ids: list[str], race_date: str,
                    conn: sqlite3.Connection) -> dict[str, dict]:
        """レース内全選手の特徴量を一括計算"""
        all_feats = self._build_all(conn, race_id)
        default = {name: 0.0 for name in self.feature_names}
        return {rid: all_feats.get(rid, default.copy()) for rid in rider_ids}

    def _build_all(self, conn: sqlite3.Connection, race_id: str) -> dict[str, dict]:
        """1レース全選手のライン特徴量を一括計算"""
        entries = conn.execute("""
            SELECT rider_id, line_group, line_role, class
            FROM entries WHERE race_id = ?
        """, (race_id,)).fetchall()

        if not entries:
            entries = conn.execute("""
                SELECT rider_id, line_group, line_role, class
                FROM race_results WHERE race_id = ?
            """, (race_id,)).fetchall()

        if not entries:
            return {}

        # ライングループごとに集計
        line_groups: dict[str, list] = {}
        rider_info: dict[str, tuple] = {}  # rider_id -> (line_group, line_role)

        for e in entries:
            rid = e["rider_id"]
            lg = e["line_group"]
            role = e["line_role"]
            rider_info[rid] = (lg, role)
            if lg:
                if lg not in line_groups:
                    line_groups[lg] = []
                line_groups[lg].append(e)

        # ライン強度スコア
        line_scores = {}
        for lg, members in line_groups.items():
            classes = [CLASS_MAP.get(e["class"], 6) for e in members]
            line_scores[lg] = sum(7 - c for c in classes)

        strongest = max(line_scores, key=line_scores.get) if line_scores else None

        # 各選手の特徴量を生成
        all_feats = {}
        for rid, (lg, role) in rider_info.items():
            feats = {name: 0.0 for name in self.feature_names}

            if lg and lg in line_groups:
                members = line_groups[lg]
                feats["line_size"] = len(members)
                classes = [CLASS_MAP.get(e["class"], 6) for e in members]
                feats["line_avg_class"] = sum(classes) / len(classes) if classes else 6.0
                feats["line_strength_score"] = line_scores.get(lg, 0.0)
                feats["rider_in_strongest_line"] = 1.0 if lg == strongest else 0.0
            else:
                feats["line_size"] = 1
                feats["line_avg_class"] = 6.0

            # 役割
            if role:
                feats["line_is_jiku"] = 1.0 if "自力" in role or "逃" in role else 0.0
                feats["line_is_番手"] = 1.0 if "番手" in role and "3" not in role and "三" not in role else 0.0
                feats["line_is_3番手"] = 1.0 if "3番手" in role or "三番手" in role else 0.0

            all_feats[rid] = feats

        return all_feats
