"""ライン特徴量 — 競輪最大の特徴であるライン戦術を数値化"""

from features.base import BaseFeatureBuilder
from db.schema import get_connection
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

    def build(self, race_id: str, rider_id: str, race_date: str) -> dict:
        conn = get_connection()
        try:
            # 出走表または結果からライン情報を取得
            entries = conn.execute("""
                SELECT rider_id, line_group, line_role, class
                FROM entries WHERE race_id = ?
            """, (race_id,)).fetchall()

            if not entries:
                entries = conn.execute("""
                    SELECT rider_id, line_group, line_role, class
                    FROM race_results WHERE race_id = ?
                """, (race_id,)).fetchall()

            feats = {name: 0.0 for name in self.feature_names}

            if not entries:
                return feats

            # 対象選手のライン
            my_line = None
            my_role = None
            for e in entries:
                if e["rider_id"] == rider_id:
                    my_line = e["line_group"]
                    my_role = e["line_role"]
                    break

            # ライングループごとの集計
            line_groups = {}
            for e in entries:
                lg = e["line_group"]
                if not lg:
                    continue
                if lg not in line_groups:
                    line_groups[lg] = []
                line_groups[lg].append(e)

            if my_line and my_line in line_groups:
                my_group = line_groups[my_line]
                feats["line_size"] = len(my_group)
                classes = [CLASS_MAP.get(e["class"], 6) for e in my_group]
                feats["line_avg_class"] = sum(classes) / len(classes) if classes else 6.0
            else:
                feats["line_size"] = 1
                feats["line_avg_class"] = 6.0

            # 役割
            if my_role:
                role_lower = my_role.lower()
                feats["line_is_jiku"] = 1.0 if "自力" in my_role or "逃" in my_role else 0.0
                feats["line_is_番手"] = 1.0 if "番手" in my_role else 0.0
                feats["line_is_3番手"] = 1.0 if "3番手" in my_role or "三番手" in my_role else 0.0

            # ライン強度スコア（級班の合計の逆数が小さいほど強い）
            line_scores = {}
            for lg, members in line_groups.items():
                classes = [CLASS_MAP.get(e["class"], 6) for e in members]
                # 低いほど強い（SS=1, S1=2...）ので逆転
                line_scores[lg] = sum(7 - c for c in classes)

            if my_line and my_line in line_scores:
                feats["line_strength_score"] = line_scores[my_line]

            # 最強ラインに所属しているか
            if line_scores:
                strongest = max(line_scores, key=line_scores.get)
                feats["rider_in_strongest_line"] = 1.0 if my_line == strongest else 0.0

            return feats
        finally:
            conn.close()
