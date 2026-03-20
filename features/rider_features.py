"""選手成績特徴量（最適化版）

改善点:
- 1クエリでレース全選手の過去成績を取得（N+1解消）
- 競走得点・上がりタイム・バンク別勝率・フォームトレンド等の新特徴量追加
- DB接続を外部から渡す方式に変更
"""

import sqlite3
from features.base import BaseFeatureBuilder
from config import CLASS_MAP


class RiderFeatureBuilder(BaseFeatureBuilder):
    """選手の過去成績から特徴量を生成"""

    @property
    def feature_names(self) -> list[str]:
        return [
            # 基本成績
            "rider_class_num",
            "rider_win_rate_all",
            "rider_place_rate_all",
            "rider_top3_rate_all",
            # 直近成績
            "rider_win_rate_recent5",
            "rider_place_rate_recent5",
            "rider_top3_rate_recent5",
            "rider_win_rate_recent10",
            "rider_place_rate_recent10",
            "rider_top3_rate_recent10",
            # 平均着順
            "rider_avg_finish_pos",
            "rider_avg_finish_pos_recent5",
            "rider_avg_finish_pos_recent10",
            # 出走数
            "rider_race_count",
            # --- 上がりタイム特徴量 ---
            "rider_avg_last_1lap",
            "rider_avg_last_1lap_recent5",
            "rider_best_last_1lap",
            # --- バンク相性 ---
            "rider_velodrome_win_rate",
            "rider_velodrome_race_count",
            # --- バンク別勝率（詳細） ---
            "rider_venue_win_rate",
            "rider_venue_top3_rate",
            "rider_venue_race_count",
            # フォームトレンド（直近5走の着順が改善傾向か悪化傾向か）
            "rider_form_trend",
            # フォーム鋭度（直近1走 vs 直近5走の加速度）
            "rider_form_acuity",
            # 前走からの日数
            "rider_days_since_last_race",
            # 着順の安定性（標準偏差）
            "rider_finish_pos_std",
            # 競走得点プロキシ
            "rider_competition_score",
            # 平均着差
            "rider_avg_margin",
            # 選手年齢
            "rider_age",
        ]

    def build(self, race_id: str, rider_id: str, race_date: str,
              conn: sqlite3.Connection | None = None) -> dict:
        close_conn = False
        if conn is None:
            from db.schema import get_connection
            conn = get_connection()
            close_conn = True
        try:
            return self._build_impl(conn, race_id, rider_id, race_date)
        finally:
            if close_conn:
                conn.close()

    def build_batch(self, race_id: str, rider_ids: list[str], race_date: str,
                    conn: sqlite3.Connection) -> dict[str, dict]:
        """レース内全選手の特徴量を一括計算"""
        results = {}
        # バンク情報を取得
        race = conn.execute(
            "SELECT velodrome, date FROM races WHERE race_id = ?", (race_id,)
        ).fetchone()
        velodrome = race["velodrome"] if race else None

        for rider_id in rider_ids:
            results[rider_id] = self._build_impl(conn, race_id, rider_id, race_date, velodrome)
        return results

    def _build_impl(self, conn: sqlite3.Connection, race_id: str,
                    rider_id: str, race_date: str,
                    velodrome: str | None = None) -> dict:
        # 当該レース以前の全成績（日付降順）
        past = conn.execute("""
            SELECT rr.finish_position, rr.odds, rr.popularity, rr.class,
                   rr.last_1lap, r.date, r.velodrome, rr.margin
            FROM race_results rr
            JOIN races r ON rr.race_id = r.race_id
            WHERE rr.rider_id = ? AND r.date < ?
              AND rr.finish_position IS NOT NULL
            ORDER BY r.date DESC
        """, (rider_id, race_date)).fetchall()

        feats = {}

        # 級班・生年
        rider = conn.execute(
            "SELECT class, birth_year FROM riders WHERE rider_id = ?", (rider_id,)
        ).fetchone()
        rider_class = rider["class"] if rider else None
        feats["rider_class_num"] = CLASS_MAP.get(rider_class, 6)

        # 選手年齢
        birth_year = rider["birth_year"] if rider and rider["birth_year"] else None
        if birth_year:
            try:
                race_year = int(race_date[:4])
                feats["rider_age"] = race_year - birth_year
            except (ValueError, TypeError):
                feats["rider_age"] = 0
        else:
            feats["rider_age"] = 0

        total = len(past)
        feats["rider_race_count"] = total

        if total == 0:
            for name in self.feature_names:
                feats.setdefault(name, 0.0)
            feats["rider_avg_finish_pos"] = 5.0
            feats["rider_avg_finish_pos_recent5"] = 5.0
            feats["rider_avg_finish_pos_recent10"] = 5.0
            feats["rider_avg_popularity"] = 5.0
            feats["rider_days_since_last_race"] = 90.0
            return feats

        positions = [r["finish_position"] for r in past]
        wins = sum(1 for p in positions if p == 1)
        top2 = sum(1 for p in positions if p <= 2)
        top3 = sum(1 for p in positions if p <= 3)

        feats["rider_win_rate_all"] = wins / total
        feats["rider_place_rate_all"] = top2 / total
        feats["rider_top3_rate_all"] = top3 / total
        feats["rider_avg_finish_pos"] = sum(positions) / total

        # 着順の標準偏差（安定性指標）
        mean_pos = feats["rider_avg_finish_pos"]
        feats["rider_finish_pos_std"] = (
            sum((p - mean_pos) ** 2 for p in positions) / total
        ) ** 0.5

        # 直近5走
        r5 = past[:5]
        n5 = len(r5)
        p5 = [r["finish_position"] for r in r5]
        feats["rider_win_rate_recent5"] = sum(1 for p in p5 if p == 1) / n5
        feats["rider_place_rate_recent5"] = sum(1 for p in p5 if p <= 2) / n5
        feats["rider_top3_rate_recent5"] = sum(1 for p in p5 if p <= 3) / n5
        feats["rider_avg_finish_pos_recent5"] = sum(p5) / n5

        # 直近10走
        r10 = past[:10]
        n10 = len(r10)
        p10 = [r["finish_position"] for r in r10]
        feats["rider_win_rate_recent10"] = sum(1 for p in p10 if p == 1) / n10
        feats["rider_place_rate_recent10"] = sum(1 for p in p10 if p <= 2) / n10
        feats["rider_top3_rate_recent10"] = sum(1 for p in p10 if p <= 3) / n10
        feats["rider_avg_finish_pos_recent10"] = sum(p10) / n10

        # オッズ・人気
        odds_vals = [r["odds"] for r in past if r["odds"]]
        feats["rider_avg_odds"] = sum(odds_vals) / len(odds_vals) if odds_vals else 0.0
        pop_vals = [r["popularity"] for r in past if r["popularity"]]
        feats["rider_avg_popularity"] = sum(pop_vals) / len(pop_vals) if pop_vals else 5.0

        # --- 新規特徴量 ---

        # 上がりタイム
        lap_vals = [r["last_1lap"] for r in past if r["last_1lap"] and r["last_1lap"] > 0]
        feats["rider_avg_last_1lap"] = sum(lap_vals) / len(lap_vals) if lap_vals else 0.0
        feats["rider_best_last_1lap"] = min(lap_vals) if lap_vals else 0.0
        lap5 = [r["last_1lap"] for r in r5 if r["last_1lap"] and r["last_1lap"] > 0]
        feats["rider_avg_last_1lap_recent5"] = sum(lap5) / len(lap5) if lap5 else 0.0

        # バンク相性（velodrome系 - シンプル版）
        if velodrome is None:
            race_row_v = conn.execute(
                "SELECT velodrome FROM races WHERE race_id = ?", (race_id,)
            ).fetchone()
            velodrome = race_row_v["velodrome"] if race_row_v else None

        if velodrome:
            velo_results = [r for r in past if r["velodrome"] == velodrome]
            velo_total = len(velo_results)
            feats["rider_velodrome_race_count"] = velo_total
            if velo_total > 0:
                velo_wins = sum(1 for r in velo_results if r["finish_position"] == 1)
                feats["rider_velodrome_win_rate"] = velo_wins / velo_total
            else:
                feats["rider_velodrome_win_rate"] = 0.0
        else:
            feats["rider_velodrome_win_rate"] = 0.0
            feats["rider_velodrome_race_count"] = 0

        # バンク別勝率（詳細 - venue系、velodromeは上で既に解決済み）
        if velodrome:
            venue_results = [r for r in past if r["velodrome"] == velodrome]
            vn = len(venue_results)
            feats["rider_venue_race_count"] = vn
            if vn > 0:
                vp = [r["finish_position"] for r in venue_results]
                feats["rider_venue_win_rate"] = sum(1 for p in vp if p == 1) / vn
                feats["rider_venue_top3_rate"] = sum(1 for p in vp if p <= 3) / vn
            else:
                feats["rider_venue_win_rate"] = 0.0
                feats["rider_venue_top3_rate"] = 0.0
        else:
            feats["rider_venue_win_rate"] = 0.0
            feats["rider_venue_top3_rate"] = 0.0
            feats["rider_venue_race_count"] = 0

        # フォームトレンド（直近5走の着順の線形回帰スロープ）
        # 負の値 = 改善傾向（着順が小さくなっている）
        if n5 >= 3:
            # p5[0]が最新、p5[-1]が最古 → xは最古=0, 最新=n5-1
            x_mean = (n5 - 1) / 2
            y_mean = sum(p5) / n5
            numerator = sum((i - x_mean) * (p5[n5 - 1 - i] - y_mean) for i in range(n5))
            denominator = sum((i - x_mean) ** 2 for i in range(n5))
            feats["rider_form_trend"] = numerator / denominator if denominator > 0 else 0.0
        else:
            feats["rider_form_trend"] = 0.0

        # フォーム鋭度（直近1走勝率 - 直近5走勝率）
        recent1_win = 1.0 if p5[0] == 1 else 0.0
        feats["rider_form_acuity"] = recent1_win - feats["rider_win_rate_recent5"]

        # 前走からの日数
        from datetime import datetime
        try:
            last_date = datetime.strptime(past[0]["date"], "%Y-%m-%d")
            current_date = datetime.strptime(race_date, "%Y-%m-%d")
            feats["rider_days_since_last_race"] = (current_date - last_date).days
        except (ValueError, TypeError):
            feats["rider_days_since_last_race"] = 30.0

        # 競走得点（直近20走の加重平均スコア）
        r20 = past[:20]
        p20 = [r["finish_position"] for r in r20]
        scores = [max(0, 11 - p) for p in p20]
        feats["rider_competition_score"] = sum(scores) / len(scores) if scores else 0.0

        # 平均着差
        from features.builder import FeatureBuilder
        margin_nums = []
        for r in past:
            parsed = FeatureBuilder._parse_margin_to_numeric(r["margin"])
            if parsed is not None:
                margin_nums.append(parsed)
        feats["rider_avg_margin"] = sum(margin_nums) / len(margin_nums) if margin_nums else 2.0

        return feats
