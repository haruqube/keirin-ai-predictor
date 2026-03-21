"""全特徴量を集約するビルダー（バッチ最適化版）

改善点:
- DB接続を1つにまとめてレース単位でバッチ処理
- N+1クエリ問題を解消（旧: 9選手×3ビルダー=27接続 → 新: 1接続で一括）
- build_dataset()にプログレス表示追加
"""


import logging
import pandas as pd
import numpy as np
from db.schema import get_connection
from config import (CLASS_MAP, GRADE_MAP, BANK_LENGTH, DEFAULT_BANK_LENGTH,
                     WEATHER_MAP, TRACK_CONDITION_MAP)

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """全特徴量をバッチ処理で一括生成"""

    def __init__(self):
        from features.rider_features import RiderFeatureBuilder
        from features.race_features import RaceFeatureBuilder
        from features.line_features import LineFeatureBuilder
        self.rider_builder = RiderFeatureBuilder()
        self.race_builder = RaceFeatureBuilder()
        self.line_builder = LineFeatureBuilder()
        self.builders = [self.rider_builder, self.race_builder, self.line_builder]

    @property
    def feature_names(self) -> list[str]:
        return [
            # 選手成績 (11)
            "rider_class_num",
            "rider_win_rate_all",
            "rider_place_rate_all",
            "rider_top3_rate_all",
            "rider_win_rate_recent10",
            "rider_place_rate_recent10",
            "rider_top3_rate_recent10",
            "rider_avg_finish_pos",
            "rider_avg_finish_pos_recent5",
            "rider_avg_finish_pos_recent10",
            "rider_race_count",
            # 競走得点 (1)
            "rider_competition_score",
            # 着差 (1)
            "rider_avg_margin",
            # 上がりタイム (3)
            "rider_avg_last_1lap",
            "rider_avg_last_1lap_recent5",
            "rider_best_last_1lap",
            # 上がりタイム正規化 (2) — Phase 2c
            "rider_normalized_last_1lap",
            "rider_normalized_last_1lap_recent5",
            # バンク相性 (2)
            "rider_velodrome_race_count",
            "rider_venue_top3_rate",
            # フォーム・安定性 (4)
            "rider_form_trend",
            "rider_form_acuity",
            "rider_days_since_last_race",
            "rider_finish_pos_std",
            # レース条件 (5)
            "race_grade_num",
            "race_bank_length",
            "race_number",
            "entry_frame_number",
            "entry_bike_number",
            # 出走表データ (4)
            "entry_competition_score",
            "entry_win_rate",
            "entry_place_rate",
            "entry_gear_ratio",
            "race_rider_count",
            # ライン (6)
            "line_size",
            "line_avg_class",
            "line_is_jiku",
            "line_is_3番手",
            "line_strength_score",
            "rider_in_strongest_line",
            # レース内相対順位 (2)
            "class_rank_in_race",
            "score_rank_in_race",
            # 交互作用 (2)
            "rider_class_x_jiku",
            "form_trend_x_line_strength",
        ]

    def build_dataset(self, year_start: int, year_end: int) -> pd.DataFrame:
        """指定年範囲のレースすべてから学習データセットをバッチ生成"""
        conn = get_connection()

        date_start = f"{year_start}-01-01"
        date_end = f"{year_end}-12-31"

        # 対象レース一覧（天候・バンク状態も取得）
        races_df = pd.read_sql("""
            SELECT race_id, date, velodrome, race_number, grade, rider_count,
                   weather, track_condition
            FROM races
            WHERE date >= ? AND date <= ?
            ORDER BY date
        """, conn, params=(date_start, date_end))
        logger.info("Target races: %d", len(races_df))

        if races_df.empty:
            conn.close()
            return pd.DataFrame()

        # 全結果データを一括読み込み
        all_results = pd.read_sql("""
            SELECT rr.race_id, rr.rider_id, rr.finish_position, rr.frame_number,
                   rr.bike_number, rr.gear_ratio, rr.odds, rr.popularity,
                   rr.class, rr.line_group, rr.line_role, rr.last_1lap,
                   rr.margin,
                   r.date as race_date, r.velodrome
            FROM race_results rr
            JOIN races r ON rr.race_id = r.race_id
            WHERE rr.finish_position IS NOT NULL
            ORDER BY r.date
        """, conn)
        logger.info("Total results loaded: %d", len(all_results))

        # 出走表データ（entry_competition_score, entry_win_rate, entry_place_rate用）
        entries_df = pd.read_sql("""
            SELECT race_id, rider_id, avg_competition_score, win_rate, place_rate,
                   odds AS entry_odds_val, popularity AS entry_popularity_val
            FROM entries
            WHERE race_id IN (SELECT race_id FROM races WHERE date >= ? AND date <= ?)
        """, conn, params=(date_start, date_end))
        logger.info("Entries loaded: %d", len(entries_df))

        # 選手マスタ（birth_yearも取得 — Phase 2b 年齢計算用）
        riders_df = pd.read_sql("SELECT rider_id, class, birth_year FROM riders", conn)
        rider_class_map = dict(zip(riders_df["rider_id"], riders_df["class"]))
        rider_birth_year_map = dict(zip(riders_df["rider_id"], riders_df["birth_year"]))
        conn.close()

        # 対象期間の結果（特徴量付与対象）
        target_results = all_results[
            (all_results["race_date"] >= date_start) &
            (all_results["race_date"] <= date_end)
        ].copy()
        logger.info("Target results: %d", len(target_results))

        # --- レース条件特徴量 ---
        race_map = races_df.set_index("race_id")
        target_results["race_grade_num"] = target_results["race_id"].map(
            lambda rid: GRADE_MAP.get(race_map.loc[rid, "grade"] if rid in race_map.index else None, 6)
        )
        target_results["race_bank_length"] = target_results["race_id"].map(
            lambda rid: BANK_LENGTH.get(race_map.loc[rid, "velodrome"] if rid in race_map.index else "", DEFAULT_BANK_LENGTH)
        )
        target_results["race_rider_count"] = target_results["race_id"].map(
            lambda rid: race_map.loc[rid, "rider_count"] if rid in race_map.index else 9
        ).fillna(9).astype(int)
        target_results["race_number"] = target_results["race_id"].map(
            lambda rid: race_map.loc[rid, "race_number"] if rid in race_map.index else 1
        ).fillna(1).astype(int)
        target_results["entry_frame_number"] = target_results["frame_number"].fillna(1)
        target_results["entry_bike_number"] = target_results["bike_number"].fillna(1)
        target_results["entry_gear_ratio"] = pd.to_numeric(target_results["gear_ratio"], errors="coerce").fillna(3.93)

        # --- Phase 2a: 天候・バンク状態 ---
        target_results["race_weather"] = target_results["race_id"].map(
            lambda rid: WEATHER_MAP.get(
                race_map.loc[rid, "weather"] if rid in race_map.index else None, 0
            )
        )
        target_results["race_track_condition"] = target_results["race_id"].map(
            lambda rid: TRACK_CONDITION_MAP.get(
                race_map.loc[rid, "track_condition"] if rid in race_map.index else None, 0
            )
        )

        # --- Phase 2d: オッズ・人気（race_resultsの値を使用） ---
        target_results["entry_odds"] = pd.to_numeric(
            target_results["odds"], errors="coerce"
        ).fillna(0.0)
        target_results["entry_popularity"] = pd.to_numeric(
            target_results["popularity"], errors="coerce"
        ).fillna(5).astype(int)

        # --- Phase 1: 出走表の公式競走得点・勝率・連対率 ---
        if not entries_df.empty:
            entries_merge = entries_df[["race_id", "rider_id",
                                        "avg_competition_score", "win_rate", "place_rate"]].copy()
            entries_merge = entries_merge.rename(columns={
                "avg_competition_score": "entry_competition_score",
                "win_rate": "entry_win_rate",
                "place_rate": "entry_place_rate",
            })
            target_results = target_results.merge(
                entries_merge, on=["race_id", "rider_id"], how="left"
            )
        target_results["entry_competition_score"] = target_results.get(
            "entry_competition_score", pd.Series(0.0, index=target_results.index)
        ).fillna(0.0)
        target_results["entry_win_rate"] = target_results.get(
            "entry_win_rate", pd.Series(0.0, index=target_results.index)
        ).fillna(0.0)
        target_results["entry_place_rate"] = target_results.get(
            "entry_place_rate", pd.Series(0.0, index=target_results.index)
        ).fillna(0.0)

        logger.info("Race features done")

        # --- 選手成績特徴量（累積計算）---
        # 日付でソートして累積統計を計算
        all_results_sorted = all_results.sort_values(["race_date", "race_id"]).reset_index(drop=True)

        # 各選手の累積統計を計算
        rider_stats = self._compute_rider_stats(all_results_sorted, date_start, date_end)
        logger.info("Rider stats computed: %d entries", len(rider_stats))

        # rider_class_num
        target_results["rider_class_num"] = target_results["rider_id"].map(
            lambda rid: CLASS_MAP.get(rider_class_map.get(rid), 6)
        )

        # --- Phase 2b: 選手年齢 ---
        target_results["rider_age"] = target_results.apply(
            lambda row: (
                int(str(row["race_date"])[:4]) - rider_birth_year_map.get(row["rider_id"], 0)
                if rider_birth_year_map.get(row["rider_id"])
                else 0
            ), axis=1
        )

        # rider_statsをマージ
        stats_df = pd.DataFrame(rider_stats)
        if not stats_df.empty:
            target_results = target_results.merge(
                stats_df, on=["race_id", "rider_id"], how="left"
            )

        # --- ライン特徴量 ---
        line_feats = self._compute_line_features(target_results)
        target_results = target_results.merge(
            line_feats, on=["race_id", "rider_id"], how="left"
        )

        # --- Phase 2c: 上がりタイム正規化 ---
        bank_len = target_results["race_bank_length"].replace(0, DEFAULT_BANK_LENGTH)
        avg_lap = target_results.get("rider_avg_last_1lap", 0)
        avg_lap_r5 = target_results.get("rider_avg_last_1lap_recent5", 0)
        target_results["rider_normalized_last_1lap"] = np.where(
            avg_lap > 0, avg_lap / bank_len * 400, 0.0
        )
        target_results["rider_normalized_last_1lap_recent5"] = np.where(
            avg_lap_r5 > 0, avg_lap_r5 / bank_len * 400, 0.0
        )

        # --- 交互作用特徴量 ---
        target_results["rider_class_x_jiku"] = (
            target_results.get("rider_class_num", 6) *
            target_results.get("line_is_jiku", 0)
        )
        target_results["form_trend_x_line_strength"] = (
            target_results.get("rider_form_trend", 0) *
            target_results.get("line_strength_score", 0)
        )

        # --- レース内相対順位 ---
        target_results["class_rank_in_race"] = target_results.groupby("race_id")["rider_class_num"].rank(method="min")
        target_results["score_rank_in_race"] = target_results.groupby("race_id")["rider_competition_score"].rank(method="min", ascending=False)

        logger.info("All features computed")

        # 欠損値埋め
        for col in self.feature_names:
            if col not in target_results.columns:
                target_results[col] = 0.0

        target_results = target_results.fillna(0)

        return target_results

    def _compute_rider_stats(self, all_results: pd.DataFrame,
                              date_start: str, date_end: str) -> list[dict]:
        """各選手の累積成績を効率的に計算"""
        stats_list = []

        # rider_idごとにグループ化
        grouped = all_results.groupby("rider_id")

        # 対象レースの (race_id, rider_id, race_date, velodrome) を取得
        target_mask = (all_results["race_date"] >= date_start) & (all_results["race_date"] <= date_end)
        target_entries = all_results[target_mask][["race_id", "rider_id", "race_date", "velodrome"]].values

        # 各選手の全結果を事前にインデックス化
        rider_history = {}
        for rider_id, group in grouped:
            rider_history[rider_id] = group[["race_date", "race_id", "finish_position", "odds", "popularity", "last_1lap", "velodrome", "margin"]].values

        empty_stats = {
            "rider_win_rate_all": 0, "rider_place_rate_all": 0,
            "rider_top3_rate_all": 0, "rider_win_rate_recent5": 0,
            "rider_place_rate_recent5": 0, "rider_top3_rate_recent5": 0,
            "rider_win_rate_recent10": 0,
            "rider_place_rate_recent10": 0, "rider_top3_rate_recent10": 0,
            "rider_avg_finish_pos": 5.0,
            "rider_avg_finish_pos_recent5": 5.0,
            "rider_avg_finish_pos_recent10": 5.0,
            "rider_race_count": 0,
            "rider_avg_odds": 0, "rider_avg_popularity": 5,
            "rider_avg_last_1lap": 0.0, "rider_avg_last_1lap_recent5": 0.0,
            "rider_best_last_1lap": 0.0,
            "rider_velodrome_win_rate": 0.0, "rider_velodrome_race_count": 0,
            "rider_venue_win_rate": 0.0, "rider_venue_top3_rate": 0.0,
            "rider_venue_race_count": 0,
            "rider_form_trend": 0.0, "rider_form_acuity": 0.0,
            "rider_days_since_last_race": 90.0,
            "rider_finish_pos_std": 0.0,
            "rider_competition_score": 0.0,
            "rider_avg_margin": 2.0,
        }

        total = len(target_entries)
        for i, (race_id, rider_id, race_date, velodrome) in enumerate(target_entries):
            if i > 0 and i % 20000 == 0:
                logger.info("  Rider stats progress: %d/%d (%.0f%%)", i, total, i/total*100)

            history = rider_history.get(rider_id)
            if history is None:
                stats_list.append({"race_id": race_id, "rider_id": rider_id, **empty_stats})
                continue

            # 当該レース以前のデータ
            past_mask = history[:, 0] < race_date
            past = history[past_mask]
            total_past = len(past)

            if total_past == 0:
                stats_list.append({"race_id": race_id, "rider_id": rider_id, **empty_stats})
                continue

            positions = past[:, 2].astype(float)
            positions_rev = positions[::-1]

            wins = np.sum(positions == 1)
            top2 = np.sum(positions <= 2)
            top3 = np.sum(positions <= 3)

            recent5 = positions_rev[:5]
            r5 = len(recent5)
            recent10 = positions_rev[:10]
            r10 = len(recent10)

            odds_vals = past[:, 3]
            odds_valid = odds_vals[odds_vals != None]
            try:
                odds_valid = odds_valid.astype(float)
                odds_valid = odds_valid[~np.isnan(odds_valid)]
                avg_odds = float(np.mean(odds_valid)) if len(odds_valid) > 0 else 0.0
            except (ValueError, TypeError):
                avg_odds = 0.0

            pop_vals = past[:, 4]
            try:
                pop_valid = pop_vals[pop_vals != None].astype(float)
                pop_valid = pop_valid[~np.isnan(pop_valid)]
                avg_pop = float(np.mean(pop_valid)) if len(pop_valid) > 0 else 5.0
            except (ValueError, TypeError):
                avg_pop = 5.0

            # 上がりタイム（last_1lap）
            lap_vals = past[:, 5]
            try:
                lap_valid = []
                for v in lap_vals:
                    if v is not None and v != "":
                        fv = float(v)
                        if fv > 0 and not np.isnan(fv):
                            lap_valid.append(fv)
                avg_lap = sum(lap_valid) / len(lap_valid) if lap_valid else 0.0
                best_lap = min(lap_valid) if lap_valid else 0.0
            except (ValueError, TypeError):
                avg_lap = 0.0
                best_lap = 0.0

            # 直近5走の上がりタイム
            lap_rev = past[::-1][:5, 5]
            try:
                lap_r5 = []
                for v in lap_rev:
                    if v is not None and v != "":
                        fv = float(v)
                        if fv > 0 and not np.isnan(fv):
                            lap_r5.append(fv)
                avg_lap_r5 = sum(lap_r5) / len(lap_r5) if lap_r5 else 0.0
            except (ValueError, TypeError):
                avg_lap_r5 = 0.0

            # バンク相性
            velo_mask = past[:, 6] == velodrome
            velo_past = past[velo_mask]
            velo_total = len(velo_past)
            if velo_total > 0:
                velo_positions = velo_past[:, 2].astype(float)
                velo_wins = float(np.sum(velo_positions == 1))
                velo_win_rate = velo_wins / velo_total
                velo_top3_rate = float(np.sum(velo_positions <= 3)) / velo_total
            else:
                velo_win_rate = 0.0
                velo_top3_rate = 0.0

            # フォームトレンド（直近5走の線形回帰スロープ）
            if r5 >= 3:
                recent5_rev = recent5[::-1]  # 最古→最新の順
                n5 = len(recent5_rev)
                x_mean = (n5 - 1) / 2.0
                y_mean = float(np.mean(recent5_rev))
                numer = sum((i - x_mean) * (float(recent5_rev[i]) - y_mean) for i in range(n5))
                denom = sum((i - x_mean) ** 2 for i in range(n5))
                form_trend = numer / denom if denom > 0 else 0.0
            else:
                form_trend = 0.0

            # 前走からの日数
            try:
                from datetime import datetime
                last_date = datetime.strptime(str(past[-1, 0])[:10], "%Y-%m-%d")
                current_date = datetime.strptime(str(race_date)[:10], "%Y-%m-%d")
                days_since = (current_date - last_date).days
            except (ValueError, TypeError):
                days_since = 30.0

            # 着順の標準偏差
            finish_pos_std = float(np.std(positions))

            # フォーム鋭度（直近1走勝率 - 直近5走勝率 = 加速度）
            recent1_win = 1.0 if positions_rev[0] == 1 else 0.0
            form_acuity = recent1_win - (float(np.sum(recent5 == 1)) / r5)

            # 競走得点（直近20走の加重平均スコア）
            recent20 = positions_rev[:20]
            r20 = len(recent20)
            scores = np.maximum(0, 11.0 - recent20)  # 1着=10, 2着=9, ..., 9着=2
            competition_score = float(np.mean(scores))

            # 着差（margin）の数値化
            margin_vals = past[:, 7]  # index 7 = margin
            margin_nums = []
            for mv in margin_vals:
                parsed = self._parse_margin_to_numeric(mv)
                if parsed is not None:
                    margin_nums.append(parsed)
            avg_margin = sum(margin_nums) / len(margin_nums) if margin_nums else 2.0

            stats_list.append({
                "race_id": race_id, "rider_id": rider_id,
                "rider_win_rate_all": wins / total_past,
                "rider_place_rate_all": top2 / total_past,
                "rider_top3_rate_all": top3 / total_past,
                "rider_win_rate_recent5": np.sum(recent5 == 1) / r5,
                "rider_place_rate_recent5": np.sum(recent5 <= 2) / r5,
                "rider_top3_rate_recent5": np.sum(recent5 <= 3) / r5,
                "rider_win_rate_recent10": np.sum(recent10 == 1) / r10,
                "rider_place_rate_recent10": float(np.sum(recent10 <= 2)) / r10,
                "rider_top3_rate_recent10": float(np.sum(recent10 <= 3)) / r10,
                "rider_avg_finish_pos": float(np.mean(positions)),
                "rider_avg_finish_pos_recent5": float(np.mean(recent5)),
                "rider_avg_finish_pos_recent10": float(np.mean(recent10)),
                "rider_race_count": total_past,
                "rider_avg_odds": avg_odds,
                "rider_avg_popularity": avg_pop,
                "rider_avg_last_1lap": avg_lap,
                "rider_avg_last_1lap_recent5": avg_lap_r5,
                "rider_best_last_1lap": best_lap,
                "rider_velodrome_win_rate": velo_win_rate,
                "rider_velodrome_race_count": velo_total,
                "rider_venue_win_rate": velo_win_rate,
                "rider_venue_top3_rate": velo_top3_rate,
                "rider_venue_race_count": velo_total,
                "rider_form_trend": form_trend,
                "rider_form_acuity": form_acuity,
                "rider_days_since_last_race": days_since,
                "rider_finish_pos_std": finish_pos_std,
                "rider_competition_score": competition_score,
                "rider_avg_margin": avg_margin,
            })

        return stats_list

    @staticmethod
    def _parse_fraction(s: str) -> float:
        """分数文字列を数値に変換 ('1/2' -> 0.5)"""
        if not s:
            return 0.0
        if '/' in s:
            parts = s.split('/')
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0

    @staticmethod
    def _parse_margin_to_numeric(margin_str) -> float | None:
        """着差文字列を車身数(float)に変換"""
        if not margin_str or not isinstance(margin_str, str):
            return None
        margin = margin_str.split('(')[0].strip()
        if not margin:
            return None
        if 'タイヤ' in margin:
            return 0.05
        if '大差' in margin:
            return 10.0
        if '車輪' in margin:
            num_part = margin.replace('車輪', '').strip()
            return FeatureBuilder._parse_fraction(num_part) * 0.5
        if '車身' in margin:
            parts = margin.split('車身')
            main = FeatureBuilder._parse_fraction(parts[0].strip()) if parts[0].strip() else 0
            frac = FeatureBuilder._parse_fraction(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
            return main + frac
        return None

    def _compute_line_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ライン特徴量をバッチ計算"""
        rows = []

        for race_id, group in df.groupby("race_id"):
            # ライングループ集計
            line_groups = {}
            for _, row in group.iterrows():
                lg = row.get("line_group")
                if not lg or pd.isna(lg):
                    continue
                if lg not in line_groups:
                    line_groups[lg] = []
                line_groups[lg].append({
                    "rider_id": row["rider_id"],
                    "class": row.get("class"),
                    "line_role": row.get("line_role"),
                })

            # ライン強度スコア
            line_scores = {}
            for lg, members in line_groups.items():
                classes = [CLASS_MAP.get(m["class"], 6) for m in members]
                line_scores[lg] = sum(7 - c for c in classes)

            strongest = max(line_scores, key=line_scores.get) if line_scores else None

            for _, row in group.iterrows():
                rider_id = row["rider_id"]
                my_line = row.get("line_group")
                my_role = row.get("line_role") or ""

                feats = {
                    "race_id": race_id,
                    "rider_id": rider_id,
                    "line_size": 1,
                    "line_avg_class": 6.0,
                    "line_is_jiku": 0.0,
                    "line_is_番手": 0.0,
                    "line_is_3番手": 0.0,
                    "line_strength_score": 0.0,
                    "rider_in_strongest_line": 0.0,
                }

                if my_line and not pd.isna(my_line) and my_line in line_groups:
                    members = line_groups[my_line]
                    feats["line_size"] = len(members)
                    classes = [CLASS_MAP.get(m["class"], 6) for m in members]
                    feats["line_avg_class"] = sum(classes) / len(classes)
                    feats["line_strength_score"] = line_scores.get(my_line, 0)
                    feats["rider_in_strongest_line"] = 1.0 if my_line == strongest else 0.0

                if my_role:
                    feats["line_is_jiku"] = 1.0 if "自力" in my_role or "逃" in my_role else 0.0
                    feats["line_is_番手"] = 1.0 if "番手" in my_role else 0.0
                    feats["line_is_3番手"] = 1.0 if "3番手" in my_role or "三番手" in my_role else 0.0

                rows.append(feats)

        return pd.DataFrame(rows)

    def build_race_features(self, race_id: str, race_date: str,
                            conn=None) -> pd.DataFrame:
        """1レース分の全選手の特徴量DataFrameを作成（最適化版）"""
        close_conn = False
        if conn is None:
            conn = get_connection()
            close_conn = True

        try:
            # 出走選手一覧を取得
            riders = conn.execute(
                "SELECT DISTINCT rider_id FROM entries WHERE race_id = ?",
                (race_id,)
            ).fetchall()

            if not riders:
                riders = conn.execute(
                    "SELECT DISTINCT rider_id FROM race_results WHERE race_id = ?",
                    (race_id,)
                ).fetchall()

            if not riders:
                return pd.DataFrame()

            rider_ids = [r["rider_id"] for r in riders if r["rider_id"]]
            if not rider_ids:
                return pd.DataFrame()

            # バッチ計算（1接続でまとめて処理）
            rider_feats = self.rider_builder.build_batch(
                race_id, rider_ids, race_date, conn)
            race_feats = self.race_builder.build_batch(
                race_id, rider_ids, race_date, conn)
            line_feats = self.line_builder.build_batch(
                race_id, rider_ids, race_date, conn)

            rows = []
            for rid in rider_ids:
                feat_row = {"race_id": race_id, "rider_id": rid}
                feat_row.update(rider_feats.get(rid, {}))
                feat_row.update(race_feats.get(rid, {}))
                feat_row.update(line_feats.get(rid, {}))

                # 交互作用特徴量
                feat_row["rider_class_x_jiku"] = (
                    feat_row.get("rider_class_num", 6) *
                    feat_row.get("line_is_jiku", 0)
                )
                feat_row["form_trend_x_line_strength"] = (
                    feat_row.get("rider_form_trend", 0) *
                    feat_row.get("line_strength_score", 0)
                )

                rows.append(feat_row)

            df = pd.DataFrame(rows)
            if not df.empty:
                # Phase 2c: 上がりタイム正規化
                bl = df["race_bank_length"].replace(0, DEFAULT_BANK_LENGTH)
                avg_l = df.get("rider_avg_last_1lap", 0)
                avg_l5 = df.get("rider_avg_last_1lap_recent5", 0)
                df["rider_normalized_last_1lap"] = np.where(
                    avg_l > 0, avg_l / bl * 400, 0.0
                )
                df["rider_normalized_last_1lap_recent5"] = np.where(
                    avg_l5 > 0, avg_l5 / bl * 400, 0.0
                )
                # レース内相対順位
                df["class_rank_in_race"] = df["rider_class_num"].rank(method="min")
                df["score_rank_in_race"] = df["rider_competition_score"].rank(method="min", ascending=False)
            return df
        finally:
            if close_conn:
                conn.close()
