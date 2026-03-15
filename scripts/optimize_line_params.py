"""ライン補正パラメータの最適化（高速版）

特徴量とベーススコアを事前計算し、ライン補正のみグリッドサーチする。
"""

import sys
import logging
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from bs4 import BeautifulSoup

from config import NETKEIRIN_BASE_URL, RESULTS_DIR, CLASS_MAP
from db.schema import get_connection
from data.scraper import KeirinScraper
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_path = str(RESULTS_DIR / "model_lgbm.pkl")
    model = LGBMRanker()
    model.load(model_path)
    logger.info("Model loaded")

    scraper = KeirinScraper()
    builder = FeatureBuilder()
    conn = get_connection()

    # 2026年のレースを取得
    races = conn.execute("""
        SELECT race_id, date FROM races
        WHERE date >= '2026-01-01'
        ORDER BY date
    """).fetchall()

    logger.info("Checking %d races for line info...", len(races))

    # 事前に全レースの特徴量・ベーススコア・ライン情報を計算
    race_cache = []

    for race in races:
        race_id = race["race_id"]
        race_date = race["date"]

        url = f"{NETKEIRIN_BASE_URL}/db/result/?race_id={race_id}"
        html = scraper._get(url)
        soup = BeautifulSoup(html, "lxml")

        line_formation = scraper._parse_line_formation(soup)
        if not line_formation:
            continue

        results = scraper._parse_result_table(soup, race_id)
        if not results or len(results) < 5:
            continue

        # 実際の結果
        sorted_results = sorted(results, key=lambda x: x.get("finish_position", 99))
        actual_1st = sorted_results[0].get("rider_id")
        actual_top3 = {r.get("rider_id") for r in sorted_results[:3]}

        # 特徴量・ベーススコア
        df = builder.build_race_features(race_id, race_date)
        if df.empty:
            continue

        feature_cols = builder.feature_names
        X = df[feature_cols].fillna(0)
        base_scores = model.predict(X).values
        rider_ids = df["rider_id"].values

        # 車番→rider_id, rider_id→ライン情報
        bike_to_rider = {}
        rider_to_bike = {}
        for r in results:
            if r.get("rider_id") and r.get("bike_number"):
                bike_to_rider[r["bike_number"]] = r["rider_id"]
                rider_to_bike[r["rider_id"]] = r["bike_number"]

        # ライン情報
        line_info = {}  # rider_id -> (line_size, role, is_strongest)
        line_scores = []
        for line_idx, members in enumerate(line_formation, 1):
            score = 0
            for member in members:
                bn = member["bike_number"]
                rid = bike_to_rider.get(bn, "")
                for r in results:
                    if r.get("rider_id") == rid:
                        cls_num = CLASS_MAP.get(r.get("class", ""), 6)
                        score += (7 - cls_num)
                        break
            line_scores.append(score)

        strongest_idx = line_scores.index(max(line_scores)) + 1 if line_scores else -1

        for line_idx, members in enumerate(line_formation, 1):
            line_size = len(members)
            for pos, member in enumerate(members):
                bn = member["bike_number"]
                rid = bike_to_rider.get(bn)
                if not rid:
                    continue
                if pos == 0:
                    role = "jiku"
                elif pos == 1:
                    role = "bante"
                else:
                    role = "3bante"
                is_strongest = (line_idx == strongest_idx)
                line_info[rid] = (line_size, role, is_strongest)

        race_cache.append({
            "rider_ids": rider_ids,
            "base_scores": base_scores,
            "line_info": line_info,
            "actual_1st": actual_1st,
            "actual_top3": actual_top3,
        })

    conn.close()
    logger.info("Cached %d races with line info", len(race_cache))

    if not race_cache:
        logger.error("No data to optimize")
        return

    # ベースライン評価
    base_top1, base_top3 = evaluate_fast(race_cache, {})
    logger.info("Baseline: Top1=%.1f%% Top3=%.1f%%", base_top1*100, base_top3*100)

    # グリッドサーチ（高速版）
    logger.info("Starting grid search...")
    best_top3 = base_top3
    best_top1 = base_top1
    best_params = {}
    tested = 0

    vals = np.arange(-0.3, 0.61, 0.05)

    for b3_bante in np.arange(0.1, 0.61, 0.05):
        for b3_jiku in np.arange(-0.1, b3_bante, 0.05):
            for b3_3bante in np.arange(-0.2, b3_jiku + 0.01, 0.05):
                for b2_bante in np.arange(0.05, 0.41, 0.05):
                    for b2_jiku in np.arange(-0.15, b2_bante, 0.05):
                        for b1 in np.arange(-0.3, 0.11, 0.05):
                            for strongest in np.arange(0.0, 0.31, 0.05):
                                params = {
                                    (3, "bante"): float(b3_bante),
                                    (3, "jiku"): float(b3_jiku),
                                    (3, "3bante"): float(b3_3bante),
                                    (2, "bante"): float(b2_bante),
                                    (2, "jiku"): float(b2_jiku),
                                    (1, "jiku"): float(b1),
                                    "strongest": float(strongest),
                                }
                                t1, t3 = evaluate_fast(race_cache, params)
                                tested += 1

                                if t3 > best_top3 or (t3 == best_top3 and t1 > best_top1):
                                    best_top3 = t3
                                    best_top1 = t1
                                    best_params = params.copy()

                                if tested % 10000 == 0:
                                    logger.info("[%d] current best: Top1=%.1f%% Top3=%.1f%%",
                                                tested, best_top1*100, best_top3*100)

    logger.info("Grid search complete (%d combinations)", tested)
    logger.info("Best: Top1=%.1f%% Top3=%.1f%%", best_top1*100, best_top3*100)
    logger.info("Params: %s", {str(k): v for k, v in best_params.items()})

    # 保存
    out = RESULTS_DIR / "line_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_params = {
        "bonus_3_jiku": best_params.get((3, "jiku"), 0.0),
        "bonus_3_bante": best_params.get((3, "bante"), 0.0),
        "bonus_3_3bante": best_params.get((3, "3bante"), 0.0),
        "bonus_2_jiku": best_params.get((2, "jiku"), 0.0),
        "bonus_2_bante": best_params.get((2, "bante"), 0.0),
        "bonus_1_jiku": best_params.get((1, "jiku"), 0.0),
        "strongest_line_bonus": best_params.get("strongest", 0.0),
    }
    with open(out, "w") as f:
        json.dump({
            "best_params": save_params,
            "baseline_top1": base_top1,
            "baseline_top3": base_top3,
            "best_top1": best_top1,
            "best_top3": best_top3,
            "num_races": len(race_cache),
            "num_tested": tested,
        }, f, indent=2)
    logger.info("Saved to %s", out)

    # predict_races.pyに適用するパラメータを表示
    print("\n=== predict_races.py LINE_BONUS 更新用 ===")
    print("LINE_BONUS = {")
    for key, val in save_params.items():
        if key == "strongest_line_bonus":
            continue
        size = int(key.split("_")[1])
        role_map = {"jiku": "自力", "bante": "番手", "3bante": "3番手"}
        role = role_map[key.split("_")[2]]
        print(f'    ({size}, "{role}"): {val:.2f},')
    print("}")
    print(f"STRONGEST_LINE_BONUS = {save_params['strongest_line_bonus']:.2f}")


def evaluate_fast(race_cache, params):
    """キャッシュ済みデータでパラメータを高速評価"""
    strongest_bonus = params.get("strongest", 0.0)
    top1_hits = 0
    top3_hits = 0
    total = len(race_cache)

    for rc in race_cache:
        rider_ids = rc["rider_ids"]
        scores = rc["base_scores"].copy()
        line_info = rc["line_info"]

        # ライン補正適用
        for i, rid in enumerate(rider_ids):
            info = line_info.get(rid)
            if info is None:
                continue
            line_size, role, is_strongest = info
            key = (line_size if line_size <= 3 else 3, role)
            bonus = params.get(key, 0.0)
            if is_strongest:
                bonus += strongest_bonus
            scores[i] += bonus

        # Top3評価
        sorted_idx = np.argsort(-scores)
        pred_1st = rider_ids[sorted_idx[0]]
        pred_top3 = set(rider_ids[sorted_idx[:3]])

        if pred_1st == rc["actual_1st"]:
            top1_hits += 1
        top3_hits += len(pred_top3 & rc["actual_top3"])

    if total == 0:
        return 0, 0
    return top1_hits / total, top3_hits / (total * 3)


if __name__ == "__main__":
    main()
