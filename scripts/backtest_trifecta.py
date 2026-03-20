"""大規模バックテスト: 3連単戦略の収支分析

訓練データ全体（2022-2025）で信頼度と的中率の関係を検証。
※ in-sample分析のため実運用より良い結果が出るが、相対的なパターンは参考になる。
"""
import sys, io, logging, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    t0 = time.time()

    # モデルロード
    model = LGBMRanker()
    model.load(str(RESULTS_DIR / "model_lgbm.pkl"))
    feature_names = model.feature_names
    logger.info("Model loaded, %d features", len(feature_names))

    # 特徴量構築（全期間）
    builder = FeatureBuilder()
    df = builder.build_dataset(2022, 2025)
    logger.info("Dataset built: %d rows, %.1f sec", len(df), time.time() - t0)

    # 予測スコア計算
    X = df[feature_names].fillna(0)
    df["pred_score"] = model.predict(X).values
    logger.info("Predictions done")

    # レースごとに分析
    results = []
    race_groups = df.groupby("race_id")

    for race_id, group in race_groups:
        group = group.copy()
        n = len(group)
        if n < 5:
            continue

        # 予測順位
        group = group.sort_values("pred_score", ascending=False).reset_index(drop=True)
        group["pred_rank"] = range(1, n + 1)

        # 実際の着順
        if group["finish_position"].isna().all():
            continue

        # 信頼度計算
        top3_scores = group.loc[group["pred_rank"] <= 3, "pred_score"].values
        rest_scores = group.loc[group["pred_rank"] > 3, "pred_score"].values

        if len(top3_scores) < 3 or len(rest_scores) == 0:
            continue

        confidence = float(min(top3_scores) - max(rest_scores))
        score_gap_12 = float(top3_scores[0] - top3_scores[1])
        top1_score = float(top3_scores[0])

        # 予測TOP-N riders
        pred_top3 = set(group.loc[group["pred_rank"] <= 3, "rider_id"])
        pred_top4 = set(group.loc[group["pred_rank"] <= 4, "rider_id"])
        pred_top5 = set(group.loc[group["pred_rank"] <= 5, "rider_id"])

        # 実際TOP3 riders
        valid = group.dropna(subset=["finish_position"])
        actual_top3 = set(valid.loc[valid["finish_position"] <= 3, "rider_id"])

        if len(actual_top3) < 3:
            continue

        # 各種的中判定
        top1_hit = int(group.iloc[0]["finish_position"] == 1) if pd.notna(group.iloc[0]["finish_position"]) else 0
        overlap3 = len(pred_top3 & actual_top3)
        overlap4 = len(pred_top4 & actual_top3)
        overlap5 = len(pred_top5 & actual_top3)
        box3_hit = int(overlap3 == 3)
        box4_hit = int(overlap4 == 3)
        box5_hit = int(overlap5 == 3)

        # 3連単的中（順序一致）
        trifecta_hit = 0
        if box3_hit:
            pred_order = list(group.loc[group["pred_rank"] <= 3].sort_values("pred_rank")["rider_id"])
            actual_order = list(valid.loc[valid["finish_position"] <= 3].sort_values("finish_position")["rider_id"])
            if pred_order == actual_order:
                trifecta_hit = 1

        # 3連複的中（TOP3メンバー一致、順不同）
        trio_hit = box3_hit

        # grade情報
        grade_num = group.iloc[0].get("race_grade_num", 6) if "race_grade_num" in group.columns else 6

        results.append({
            "race_id": race_id,
            "confidence": confidence,
            "score_gap_12": score_gap_12,
            "top1_score": top1_score,
            "n_riders": n,
            "grade_num": grade_num,
            "top1_hit": top1_hit,
            "box3_hit": box3_hit,
            "box4_hit": box4_hit,
            "box5_hit": box5_hit,
            "trifecta_hit": trifecta_hit,
            "trio_hit": trio_hit,
            "overlap3": overlap3,
            "overlap4": overlap4,
            "overlap5": overlap5,
        })

    rdf = pd.DataFrame(results)
    logger.info("Analysis done: %d races, %.1f sec", len(rdf), time.time() - t0)

    # === 結果出力 ===
    total = len(rdf)
    print()
    print("=" * 70)
    print("  大規模バックテスト結果（2022-2025, {}レース）".format(total))
    print("  ※ in-sample分析（訓練データ上の予測）")
    print("=" * 70)
    print()

    # 全体統計
    print("=== 全体統計 ===")
    print("1着的中:      {}/{} ({:.1f}%)".format(rdf["top1_hit"].sum(), total, 100*rdf["top1_hit"].mean()))
    print("3連単的中:    {}/{} ({:.2f}%)".format(rdf["trifecta_hit"].sum(), total, 100*rdf["trifecta_hit"].mean()))
    print("3連複/BOX3:   {}/{} ({:.1f}%)".format(rdf["box3_hit"].sum(), total, 100*rdf["box3_hit"].mean()))
    print("BOX4:         {}/{} ({:.1f}%)".format(rdf["box4_hit"].sum(), total, 100*rdf["box4_hit"].mean()))
    print("BOX5:         {}/{} ({:.1f}%)".format(rdf["box5_hit"].sum(), total, 100*rdf["box5_hit"].mean()))
    print("TOP3重複(平均): {:.2f}/3".format(rdf["overlap3"].mean()))
    print()

    # 信頼度分布
    print("=== 信頼度分布 ===")
    bins = [(-999, -0.5), (-0.5, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 999)]
    labels = ["<-0.5", "-0.5~0", "0~0.25", "0.25~0.5", "0.5~0.75", "0.75~1.0", "1.0~1.5", "1.5~2.0", ">=2.0"]

    print("{:<12} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
        "信頼度", "レース数", "1着%", "3連単%", "BOX3%", "BOX4%", "BOX5%", "重複/3"))
    print("-" * 80)

    for (lo, hi), label in zip(bins, labels):
        subset = rdf[(rdf["confidence"] >= lo) & (rdf["confidence"] < hi)]
        if len(subset) == 0:
            continue
        n = len(subset)
        print("{:<12} {:>7} {:>6.1f}% {:>6.2f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.2f}".format(
            label, n,
            100*subset["top1_hit"].mean(),
            100*subset["trifecta_hit"].mean(),
            100*subset["box3_hit"].mean(),
            100*subset["box4_hit"].mean(),
            100*subset["box5_hit"].mean(),
            subset["overlap3"].mean()
        ))

    print()

    # 信頼度閾値ごとの累積分析
    print("=== 信頼度閾値別の累積分析（閾値以上のレースのみ） ===")
    print("{:<12} {:>7} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
        "閾値", "レース数", "割合%", "1着%", "3連単%", "BOX3%", "BOX4%", "BOX5%"))
    print("-" * 80)

    for threshold in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        subset = rdf[rdf["confidence"] >= threshold]
        n = len(subset)
        if n == 0:
            continue
        print("{:<12} {:>7} {:>5.1f}% {:>6.1f}% {:>6.2f}% {:>6.1f}% {:>6.1f}% {:>6.1f}%".format(
            ">={:.2f}".format(threshold), n, 100*n/total,
            100*subset["top1_hit"].mean(),
            100*subset["trifecta_hit"].mean(),
            100*subset["box3_hit"].mean(),
            100*subset["box4_hit"].mean(),
            100*subset["box5_hit"].mean()
        ))

    print()

    # 収支シミュレーション
    print("=== 収支シミュレーション（100円/通り）===")
    print()

    # 3連単の配当分布を推定できないので、損益分岐配当を計算
    strategies = [
        ("全レース BOX3 (6通り)", 6, lambda r: True, "box3_hit"),
        ("全レース BOX4 (24通り)", 24, lambda r: True, "box4_hit"),
        ("信頼度>=0.5 BOX3 (6通り)", 6, lambda r: r["confidence"] >= 0.5, "box3_hit"),
        ("信頼度>=0.5 BOX4 (24通り)", 24, lambda r: r["confidence"] >= 0.5, "box4_hit"),
        ("信頼度>=0.75 BOX3 (6通り)", 6, lambda r: r["confidence"] >= 0.75, "box3_hit"),
        ("信頼度>=0.75 BOX4 (24通り)", 24, lambda r: r["confidence"] >= 0.75, "box4_hit"),
        ("信頼度>=1.0 BOX3 (6通り)", 6, lambda r: r["confidence"] >= 1.0, "box3_hit"),
        ("信頼度>=1.0 BOX4 (24通り)", 24, lambda r: r["confidence"] >= 1.0, "box4_hit"),
        ("信頼度>=1.5 BOX3 (6通り)", 6, lambda r: r["confidence"] >= 1.5, "box3_hit"),
        ("信頼度>=1.5 BOX4 (24通り)", 24, lambda r: r["confidence"] >= 1.5, "box4_hit"),
    ]

    print("{:<30} {:>7} {:>6} {:>7} {:>10} {:>10}".format(
        "戦略", "対象R", "的中率", "的中数", "投資額", "損益分岐"))
    print("-" * 80)

    for label, cost_per_race, filter_fn, hit_col in strategies:
        subset = rdf[rdf.apply(filter_fn, axis=1)]
        n = len(subset)
        if n == 0:
            continue
        hits = int(subset[hit_col].sum())
        hit_rate = 100 * hits / n if n > 0 else 0
        total_cost = n * cost_per_race * 100
        breakeven = total_cost / hits if hits > 0 else float('inf')
        print("{:<30} {:>7} {:>5.1f}% {:>7} {:>9,}円 {:>9,.0f}円".format(
            label, n, hit_rate, hits, total_cost, breakeven))

    print()
    print("※ 損益分岐: 的中時の平均配当がこの金額以上なら収支プラス")
    print("※ 3連単の一般的な平均配当: F2=10,000~20,000円, F1=15,000~30,000円, G3+=30,000円~")
    print()

    # 1着スコア差（◎の圧倒度）別分析
    print("=== 1着スコア差（◎と○の差）別分析 ===")
    print("{:<12} {:>7} {:>7} {:>7} {:>7}".format("スコア差", "レース数", "1着%", "BOX3%", "BOX4%"))
    print("-" * 50)
    for lo, hi, label in [(0, 0.3, "<0.3"), (0.3, 0.6, "0.3~0.6"), (0.6, 1.0, "0.6~1.0"),
                          (1.0, 1.5, "1.0~1.5"), (1.5, 2.0, "1.5~2.0"), (2.0, 99, ">=2.0")]:
        subset = rdf[(rdf["score_gap_12"] >= lo) & (rdf["score_gap_12"] < hi)]
        if len(subset) == 0:
            continue
        n = len(subset)
        print("{:<12} {:>7} {:>6.1f}% {:>6.1f}% {:>6.1f}%".format(
            label, n, 100*subset["top1_hit"].mean(), 100*subset["box3_hit"].mean(), 100*subset["box4_hit"].mean()))

    print()
    print("総処理時間: {:.1f}秒".format(time.time() - t0))


if __name__ == "__main__":
    main()
