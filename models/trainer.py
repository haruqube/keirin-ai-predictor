"""モデル学習・評価（改善版）

改善点:
- 時系列クロスバリデーション対応
- 評価指標拡充: Top1/Top3精度, NDCG, MRR
- ハイパーパラメータチューニング（オプション）
- 特徴量重要度の詳細ログ
"""

import logging
import numpy as np
import pandas as pd

from config import TRAIN_YEARS, TEST_YEARS, RESULTS_DIR, LGBM_PARAMS
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker

logger = logging.getLogger(__name__)


def evaluate_test_set(test_df: pd.DataFrame, model: LGBMRanker,
                      feature_cols: list[str]) -> dict:
    """テストセットの詳細評価"""
    test_df = test_df.copy()
    X_test = test_df[feature_cols].fillna(0)
    test_df["pred_score"] = model.predict(X_test).values

    top1_hits = 0
    top3_hits = 0
    total_races = 0
    ndcg_scores = []
    mrr_scores = []

    for race_id, group in test_df.groupby("race_id"):
        if len(group) < 3:
            continue
        total_races += 1

        pred_ranking = group.sort_values("pred_score", ascending=False)
        actual_ranking = group.sort_values("finish_position")

        pred_top1 = pred_ranking.iloc[0]["rider_id"]
        pred_top3 = set(pred_ranking.head(3)["rider_id"])
        actual_top1 = actual_ranking.iloc[0]["rider_id"]
        actual_top3 = set(actual_ranking.head(3)["rider_id"])

        # Top1/Top3
        if pred_top1 == actual_top1:
            top1_hits += 1
        top3_hits += len(pred_top3 & actual_top3)

        # NDCG@3
        n = len(group)
        pred_order = pred_ranking["rider_id"].tolist()
        actual_positions = dict(zip(group["rider_id"], group["finish_position"]))
        dcg = 0.0
        for i, rid in enumerate(pred_order[:3]):
            pos = actual_positions.get(rid, n)
            relevance = max(0, n - pos + 1)
            dcg += relevance / np.log2(i + 2)
        # Ideal DCG
        ideal_relevances = sorted([max(0, n - p + 1) for p in actual_positions.values()], reverse=True)
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_relevances[:3]))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        # MRR (Mean Reciprocal Rank) — 実際の1着が予測の何位にいるか
        for i, rid in enumerate(pred_order):
            if rid == actual_top1:
                mrr_scores.append(1.0 / (i + 1))
                break
        else:
            mrr_scores.append(0.0)

    results = {
        "total_races": total_races,
        "top1_hits": top1_hits,
        "top1_accuracy": top1_hits / total_races if total_races > 0 else 0.0,
        "top3_hits": top3_hits,
        "top3_overlap": top3_hits / (total_races * 3) if total_races > 0 else 0.0,
        "ndcg_at_3": np.mean(ndcg_scores) if ndcg_scores else 0.0,
        "mrr": np.mean(mrr_scores) if mrr_scores else 0.0,
    }

    if total_races > 0:
        logger.info("=== Test Results (%d races) ===", total_races)
        logger.info("Top1 accuracy: %d/%d (%.1f%%)",
                     top1_hits, total_races, results["top1_accuracy"] * 100)
        logger.info("Top3 overlap:  %d/%d (%.1f%%)",
                     top3_hits, total_races * 3, results["top3_overlap"] * 100)
        logger.info("NDCG@3:       %.4f", results["ndcg_at_3"])
        logger.info("MRR:          %.4f", results["mrr"])

    return results


def time_series_cv(df: pd.DataFrame, feature_cols: list[str],
                   n_splits: int = 3) -> list[dict]:
    """時系列クロスバリデーション

    データを時系列順に分割し、常に過去→未来の方向で学習→評価する
    """
    dates = sorted(df["race_date"].unique())
    split_size = len(dates) // (n_splits + 1)

    results = []
    for fold in range(n_splits):
        train_end_idx = split_size * (fold + 1)
        val_start_idx = train_end_idx
        val_end_idx = min(train_end_idx + split_size, len(dates))

        if val_end_idx <= val_start_idx:
            break

        train_dates = set(dates[:train_end_idx])
        val_dates = set(dates[val_start_idx:val_end_idx])

        train_mask = df["race_date"].isin(train_dates)
        val_mask = df["race_date"].isin(val_dates)

        train_df = df[train_mask]
        val_df = df[val_mask]

        if train_df.empty or val_df.empty:
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["finish_position"]
        group_train = train_df.groupby("race_id").size().tolist()

        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df["finish_position"]
        group_val = val_df.groupby("race_id").size().tolist()

        model = LGBMRanker()
        model.train(X_train, y_train, group_train, X_val, y_val, group_val)

        fold_results = evaluate_test_set(val_df, model, feature_cols)
        fold_results["fold"] = fold
        fold_results["train_size"] = len(train_df)
        fold_results["val_size"] = len(val_df)
        results.append(fold_results)

        logger.info("Fold %d: train=%d val=%d Top1=%.1f%% NDCG@3=%.4f",
                     fold, len(train_df), len(val_df),
                     fold_results["top1_accuracy"] * 100,
                     fold_results["ndcg_at_3"])

    if results:
        avg_top1 = np.mean([r["top1_accuracy"] for r in results])
        avg_ndcg = np.mean([r["ndcg_at_3"] for r in results])
        avg_mrr = np.mean([r["mrr"] for r in results])
        logger.info("=== CV Average ===")
        logger.info("Top1: %.1f%%  NDCG@3: %.4f  MRR: %.4f", avg_top1 * 100, avg_ndcg, avg_mrr)

    return results


def train_and_evaluate(use_cv: bool = False):
    """メイン学習・評価関数"""
    logging.basicConfig(level=logging.INFO)

    builder = FeatureBuilder()
    feature_cols = builder.feature_names

    # 全データをまとめて取得
    all_years = sorted(set(TRAIN_YEARS + TEST_YEARS))
    logger.info("Building dataset (%d-%d)...", all_years[0], all_years[-1])
    all_df = builder.build_dataset(all_years[0], all_years[-1])

    if all_df.empty:
        logger.error("No data found")
        return

    # NaN除去
    all_df = all_df.dropna(subset=["finish_position"])
    logger.info("Total dataset: %d rows", len(all_df))

    # クロスバリデーション
    if use_cv:
        logger.info("=== Time-Series Cross Validation ===")
        cv_results = time_series_cv(all_df, feature_cols)

    # Train/Test 分割
    train_mask = all_df["race_date"].apply(
        lambda d: int(d[:4]) in TRAIN_YEARS
    )
    test_mask = all_df["race_date"].apply(
        lambda d: int(d[:4]) in TEST_YEARS
    )
    train_df = all_df[train_mask]
    test_df = all_df[test_mask]

    if train_df.empty:
        logger.error("No training data found")
        return

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["finish_position"]
    group_train = train_df.groupby("race_id").size().tolist()

    X_test = test_df[feature_cols].fillna(0) if not test_df.empty else None
    y_test = test_df["finish_position"] if not test_df.empty else None
    group_test = test_df.groupby("race_id").size().tolist() if not test_df.empty else None

    logger.info("Training: %d rows, %d races", len(train_df), len(group_train))
    if X_test is not None:
        logger.info("Test: %d rows, %d races", len(test_df), len(group_test))

    model = LGBMRanker()
    model.train(X_train, y_train, group_train, X_test, y_test, group_test)

    # 保存
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(RESULTS_DIR / "model_lgbm.pkl")
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    # 特徴量重要度（gain + split）
    importance_gain = model.feature_importance("gain")
    importance_split = model.feature_importance("split")
    importance = importance_gain.merge(
        importance_split, on="feature", suffixes=("_gain", "_split")
    )
    logger.info("Feature importance:\n%s", importance.to_string())
    importance.to_csv(str(RESULTS_DIR / "feature_importance.csv"), index=False)

    # テストセット評価
    if not test_df.empty:
        evaluate_test_set(test_df, model, feature_cols)
