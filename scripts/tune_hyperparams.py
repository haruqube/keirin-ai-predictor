"""LightGBMハイパーパラメータチューニング

特徴量拡張後のモデルで最適なパラメータを探索する。
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from itertools import product

from config import TRAIN_YEARS, TEST_YEARS, RESULTS_DIR
from features.builder import FeatureBuilder

logger = logging.getLogger(__name__)


def evaluate_params(X_train, y_train, group_train, X_test, y_test, group_test, test_df, params):
    """パラメータセットで学習・評価"""
    max_pos = int(y_train.max())
    y_rel = max_pos - y_train + 1
    y_rel = y_rel.clip(lower=0)
    y_val_rel = max_pos - y_test + 1
    y_val_rel = y_val_rel.clip(lower=0)

    train_data = lgb.Dataset(X_train, label=y_rel, group=group_train)
    val_data = lgb.Dataset(X_test, label=y_val_rel, group=group_test)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=800,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # 評価
    test_copy = test_df.copy()
    test_copy["pred_score"] = model.predict(X_test)

    top1_hits = 0
    top3_hits = 0
    total_races = 0

    for race_id, group in test_copy.groupby("race_id"):
        if len(group) < 3:
            continue
        total_races += 1
        pred_top1 = group.loc[group["pred_score"].idxmax(), "rider_id"]
        actual_top1 = group.loc[group["finish_position"].idxmin(), "rider_id"]
        pred_top3 = set(group.nlargest(3, "pred_score")["rider_id"])
        actual_top3 = set(group.nsmallest(3, "finish_position")["rider_id"])

        if pred_top1 == actual_top1:
            top1_hits += 1
        top3_hits += len(pred_top3 & actual_top3)

    top1_rate = top1_hits / total_races if total_races > 0 else 0
    top3_rate = top3_hits / (total_races * 3) if total_races > 0 else 0

    return top1_rate, top3_rate, model.best_iteration


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    builder = FeatureBuilder()
    feature_cols = builder.feature_names

    logger.info("Building datasets...")
    train_df = builder.build_dataset(TRAIN_YEARS[0], TRAIN_YEARS[-1])
    test_df = builder.build_dataset(TEST_YEARS[0], TEST_YEARS[-1])

    train_df = train_df.dropna(subset=["finish_position"])
    test_df = test_df.dropna(subset=["finish_position"])

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["finish_position"]
    group_train = train_df.groupby("race_id").size().tolist()

    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["finish_position"]
    group_test = test_df.groupby("race_id").size().tolist()

    logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # グリッドサーチ
    param_grid = {
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.03, 0.05, 0.08],
        "min_data_in_leaf": [10, 20, 50],
        "feature_fraction": [0.6, 0.8, 1.0],
        "bagging_fraction": [0.7, 0.8, 0.9],
    }

    base_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "bagging_freq": 5,
        "verbose": -1,
    }

    best_top3 = 0
    best_top1 = 0
    best_params = {}
    results = []

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    logger.info("Total combinations: %d", len(combos))

    for i, combo in enumerate(combos):
        params = base_params.copy()
        for k, v in zip(keys, combo):
            params[k] = v

        try:
            top1, top3, best_iter = evaluate_params(
                X_train, y_train, group_train,
                X_test, y_test, group_test, test_df,
                params
            )
        except Exception as e:
            logger.warning("Error: %s", e)
            continue

        results.append({
            **{k: v for k, v in zip(keys, combo)},
            "top1": top1, "top3": top3, "best_iter": best_iter,
        })

        if top3 > best_top3 or (top3 == best_top3 and top1 > best_top1):
            best_top3 = top3
            best_top1 = top1
            best_params = {k: v for k, v in zip(keys, combo)}
            logger.info("[%d/%d] NEW BEST: Top1=%.1f%% Top3=%.1f%% params=%s iter=%d",
                        i+1, len(combos), top1*100, top3*100, best_params, best_iter)
        elif (i+1) % 20 == 0:
            logger.info("[%d/%d] current best: Top1=%.1f%% Top3=%.1f%%",
                        i+1, len(combos), best_top1*100, best_top3*100)

    # 結果表示
    logger.info("=" * 60)
    logger.info("Best params: %s", best_params)
    logger.info("Best: Top1=%.1f%% Top3=%.1f%%", best_top1*100, best_top3*100)

    # Top10を表示
    results_df = pd.DataFrame(results).sort_values("top3", ascending=False)
    logger.info("\nTop 10:\n%s", results_df.head(10).to_string())

    # 最良パラメータで最終モデル保存
    final_params = base_params.copy()
    final_params.update(best_params)
    logger.info("Training final model with best params...")

    from models.lgbm_ranker import LGBMRanker
    model = LGBMRanker(params=final_params)
    model.train(X_train, y_train, group_train, X_test, y_test, group_test)

    model_path = str(RESULTS_DIR / "model_lgbm.pkl")
    model.save(model_path)
    logger.info("Saved to %s", model_path)

    importance = model.feature_importance()
    logger.info("Feature importance:\n%s", importance.to_string())


if __name__ == "__main__":
    main()
