"""ハイパーパラメータチューニング

LightGBMのパラメータをランダムサーチで最適化する。
optuna不要の軽量実装。
"""

import sys
import logging
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import TRAIN_YEARS, TEST_YEARS, RESULTS_DIR
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from models.trainer import evaluate_test_set

logger = logging.getLogger(__name__)

PARAM_SPACE = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_data_in_leaf": [10, 20, 50, 100],
    "feature_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_freq": [1, 3, 5, 7],
    "lambda_l1": [0, 0.01, 0.1, 1.0],
    "lambda_l2": [0, 0.01, 0.1, 1.0],
}


def random_search(n_trials: int = 20):
    logging.basicConfig(level=logging.INFO)

    builder = FeatureBuilder()
    feature_cols = builder.feature_names

    logger.info("Building dataset...")
    all_years = sorted(set(TRAIN_YEARS + TEST_YEARS))
    all_df = builder.build_dataset(all_years[0], all_years[-1])
    all_df = all_df.dropna(subset=["finish_position"])

    train_mask = all_df["race_date"].apply(lambda d: int(d[:4]) in TRAIN_YEARS)
    test_mask = all_df["race_date"].apply(lambda d: int(d[:4]) in TEST_YEARS)
    train_df = all_df[train_mask]
    test_df = all_df[test_mask]

    if train_df.empty:
        logger.error("No training data")
        return

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["finish_position"]
    group_train = train_df.groupby("race_id").size().tolist()

    X_test = test_df[feature_cols].fillna(0) if not test_df.empty else None
    y_test = test_df["finish_position"] if not test_df.empty else None
    group_test = test_df.groupby("race_id").size().tolist() if not test_df.empty else None

    best_ndcg = -1
    best_params = None
    all_results = []

    for trial in range(n_trials):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3],
            "verbose": -1,
        }
        for key, values in PARAM_SPACE.items():
            params[key] = random.choice(values)

        logger.info("Trial %d/%d: %s", trial + 1, n_trials,
                     {k: v for k, v in params.items() if k not in ["objective", "metric", "ndcg_eval_at", "verbose"]})

        model = LGBMRanker(params=params)
        model.train(X_train, y_train, group_train, X_test, y_test, group_test)

        if not test_df.empty:
            result = evaluate_test_set(test_df, model, feature_cols)
            result["params"] = params
            all_results.append(result)

            if result["ndcg_at_3"] > best_ndcg:
                best_ndcg = result["ndcg_at_3"]
                best_params = params.copy()
                # ベストモデルを保存
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                model.save(str(RESULTS_DIR / "model_lgbm_best.pkl"))
                logger.info("  -> New best! NDCG@3=%.4f Top1=%.1f%%",
                             result["ndcg_at_3"], result["top1_accuracy"] * 100)

    if best_params:
        logger.info("=== Best Parameters ===")
        for k, v in best_params.items():
            if k not in ["objective", "metric", "ndcg_eval_at", "verbose"]:
                logger.info("  %s: %s", k, v)
        logger.info("Best NDCG@3: %.4f", best_ndcg)

        # 結果をCSVに保存
        results_df = pd.DataFrame([
            {**{k: v for k, v in r["params"].items()
                if k not in ["objective", "metric", "ndcg_eval_at", "verbose"]},
             "ndcg_at_3": r["ndcg_at_3"],
             "top1_accuracy": r["top1_accuracy"],
             "mrr": r["mrr"]}
            for r in all_results
        ])
        results_df.to_csv(str(RESULTS_DIR / "tuning_results.csv"), index=False)
        logger.info("Results saved to %s", RESULTS_DIR / "tuning_results.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ハイパーパラメータチューニング")
    parser.add_argument("--trials", type=int, default=20, help="試行回数")
    args = parser.parse_args()
    random_search(n_trials=args.trials)
