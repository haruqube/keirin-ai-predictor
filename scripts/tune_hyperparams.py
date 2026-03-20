"""Optunaによるハイパーパラメータ最適化

時系列CVのNDCG@3を目的関数としてLightGBMのパラメータを自動探索する。
最適パラメータで最終モデルを保存する。
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Optuna is required: pip install optuna")
    sys.exit(1)

from config import TRAIN_YEARS, TEST_YEARS, RESULTS_DIR
from features.builder import FeatureBuilder

logger = logging.getLogger(__name__)


def ndcg_at_3(test_df: pd.DataFrame, predictions: np.ndarray) -> float:
    """NDCG@3を計算"""
    test_df = test_df.copy()
    test_df["pred_score"] = predictions
    ndcg_scores = []

    for race_id, group in test_df.groupby("race_id"):
        if len(group) < 3:
            continue
        n = len(group)
        pred_order = group.sort_values("pred_score", ascending=False)["rider_id"].tolist()
        actual_positions = dict(zip(group["rider_id"], group["finish_position"]))

        dcg = 0.0
        for i, rid in enumerate(pred_order[:3]):
            pos = actual_positions.get(rid, n)
            relevance = max(0, n - pos + 1)
            dcg += relevance / np.log2(i + 2)

        ideal_relevances = sorted(
            [max(0, n - p + 1) for p in actual_positions.values()], reverse=True
        )
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_relevances[:3]))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def top1_accuracy(test_df: pd.DataFrame, predictions: np.ndarray) -> float:
    """◎的中率を計算"""
    test_df = test_df.copy()
    test_df["pred_score"] = predictions
    hits = 0
    total = 0

    for race_id, group in test_df.groupby("race_id"):
        if len(group) < 3:
            continue
        total += 1
        pred_top1 = group.loc[group["pred_score"].idxmax(), "rider_id"]
        actual_top1 = group.loc[group["finish_position"].idxmin(), "rider_id"]
        if pred_top1 == actual_top1:
            hits += 1

    return hits / total if total > 0 else 0.0


def time_series_cv_score(df: pd.DataFrame, feature_cols: list[str],
                          params: dict, n_splits: int = 3) -> float:
    """時系列CVでNDCG@3の平均スコアを返す"""
    dates = sorted(df["race_date"].unique())
    split_size = len(dates) // (n_splits + 1)
    ndcg_scores = []

    for fold in range(n_splits):
        train_end_idx = split_size * (fold + 1)
        val_start_idx = train_end_idx
        val_end_idx = min(train_end_idx + split_size, len(dates))

        if val_end_idx <= val_start_idx:
            break

        train_dates = set(dates[:train_end_idx])
        val_dates = set(dates[val_start_idx:val_end_idx])

        train_df = df[df["race_date"].isin(train_dates)]
        val_df = df[df["race_date"].isin(val_dates)]

        if train_df.empty or val_df.empty:
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["finish_position"]
        group_train = train_df.groupby("race_id").size().tolist()

        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df["finish_position"]
        group_val = val_df.groupby("race_id").size().tolist()

        # relevance変換（レース単位）
        y_rel = _to_relevance(y_train, group_train)
        y_val_rel = _to_relevance(y_val, group_val)

        train_data = lgb.Dataset(X_train, label=y_rel, group=group_train)
        val_data = lgb.Dataset(X_val, label=y_val_rel, group=group_val)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        predictions = model.predict(X_val)
        score = ndcg_at_3(val_df, predictions)
        ndcg_scores.append(score)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def _to_relevance(y: pd.Series, groups: list[int]) -> np.ndarray:
    """着順をグループ単位でrelevanceに変換"""
    relevance = np.zeros(len(y), dtype=np.float32)
    offset = 0
    for size in groups:
        group_y = y.iloc[offset:offset + size]
        max_pos = int(group_y.max()) if len(group_y) > 0 else 9
        relevance[offset:offset + size] = np.clip(
            max_pos - group_y.values + 1, 0, None
        )
        offset += size
    return relevance


def objective(trial: optuna.Trial, df: pd.DataFrame,
              feature_cols: list[str]) -> float:
    """Optuna目的関数: 時系列CVのNDCG@3を最大化"""
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3],
        "verbose": -1,
        "bagging_freq": 5,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
    }

    return time_series_cv_score(df, feature_cols, params)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optunaハイパーパラメータ最適化")
    parser.add_argument("--n-trials", type=int, default=50, help="試行回数 (default: 50)")
    parser.add_argument("--apply", action="store_true", help="最適パラメータで最終モデルを保存")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    builder = FeatureBuilder()
    feature_cols = builder.feature_names

    logger.info("Building dataset...")
    all_years = sorted(set(TRAIN_YEARS + TEST_YEARS))
    all_df = builder.build_dataset(all_years[0], all_years[-1])
    all_df = all_df.dropna(subset=["finish_position"])
    logger.info("Dataset: %d rows, %d features", len(all_df), len(feature_cols))

    # Optuna探索
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, all_df, feature_cols),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    logger.info("=" * 60)
    logger.info("Best NDCG@3: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    # Top5を表示
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False)
    logger.info("\nTop 5 trials:\n%s",
                trials_df[["number", "value"] +
                          [c for c in trials_df.columns if c.startswith("params_")]
                          ].head(5).to_string())

    # 最適パラメータで評価（85/15分割）
    best_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3],
        "verbose": -1,
        "bagging_freq": 5,
        **study.best_params,
    }

    all_dates = sorted(all_df["race_date"].unique())
    val_cutoff = all_dates[int(len(all_dates) * 0.85)]
    train_df = all_df[all_df["race_date"] < val_cutoff]
    test_df = all_df[all_df["race_date"] >= val_cutoff]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["finish_position"]
    group_train = train_df.groupby("race_id").size().tolist()
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["finish_position"]
    group_test = test_df.groupby("race_id").size().tolist()

    y_rel = _to_relevance(y_train, group_train)
    y_val_rel = _to_relevance(y_test, group_test)

    train_data = lgb.Dataset(X_train, label=y_rel, group=group_train)
    val_data = lgb.Dataset(X_test, label=y_val_rel, group=group_test)

    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )

    predictions = model.predict(X_test)
    final_ndcg = ndcg_at_3(test_df, predictions)
    final_top1 = top1_accuracy(test_df, predictions)

    logger.info("=" * 60)
    logger.info("Final evaluation (85/15 split):")
    logger.info("  NDCG@3: %.4f", final_ndcg)
    logger.info("  Top1:   %.1f%%", final_top1 * 100)

    # パラメータ出力（config.pyにコピペ用）
    logger.info("\n=== config.py LGBM_PARAMS に設定するパラメータ ===")
    config_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3],
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in study.best_params.items()},
        "bagging_freq": 5,
        "verbose": -1,
    }
    logger.info("LGBM_PARAMS = %s", config_params)

    if args.apply:
        # 最終モデル保存
        from models.lgbm_ranker import LGBMRanker
        ranker = LGBMRanker(params=best_params)
        ranker.train(X_train, y_train, group_train, X_test, y_test, group_test)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = str(RESULTS_DIR / "model_lgbm.pkl")
        ranker.save(model_path)
        logger.info("Model saved to %s", model_path)

        importance = ranker.feature_importance()
        logger.info("Feature importance:\n%s", importance.to_string())


if __name__ == "__main__":
    main()
