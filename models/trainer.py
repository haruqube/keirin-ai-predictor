"""モデル学習・評価"""

import logging
import numpy as np
import pandas as pd

from config import TRAIN_YEARS, TEST_YEARS, RESULTS_DIR
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker

logger = logging.getLogger(__name__)


def train_and_evaluate():
    logging.basicConfig(level=logging.INFO)

    builder = FeatureBuilder()
    feature_cols = builder.feature_names

    logger.info("Building training dataset (%s-%s)...", TRAIN_YEARS[0], TRAIN_YEARS[-1])
    train_df = builder.build_dataset(TRAIN_YEARS[0], TRAIN_YEARS[-1])

    logger.info("Building test dataset (%s-%s)...", TEST_YEARS[0], TEST_YEARS[-1])
    test_df = builder.build_dataset(TEST_YEARS[0], TEST_YEARS[-1])

    if train_df.empty:
        logger.error("No training data found")
        return

    # NaN除去
    train_df = train_df.dropna(subset=["finish_position"])
    test_df = test_df.dropna(subset=["finish_position"]) if not test_df.empty else test_df

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["finish_position"]

    # グループ（レースごとの選手数）
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
    model_path = str(RESULTS_DIR / "model_lgbm.pkl")
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    # 特徴量重要度
    importance = model.feature_importance()
    logger.info("Feature importance:\n%s", importance.to_string())

    # テストセット評価
    if X_test is not None and not test_df.empty:
        test_df = test_df.copy()
        test_df["pred_score"] = model.predict(X_test).values

        top1_hits = 0
        top3_hits = 0
        total_races = 0

        for race_id, group in test_df.groupby("race_id"):
            if len(group) < 3:
                continue
            total_races += 1
            pred_top3 = set(group.nlargest(3, "pred_score")["rider_id"])
            actual_top1 = group.loc[group["finish_position"].idxmin(), "rider_id"]
            actual_top3 = set(group.nsmallest(3, "finish_position")["rider_id"])

            pred_top1 = group.loc[group["pred_score"].idxmax(), "rider_id"]
            if pred_top1 == actual_top1:
                top1_hits += 1
            top3_hits += len(pred_top3 & actual_top3)

        if total_races > 0:
            logger.info("=== Test Results ===")
            logger.info("Top1 accuracy: %d/%d (%.1f%%)", top1_hits, total_races, top1_hits / total_races * 100)
            logger.info("Top3 overlap:  %d/%d (%.1f%%)", top3_hits, total_races * 3, top3_hits / (total_races * 3) * 100)
