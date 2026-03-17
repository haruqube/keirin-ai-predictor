"""LightGBM LambdaRankモデル（改善版）

改善点:
- relevance計算をレース単位に修正（グローバルmax_posではなくグループ内max）
- feature_importance に split ベースも追加
- 予測時の安全性向上
"""

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from models.base import BasePredictor
from config import LGBM_PARAMS, LGBM_NUM_BOOST_ROUND, LGBM_EARLY_STOPPING_ROUNDS


class LGBMRanker(BasePredictor):
    """LightGBM LambdaRankによる着順予測"""

    def __init__(self, params: dict | None = None):
        self.params = params or LGBM_PARAMS.copy()
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        group_train: list[int],
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        group_val: list[int] | None = None,
    ):
        self.feature_names = list(X_train.columns)

        # 着順をrelevanceに変換（レース単位で正規化）
        y_relevance = self._to_relevance(y_train, group_train)

        train_data = lgb.Dataset(
            X_train, label=y_relevance, group=group_train,
            feature_name=self.feature_names,
        )

        callbacks = [lgb.log_evaluation(50)]
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None and group_val is not None:
            y_val_rel = self._to_relevance(y_val, group_val)
            val_data = lgb.Dataset(
                X_val, label=y_val_rel, group=group_val,
                feature_name=self.feature_names,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, first_metric_only=True))

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=LGBM_NUM_BOOST_ROUND,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        logger.info("Trees: %d (best_iteration=%d)", self.model.num_trees(), self.model.best_iteration)

    @staticmethod
    def _to_relevance(y: pd.Series, groups: list[int]) -> np.ndarray:
        """着順をグループ（レース）単位でrelevanceに変換

        各レース内で: relevance = max_pos_in_race - finish_position + 1
        これにより出走数が異なるレース間でも公平なrelevanceスコアになる
        """
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

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained")
        # 学習時に無かった特徴量を0で補完
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        scores = self.model.predict(X_aligned)
        return pd.Series(scores, index=X.index)

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained")
        importance = self.model.feature_importance(importance_type=importance_type)
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "params": self.params,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.params = data["params"]
