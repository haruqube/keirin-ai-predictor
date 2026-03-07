"""LightGBM LambdaRankモデル"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

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

        # 着順をrelevanceに変換（着順が小さいほど高スコア）
        max_pos = int(y_train.max()) if y_train.max() else 9
        y_relevance = max_pos - y_train + 1
        y_relevance = y_relevance.clip(lower=0)

        train_data = lgb.Dataset(
            X_train, label=y_relevance, group=group_train,
            feature_name=self.feature_names,
        )

        callbacks = [lgb.log_evaluation(50)]
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None and group_val is not None:
            y_val_rel = max_pos - y_val + 1
            y_val_rel = y_val_rel.clip(lower=0)
            val_data = lgb.Dataset(
                X_val, label=y_val_rel, group=group_val,
                feature_name=self.feature_names,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS))

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=LGBM_NUM_BOOST_ROUND,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained")
        scores = self.model.predict(X[self.feature_names])
        return pd.Series(scores, index=X.index)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained")
        importance = self.model.feature_importance(importance_type="gain")
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
