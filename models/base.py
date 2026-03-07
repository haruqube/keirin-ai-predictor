"""予測モデル基底クラス"""

from abc import ABC, abstractmethod
import pandas as pd


class BasePredictor(ABC):
    @abstractmethod
    def train(self, X_train, y_train, group_train, X_val=None, y_val=None, group_val=None):
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        ...

    @abstractmethod
    def save(self, path: str):
        ...

    @abstractmethod
    def load(self, path: str):
        ...
