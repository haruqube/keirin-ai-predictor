"""特徴量ビルダー基底クラス"""

from abc import ABC, abstractmethod


class BaseFeatureBuilder(ABC):
    """特徴量ビルダーの基底クラス"""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        ...

    @abstractmethod
    def build(self, race_id: str, rider_id: str, race_date: str) -> dict:
        ...
