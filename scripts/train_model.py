"""モデル学習エントリーポイント"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.trainer import train_and_evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競輪AIモデル学習")
    parser.add_argument("--cv", action="store_true", help="時系列クロスバリデーションを実行")
    args = parser.parse_args()
    train_and_evaluate(use_cv=args.cv)
