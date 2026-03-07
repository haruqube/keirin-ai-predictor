"""モデル学習スクリプト"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.trainer import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate()
