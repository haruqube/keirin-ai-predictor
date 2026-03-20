# -*- coding: utf-8 -*-
"""月次モデル再学習スクリプト

Walk-Forward方式:
- 全レース結果データを使用
- 最新3ヶ月をバリデーション、それ以前を学習に使用
- 新旧モデルの比較評価を実施し、改善時のみモデルを更新
"""

import sys, io, logging, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import RESULTS_DIR, LGBM_PARAMS
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from models.trainer import evaluate_test_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="月次モデル再学習")
    parser.add_argument("--force", action="store_true", help="改善なしでも強制更新")
    parser.add_argument("--val-months", type=int, default=3, help="バリデーション期間（月数）")
    args = parser.parse_args()

    builder = FeatureBuilder()
    feature_cols = builder.feature_names

    # 全データ取得（利用可能な最古〜最新）
    logger.info("Building full dataset...")
    all_df = builder.build_dataset(2022, 2026)
    all_df = all_df.dropna(subset=['finish_position'])

    if all_df.empty:
        logger.error("No data found")
        return

    # 日付でソート
    all_dates = sorted(all_df['race_date'].unique())
    latest_date = all_dates[-1]
    logger.info(f"Data range: {all_dates[0]} to {latest_date} ({len(all_dates)} days, {len(all_df)} rows)")

    # バリデーション期間を決定（最新N ヶ月）
    latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
    val_start_dt = latest_dt - timedelta(days=args.val_months * 30)
    val_cutoff = val_start_dt.strftime('%Y-%m-%d')

    train_df = all_df[all_df['race_date'] < val_cutoff]
    val_df = all_df[all_df['race_date'] >= val_cutoff]

    logger.info(f"Train: {len(train_df)} rows (< {val_cutoff})")
    logger.info(f"Val:   {len(val_df)} rows (>= {val_cutoff})")

    if train_df.empty or val_df.empty:
        logger.error("Insufficient data for split")
        return

    # ========================================
    # 新モデル学習
    # ========================================
    logger.info("=== Training new model ===")
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['finish_position']
    group_train = train_df.groupby('race_id').size().tolist()

    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['finish_position']
    group_val = val_df.groupby('race_id').size().tolist()

    new_model = LGBMRanker()
    new_model.train(X_train, y_train, group_train, X_val, y_val, group_val)

    new_results = evaluate_test_set(val_df, new_model, feature_cols)

    # ========================================
    # 現行モデルを同じバリデーションで評価
    # ========================================
    model_path = str(RESULTS_DIR / 'model_lgbm.pkl')
    try:
        old_model = LGBMRanker()
        old_model.load(model_path)
        old_results = evaluate_test_set(val_df, old_model, old_model.feature_names)
        has_old = True
    except Exception as e:
        logger.warning(f"Could not load old model: {e}")
        has_old = False
        old_results = {}

    # ========================================
    # 比較
    # ========================================
    print(f"\n{'='*70}")
    print(f"  モデル比較 (Val: {val_cutoff} 〜 {latest_date})")
    print(f"{'='*70}")
    print(f"{'指標':<20} {'現行モデル':>12} {'新モデル':>12} {'差分':>10}")
    print(f"{'-'*70}")

    metrics = [
        ('Top1精度', 'top1_accuracy'),
        ('Top3重複率', 'top3_overlap'),
        ('NDCG@3', 'ndcg_at_3'),
        ('MRR', 'mrr'),
    ]

    improved = 0
    for label, key in metrics:
        old_v = old_results.get(key, 0) if has_old else 0
        new_v = new_results.get(key, 0)
        diff = new_v - old_v
        marker = ' ↑' if diff > 0.001 else (' ↓' if diff < -0.001 else '')
        if diff > 0.001:
            improved += 1
        if key in ('top1_accuracy', 'top3_overlap'):
            print(f"  {label:<18} {old_v*100:>11.1f}% {new_v*100:>11.1f}% {diff*100:>+9.1f}%{marker}")
        else:
            print(f"  {label:<18} {old_v:>12.4f} {new_v:>12.4f} {diff:>+10.4f}{marker}")

    # ========================================
    # 更新判定
    # ========================================
    should_update = args.force or improved >= 2 or (
        new_results.get('ndcg_at_3', 0) > old_results.get('ndcg_at_3', 0)
    )

    if should_update:
        new_model.save(model_path)
        print(f"\n  ★ モデル更新完了: {model_path}")
        print(f"    学習データ: 〜 {val_cutoff}")
        print(f"    バリデーション: {val_cutoff} 〜 {latest_date}")

        # 特徴量重要度保存
        imp_gain = new_model.feature_importance('gain')
        imp_split = new_model.feature_importance('split')
        imp = imp_gain.merge(imp_split, on='feature', suffixes=('_gain', '_split'))
        imp.to_csv(str(RESULTS_DIR / 'feature_importance.csv'), index=False)
        print(f"    特徴量重要度保存完了")
    else:
        print(f"\n  ✗ 新モデルの改善が不十分のため更新スキップ")
        print(f"    --force オプションで強制更新可能")


if __name__ == '__main__':
    main()
