# -*- coding: utf-8 -*-
"""真のOut-of-Sample検証

学習: 2022-2024のみ → テスト: 2025のみ
現行モデル(2022-2025学習)との比較で、in-sample過学習の程度を定量化する
"""

import sys, io, re, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from config import RESULTS_DIR, CACHE_DIR, LGBM_PARAMS
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_nisyatan(race_id):
    html_path = CACHE_DIR / f"keirin_netkeiba_com_db_result__race_id_{race_id}.html"
    if not html_path.exists(): return None
    try:
        html = html_path.read_text(encoding='utf-8')
    except: return None
    soup = BeautifulSoup(html, 'lxml')
    pay = soup.select_one('.Payout_Detail_Table')
    if not pay: return None
    text = pay.get_text(' ', strip=True)
    m = re.search(r'２車単\s+(\d+[->＞\s]+\d+)\s+([\d,]+)円\s+(\d+)人気', text)
    if not m: return None
    bikes = [int(b) for b in re.findall(r'\d+', m.group(1))]
    return {'bikes': bikes, 'amount': int(m.group(2).replace(',', ''))}


def run_backtest(model, feature_cols, test_df, label):
    """指定モデルで2025年データをバックテスト"""
    X = test_df[feature_cols].reindex(columns=feature_cols, fill_value=0).fillna(0)
    test_df = test_df.copy()
    test_df['pred_score'] = model.predict(X).values

    races = []
    top1_hits = 0
    total_races = 0
    for race_id, group in test_df.groupby('race_id'):
        if len(group) < 5: continue
        total_races += 1
        ranked = group.sort_values('pred_score', ascending=False)
        bikes = ranked['bike_number'].astype(int).tolist()
        scores = ranked['pred_score'].tolist()
        actual = group.sort_values('finish_position')
        if ranked.iloc[0]['rider_id'] == actual.iloc[0]['rider_id']:
            top1_hits += 1

        payout = parse_nisyatan(race_id)
        if not payout: continue

        a1, a2 = payout['bikes'][0], payout['bikes'][1]
        hit_at = None
        if bikes[0] == a1:
            for i in range(1, min(6, len(bikes))):
                if bikes[i] == a2:
                    hit_at = i
                    break

        honmei = scores[0] - scores[1]
        races.append({
            'race_id': race_id,
            'year_month': race_id[:4] + '-' + (group['race_date'].iloc[0][5:7] if 'race_date' in group.columns else '??'),
            'honmei': honmei,
            'hit_at': hit_at,
            'payout': payout['amount'],
        })

    top1_rate = top1_hits / total_races * 100 if total_races > 0 else 0
    print(f"\n{'='*90}")
    print(f"  [{label}]")
    print(f"{'='*90}")
    print(f"  全レース数: {total_races}  ◎1着率: {top1_rate:.1f}%")

    # 各戦略でROI計算
    for pts, th_label in [(3, '3pt'), (4, '4pt')]:
        for h_th in [0.35, 0.50, 0.65, 0.89]:
            sub = [r for r in races if r['honmei'] >= h_th]
            if len(sub) < 20: continue
            n = len(sub)
            hits = [r for r in sub if r['hit_at'] is not None and r['hit_at'] <= pts]
            bet = n * pts * 500
            ret = sum(r['payout'] * 5 for r in hits)
            roi = ret / bet * 100 if bet > 0 else 0
            hit_rate = len(hits) / n * 100
            print(f"  本命>={h_th:.2f} {th_label}: {n:>5}R  的中{len(hits):>4} ({hit_rate:.1f}%)  投資{bet:>10,}円  ROI {roi:.1f}%  利益{ret-bet:>+10,.0f}円")

    # 月別ROI (3pt, 本命>=0.89)
    print(f"\n  --- 月別ROI（本命>=0.89, 3pt） ---")
    monthly_rois = []
    for ym in sorted(set(r['year_month'] for r in races)):
        sub = [r for r in races if r['year_month'] == ym and r['honmei'] >= 0.89]
        if len(sub) < 3: continue
        hits = [r for r in sub if r['hit_at'] is not None and r['hit_at'] <= 3]
        bet = len(sub) * 3 * 500
        ret = sum(r['payout'] * 5 for r in hits)
        roi = ret / bet * 100 if bet > 0 else 0
        monthly_rois.append(roi)
        marker = ' ★' if roi < 100 else ''
        print(f"    {ym}: {len(sub):>4}R  的中{len(hits):>3} ({len(hits)/len(sub)*100:.0f}%)  ROI {roi:.1f}%{marker}")

    if monthly_rois:
        profitable = sum(1 for r in monthly_rois if r > 100)
        print(f"    黒字月: {profitable}/{len(monthly_rois)}  中央値ROI: {np.median(monthly_rois):.1f}%")

    return races


def main():
    builder = FeatureBuilder()
    feature_cols = builder.feature_names

    # ========================================
    # Step 1: 2022-2024で学習（2025を一切見ない）
    # ========================================
    print("="*90)
    print("  Step 1: 2022-2024でモデル学習（真のOOS用）")
    print("="*90)

    logger.info("Building training dataset (2022-2024)...")
    train_df = builder.build_dataset(2022, 2024)
    train_df = train_df.dropna(subset=['finish_position'])
    logger.info(f"Training data: {len(train_df)} rows")

    # 時系列分割: 2022-2023=学習, 2024前半=検証
    all_dates = sorted(train_df['race_date'].unique())
    val_cutoff = all_dates[int(len(all_dates) * 0.85)]
    train_split = train_df[train_df['race_date'] < val_cutoff]
    val_split = train_df[train_df['race_date'] >= val_cutoff]

    X_train = train_split[feature_cols].fillna(0)
    y_train = train_split['finish_position']
    group_train = train_split.groupby('race_id').size().tolist()

    X_val = val_split[feature_cols].fillna(0)
    y_val = val_split['finish_position']
    group_val = val_split.groupby('race_id').size().tolist()

    logger.info(f"Train split: {len(train_split)} rows (< {val_cutoff})")
    logger.info(f"Val split: {len(val_split)} rows (>= {val_cutoff})")

    oos_model = LGBMRanker()
    oos_model.train(X_train, y_train, group_train, X_val, y_val, group_val)

    # ========================================
    # Step 2: 2025テストデータ構築
    # ========================================
    logger.info("Building test dataset (2025)...")
    test_df = builder.build_dataset(2025, 2025)
    test_df = test_df.dropna(subset=['finish_position'])
    logger.info(f"Test data: {len(test_df)} rows")

    # ========================================
    # Step 3: 現行モデル（2022-2025学習）をロード
    # ========================================
    insample_model = LGBMRanker()
    insample_model.load(str(RESULTS_DIR / 'model_lgbm.pkl'))

    # ========================================
    # Step 4: 両モデルで2025年をバックテスト
    # ========================================
    print("\n" + "#"*90)
    print("#  比較: In-Sample vs Out-of-Sample (2025年テスト)")
    print("#"*90)

    run_backtest(insample_model, insample_model.feature_names, test_df,
                 "現行モデル (2022-2025学習 → 2025テスト = in-sample)")

    run_backtest(oos_model, feature_cols, test_df,
                 "OOSモデル (2022-2024学習 → 2025テスト = true out-of-sample)")

    # ========================================
    # Step 5: OOSモデルの特徴量重要度
    # ========================================
    print(f"\n{'='*90}")
    print(f"  OOSモデル 特徴量重要度 Top15")
    print(f"{'='*90}")
    imp = oos_model.feature_importance('gain')
    for _, row in imp.head(15).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:>10,.0f}")

    # OOSモデルを一時保存（必要に応じて）
    oos_path = str(RESULTS_DIR / 'model_lgbm_oos.pkl')
    oos_model.save(oos_path)
    print(f"\n  OOSモデル保存: {oos_path}")


if __name__ == '__main__':
    main()
