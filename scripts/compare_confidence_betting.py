# -*- coding: utf-8 -*-
"""信頼度別ベッティング戦略比較

2連単◎1着固定4点をベースに、信頼度による賭け方を比較:
- 全レース均等買い vs 信頼度フィルタ vs 集中投資
"""

import sys
import io
import re
import logging
import argparse
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from config import TRAIN_YEARS, TEST_YEARS, RESULTS_DIR, CACHE_DIR
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker

logger = logging.getLogger(__name__)


def parse_nisyatan(race_id: str) -> dict | None:
    """HTML結果キャッシュから2車単の配当を取得"""
    html_path = CACHE_DIR / f"keirin_netkeiba_com_db_result__race_id_{race_id}.html"
    if not html_path.exists():
        return None
    try:
        html = html_path.read_text(encoding='utf-8')
    except Exception:
        return None
    soup = BeautifulSoup(html, 'lxml')
    pay = soup.select_one('.Payout_Detail_Table')
    if not pay:
        return None
    text = pay.get_text(' ', strip=True)
    m = re.search(r'２車単\s+(\d+[->\s]+\d+)\s+([\d,]+)円\s+(\d+)人気', text)
    if not m:
        return None
    bikes = [int(b) for b in re.findall(r'\d+', m.group(1))]
    return {'bikes': bikes, 'amount': int(m.group(2).replace(',', '')), 'pop': int(m.group(3))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2025)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # モデル・データ準備
    model = LGBMRanker()
    model.load(str(RESULTS_DIR / 'model_lgbm.pkl'))
    feature_cols = model.feature_names
    logger.info("Model loaded")

    builder = FeatureBuilder()
    all_years = sorted(set(TRAIN_YEARS + [args.year]))
    logger.info("Building dataset...")
    all_df = builder.build_dataset(all_years[0], all_years[-1])
    all_df = all_df.dropna(subset=['finish_position'])
    target_df = all_df[all_df['race_date'].str.startswith(str(args.year))].copy()
    logger.info("Target: %d entries", len(target_df))

    X = target_df[feature_cols].reindex(columns=feature_cols, fill_value=0).fillna(0)
    target_df['pred_score'] = model.predict(X).values

    # レースごとにデータ構築
    race_data = []
    race_ids = target_df['race_id'].unique()

    for i, race_id in enumerate(race_ids):
        if (i + 1) % 500 == 0:
            logger.info("  Parsing: %d/%d", i + 1, len(race_ids))

        group = target_df[target_df['race_id'] == race_id]
        if len(group) < 5:
            continue

        ranked = group.sort_values('pred_score', ascending=False)
        pred_bikes = ranked['bike_number'].astype(int).tolist()
        scores = ranked['pred_score'].tolist()

        payout = parse_nisyatan(race_id)
        if not payout:
            continue

        actual_1st, actual_2nd = payout['bikes'][0], payout['bikes'][1]

        # 2連単◎1着固定→○▲△△ 4点の的中判定
        hit_4pt = (pred_bikes[0] == actual_1st and actual_2nd in set(pred_bikes[1:5]))
        # 2連単◎→○,◎→▲ 2点の的中判定
        hit_2pt = (pred_bikes[0] == actual_1st and actual_2nd in set(pred_bikes[1:3]))

        race_data.append({
            'race_id': race_id,
            'pred_bikes': pred_bikes,
            'score_gap': scores[0] - scores[1],
            'hit_4pt': hit_4pt,
            'hit_2pt': hit_2pt,
            'amount': payout['amount'],
            'pop': payout['pop'],
        })

    logger.info("Race data ready: %d races", len(race_data))

    # 信頼度分類（3分位）
    gaps = sorted([r['score_gap'] for r in race_data])
    t_low = gaps[len(gaps) // 3]
    t_high = gaps[2 * len(gaps) // 3]
    for r in race_data:
        g = r['score_gap']
        r['conf'] = 'HIGH' if g >= t_high else ('MED' if g >= t_low else 'LOW')

    # ── 戦略シミュレーション ──
    strategies = {}

    def sim(name, desc, bet_func):
        total_bet = total_pay = hits = count = 0
        monthly = defaultdict(lambda: {'bet': 0, 'pay': 0})
        hit_amounts = []

        for r in race_data:
            bet, pay = bet_func(r)
            if bet == 0:
                continue
            count += 1
            total_bet += bet
            total_pay += pay
            m = r['race_id'][:6]
            monthly[m]['bet'] += bet
            monthly[m]['pay'] += pay
            if pay > 0:
                hits += 1
                hit_amounts.append(pay)

        mp = [(v['pay'] - v['bet']) for v in monthly.values()]
        strategies[name] = {
            'desc': desc, 'bet': total_bet, 'pay': total_pay,
            'hits': hits, 'races': count, 'hit_amounts': hit_amounts,
            'monthly': mp, 'losing_m': sum(1 for p in mp if p < 0), 'total_m': len(mp),
        }

    # A: 全レース均等 100円×4点
    sim('A. 全レース均等(400円)',
        '全レース100円×4点=400円',
        lambda r: (400, r['amount'] if r['hit_4pt'] else 0))

    # B: LOW見送り 100円×4点
    sim('B. LOW見送り(400円)',
        'MED+HIGHのみ100円×4点',
        lambda r: (0, 0) if r['conf'] == 'LOW' else (400, r['amount'] if r['hit_4pt'] else 0))

    # C: HIGH=300円, MED=100円, LOW=見送り
    sim('C. 段階調整(H300/M100/L見送)',
        'HIGH=300円×4, MED=100円×4, LOW=見送り',
        lambda r: (0, 0) if r['conf'] == 'LOW' else
        (1200, r['amount'] * 3 if r['hit_4pt'] else 0) if r['conf'] == 'HIGH' else
        (400, r['amount'] if r['hit_4pt'] else 0))

    # D: HIGHのみ 100円×4点
    sim('D. HIGHのみ(400円)',
        'HIGH信頼度のみ100円×4点',
        lambda r: (400, r['amount'] if r['hit_4pt'] else 0) if r['conf'] == 'HIGH' else (0, 0))

    # E: HIGHのみ 500円×4点
    sim('E. HIGHのみ集中(2000円)',
        'HIGH信頼度のみ500円×4点=2000円',
        lambda r: (2000, r['amount'] * 5 if r['hit_4pt'] else 0) if r['conf'] == 'HIGH' else (0, 0))

    # F: HIGH=500円, MED=100円, LOW=見送り
    sim('F. 大段階(H500/M100/L見送)',
        'HIGH=500円×4=2000円, MED=100円×4=400円, LOW=見送り',
        lambda r: (0, 0) if r['conf'] == 'LOW' else
        (2000, r['amount'] * 5 if r['hit_4pt'] else 0) if r['conf'] == 'HIGH' else
        (400, r['amount'] if r['hit_4pt'] else 0))

    # G: 全レース2点（◎→○, ◎→▲のみ）
    sim('G. 全レース2点(200円)',
        '全レース◎→○,◎→▲の2点のみ=200円',
        lambda r: (200, r['amount'] if r['hit_2pt'] else 0))

    # H: LOW見送り + 2点
    sim('H. LOW見送り2点(200円)',
        'MED+HIGHのみ◎→○,◎→▲=200円',
        lambda r: (0, 0) if r['conf'] == 'LOW' else (200, r['amount'] if r['hit_2pt'] else 0))

    # ── 表示 ──
    print()
    print('=' * 120)
    print(f'  2連単◎1着固定 信頼度別戦略比較 ({args.year}年 {len(race_data)}レース)')
    print('=' * 120)

    # 信頼度別基本統計
    print(f'\n  ── 信頼度別の基本統計（2連単◎1着固定4点） ──')
    print(f'  {"信頼度":>6} {"レース数":>8} {"的中":>6} {"的中率":>6} {"平均配当":>8} {"中央値":>7} {"最大配当":>8}')
    print('  ' + '-' * 60)
    for conf in ['HIGH', 'MED', 'LOW', '全体']:
        cr = race_data if conf == '全体' else [r for r in race_data if r['conf'] == conf]
        h = [r for r in cr if r['hit_4pt']]
        amts = [r['amount'] for r in h]
        avg_a = np.mean(amts) if amts else 0
        med_a = np.median(amts) if amts else 0
        max_a = max(amts) if amts else 0
        hr = len(h) / len(cr) * 100 if cr else 0
        print(f'  {conf:>6} {len(cr):>7} {len(h):>5} {hr:>5.1f}% {avg_a:>7,.0f}円 {med_a:>6,.0f}円 {max_a:>7,}円')

    # 戦略比較
    print(f'\n  ── 戦略比較 ──')
    print(f'  {"戦略":<36} {"投資":>10} {"回収":>10} {"収支":>10} '
          f'{"回収率":>7} {"的中":>8} {"的中率":>6} {"1R損益":>7} {"赤字月":>6}')
    print('  ' + '-' * 115)

    ranked = sorted(strategies.items(),
                     key=lambda x: x[1]['pay'] / x[1]['bet'] if x[1]['bet'] > 0 else 0,
                     reverse=True)

    for name, s in ranked:
        if s['bet'] == 0:
            continue
        roi = s['pay'] / s['bet'] * 100
        hr = s['hits'] / s['races'] * 100 if s['races'] > 0 else 0
        profit = s['pay'] - s['bet']
        per_r = profit / s['races'] if s['races'] > 0 else 0
        marker = ' ★' if roi >= 100 else ''
        print(f'  {name:<36} {s["bet"]:>9,}円 {s["pay"]:>9,}円 {profit:>+9,}円 '
              f'{roi:>6.1f}% {s["hits"]:>3}/{s["races"]:<4} {hr:>5.1f}% '
              f'{per_r:>+6.0f}円 {s["losing_m"]}/{s["total_m"]}{marker}')

    # 安定性分析
    print(f'\n  ── 安定性分析（月別収支） ──')
    print(f'  {"戦略":<36} {"月平均":>9} {"月最大損":>9} {"月最大益":>9} {"標準偏差":>9} {"シャープ":>7}')
    print('  ' + '-' * 90)

    for name, s in ranked:
        if not s['monthly']:
            continue
        mp = s['monthly']
        avg_m = np.mean(mp)
        std_m = np.std(mp) if len(mp) > 1 else 0
        sharpe = avg_m / std_m if std_m > 0 else 0
        print(f'  {name:<36} {avg_m:>+8,.0f}円 {min(mp):>+8,.0f}円 {max(mp):>+8,.0f}円 '
              f'{std_m:>8,.0f}円 {sharpe:>6.2f}')

    # 結論
    print(f'\n{"=" * 120}')
    print(f'  結論')
    print(f'{"=" * 120}')

    a = strategies['A. 全レース均等(400円)']
    a_roi = a['pay'] / a['bet'] * 100 if a['bet'] > 0 else 0
    best_name, best = ranked[0]
    b_roi = best['pay'] / best['bet'] * 100 if best['bet'] > 0 else 0

    # 年間利益最大の戦略
    profit_best_name, profit_best = max(strategies.items(),
                                         key=lambda x: x[1]['pay'] - x[1]['bet'])
    p_profit = profit_best['pay'] - profit_best['bet']
    p_roi = profit_best['pay'] / profit_best['bet'] * 100 if profit_best['bet'] > 0 else 0

    print(f'\n  ROI最高:     {best_name}')
    print(f'               ROI={b_roi:.1f}% 収支={best["pay"]-best["bet"]:+,}円 赤字月={best["losing_m"]}/{best["total_m"]}')
    print(f'\n  利益最大:    {profit_best_name}')
    print(f'               ROI={p_roi:.1f}% 収支={p_profit:+,}円 赤字月={profit_best["losing_m"]}/{profit_best["total_m"]}')
    print(f'\n  全レース均等: A. 全レース均等(400円)')
    print(f'               ROI={a_roi:.1f}% 収支={a["pay"]-a["bet"]:+,}円 赤字月={a["losing_m"]}/{a["total_m"]}')


if __name__ == '__main__':
    main()
