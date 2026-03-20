# -*- coding: utf-8 -*-
"""全賭式ROI比較バックテスト

モデルの予測Top3/5に基づき、各賭式の実際の配当データを使ってROIを比較する。
賭式: 3連単 / 3連複 / 2連単 / 2連複 / ワイド
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


# ── 配当パーサ ──

def parse_payouts_from_html(race_id: str) -> dict:
    """HTML結果キャッシュから全賭式の配当を取得"""
    html_path = CACHE_DIR / f"keirin_netkeiba_com_db_result__race_id_{race_id}.html"
    if not html_path.exists():
        return {}

    try:
        html = html_path.read_text(encoding='utf-8')
    except Exception:
        return {}

    soup = BeautifulSoup(html, 'lxml')
    pay_table = soup.select_one('.Payout_Detail_Table')
    if not pay_table:
        return {}

    text = pay_table.get_text(' ', strip=True)
    result = {}

    # 2車複, 2車単, 3連複, 3連単
    for bet_type in ['２車複', '２車単', '３連複', '３連単']:
        pattern = rf'{re.escape(bet_type)}\s+(\d+[->\s]+\d+(?:[->\s]+\d+)?)\s+([\d,]+)円\s+(\d+)人気'
        m = re.search(pattern, text)
        if m:
            combo_str = m.group(1)
            # bike numbers を抽出
            bikes = [int(b) for b in re.findall(r'\d+', combo_str)]
            is_ordered = '>' in combo_str
            result[bet_type] = {
                'bikes': bikes,
                'ordered': is_ordered,
                'amount': int(m.group(2).replace(',', '')),
                'popularity': int(m.group(3)),
            }

    # ワイド: 複数組み合わせ
    wide_start = text.find('ワイド')
    sanren_start = text.find('３連複')
    if wide_start >= 0:
        wide_text = text[wide_start:sanren_start] if sanren_start > wide_start else text[wide_start:]
        wides = re.findall(r'(\d+-\d+)\s+([\d,]+)円\s+(\d+)人気', wide_text)
        result['ワイド'] = [
            {
                'bikes': sorted([int(b) for b in w[0].split('-')]),
                'amount': int(w[1].replace(',', '')),
                'popularity': int(w[2]),
            }
            for w in wides
        ]

    return result


# ── 戦略定義 ──

def simulate_strategies(pred_bikes: list[int], payouts: dict) -> dict[str, dict]:
    """
    予測車番リスト(信頼度順)と配当データから各戦略のbet/payoutを計算。
    pred_bikes: [◎, ○, ▲, △, △, ...] の車番リスト
    """
    results = {}
    pred3 = pred_bikes[:3]
    pred5 = pred_bikes[:5]
    pred3_set = set(pred3)
    pred5_set = set(pred5)

    # ======== 3連単 ========
    if '３連単' in payouts:
        p = payouts['３連単']
        actual = p['bikes']  # [1着, 2着, 3着] 順序あり

        # 3連単 ◎○▲ 1点 (100円) — 完全一致
        hit = (pred3 == actual)
        results['3連単_◎○▲_1点'] = {
            'bet': 100, 'payout': p['amount'] if hit else 0, 'type': '3連単'}

        # 3連単 BOX 6点 (600円) — Top3が一致（順不同）
        hit = (set(actual) == pred3_set)
        results['3連単_BOX3_6点'] = {
            'bet': 600, 'payout': p['amount'] if hit else 0, 'type': '3連単'}

        # 3連単 Top4BOX 24点 (2,400円)
        hit = set(actual).issubset(set(pred_bikes[:4]))
        results['3連単_BOX4_24点'] = {
            'bet': 2400, 'payout': p['amount'] if hit else 0, 'type': '3連単'}

        # 3連単 Top5BOX 60点 (6,000円)
        hit = set(actual).issubset(pred5_set)
        results['3連単_BOX5_60点'] = {
            'bet': 6000, 'payout': p['amount'] if hit else 0, 'type': '3連単'}

        # 3連単 ◎1着固定 ○▲流し 2点 (200円)
        hit = (pred3[0] == actual[0] and set(actual[1:]) == set(pred3[1:]))
        results['3連単_◎1着固定_2点'] = {
            'bet': 200, 'payout': p['amount'] if hit else 0, 'type': '3連単'}

    # ======== 3連複 ========
    if '３連複' in payouts:
        p = payouts['３連複']
        actual_set = set(p['bikes'])

        # 3連複 ◎○▲ 1点 (100円)
        hit = (actual_set == pred3_set)
        results['3連複_◎○▲_1点'] = {
            'bet': 100, 'payout': p['amount'] if hit else 0, 'type': '3連複'}

        # 3連複 Top4BOX 4点 (400円)
        hit = actual_set.issubset(set(pred_bikes[:4]))
        results['3連複_BOX4_4点'] = {
            'bet': 400, 'payout': p['amount'] if hit else 0, 'type': '3連複'}

        # 3連複 Top5BOX 10点 (1,000円)
        hit = actual_set.issubset(pred5_set)
        results['3連複_BOX5_10点'] = {
            'bet': 1000, 'payout': p['amount'] if hit else 0, 'type': '3連複'}

        # 3連複 ◎軸2頭流し(○▲△△) 6点 (600円)
        hit = (pred3[0] in actual_set and actual_set.issubset(pred5_set))
        results['3連複_◎軸流し_6点'] = {
            'bet': 600, 'payout': p['amount'] if hit else 0, 'type': '3連複'}

    # ======== 2連単 ========
    if '２車単' in payouts:
        p = payouts['２車単']
        actual = p['bikes']  # [1着, 2着] 順序あり

        # 2連単 ◎○ 1点 (100円) — 完全一致
        hit = (pred3[0] == actual[0] and pred3[1] == actual[1])
        results['2連単_◎○_1点'] = {
            'bet': 100, 'payout': p['amount'] if hit else 0, 'type': '2連単'}

        # 2連単 ◎○▲BOX P(3,2)=6点 (600円)
        hit = (actual[0] in pred3_set and actual[1] in pred3_set)
        results['2連単_BOX3_6点'] = {
            'bet': 600, 'payout': p['amount'] if hit else 0, 'type': '2連単'}

        # 2連単 ◎1着固定→○▲△△ 4点 (400円)
        hit = (pred3[0] == actual[0] and actual[1] in pred5_set)
        results['2連単_◎1着固定_4点'] = {
            'bet': 400, 'payout': p['amount'] if hit else 0, 'type': '2連単'}

        # 2連単 ◎○ 裏表 2点 (200円)
        hit = (set(actual) == set(pred3[:2]))
        results['2連単_◎○裏表_2点'] = {
            'bet': 200, 'payout': p['amount'] if hit else 0, 'type': '2連単'}

    # ======== 2連複 ========
    if '２車複' in payouts:
        p = payouts['２車複']
        actual_set = set(p['bikes'])

        # 2連複 ◎○ 1点 (100円)
        hit = (actual_set == set(pred3[:2]))
        results['2連複_◎○_1点'] = {
            'bet': 100, 'payout': p['amount'] if hit else 0, 'type': '2連複'}

        # 2連複 ◎○▲ BOX C(3,2)=3点 (300円)
        hit = actual_set.issubset(pred3_set)
        results['2連複_BOX3_3点'] = {
            'bet': 300, 'payout': p['amount'] if hit else 0, 'type': '2連複'}

        # 2連複 ◎軸流し→○▲△△ 4点 (400円)
        hit = (pred3[0] in actual_set and actual_set.issubset(pred5_set))
        results['2連複_◎軸流し_4点'] = {
            'bet': 400, 'payout': p['amount'] if hit else 0, 'type': '2連複'}

    # ======== ワイド ========
    if 'ワイド' in payouts:
        wide_payouts = payouts['ワイド']

        # ワイド ◎○ 1点 (100円)
        payout = 0
        target = sorted(pred3[:2])
        for w in wide_payouts:
            if w['bikes'] == target:
                payout = w['amount']
        results['ワイド_◎○_1点'] = {
            'bet': 100, 'payout': payout, 'type': 'ワイド'}

        # ワイド ◎○▲ BOX 3点 (300円)
        payout = 0
        from itertools import combinations
        target_combos = [sorted(c) for c in combinations(pred3, 2)]
        for w in wide_payouts:
            if w['bikes'] in target_combos:
                payout += w['amount']
        results['ワイド_BOX3_3点'] = {
            'bet': 300, 'payout': payout, 'type': 'ワイド'}

        # ワイド ◎軸流し→○▲ 2点 (200円)
        payout = 0
        for w in wide_payouts:
            if pred3[0] in w['bikes'] and set(w['bikes']).issubset(pred3_set):
                payout += w['amount']
        results['ワイド_◎軸_2点'] = {
            'bet': 200, 'payout': payout, 'type': 'ワイド'}

        # ワイド ◎軸流し→○▲△△ 4点 (400円)
        payout = 0
        for w in wide_payouts:
            if pred3[0] in w['bikes'] and set(w['bikes']).issubset(pred5_set):
                payout += w['amount']
        results['ワイド_◎軸広め_4点'] = {
            'bet': 400, 'payout': payout, 'type': 'ワイド'}

    return results


def main():
    parser = argparse.ArgumentParser(description='全賭式ROI比較バックテスト')
    parser.add_argument('--year', type=int, default=2025,
                        help='バックテスト対象年 (default: 2025)')
    parser.add_argument('--grade', type=str, default=None,
                        help='グレードフィルタ (e.g., F1, F2, G3)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # ── 1. モデル読み込み ──
    model_path = str(RESULTS_DIR / 'model_lgbm.pkl')
    model = LGBMRanker()
    model.load(model_path)
    feature_cols = model.feature_names
    logger.info("Model loaded: %d features", len(feature_cols))

    # ── 2. データセット構築 ──
    builder = FeatureBuilder()
    all_years = sorted(set(TRAIN_YEARS + [args.year]))
    logger.info("Building dataset (%d-%d)...", all_years[0], all_years[-1])
    all_df = builder.build_dataset(all_years[0], all_years[-1])
    all_df = all_df.dropna(subset=['finish_position'])

    # 対象年にフィルタ
    target_df = all_df[all_df['race_date'].str.startswith(str(args.year))].copy()
    logger.info("Target races in %d: %d entries", args.year, len(target_df))

    if target_df.empty:
        logger.error("No data for year %d", args.year)
        return

    # グレードフィルタ
    if args.grade:
        from db.schema import get_connection
        conn = get_connection()
        grade_races = set(r['race_id'] for r in conn.execute(
            "SELECT race_id FROM races WHERE grade = ?", (args.grade,)
        ).fetchall())
        conn.close()
        target_df = target_df[target_df['race_id'].isin(grade_races)]
        logger.info("Filtered to grade %s: %d entries", args.grade, len(target_df))

    # ── 3. 予測 ──
    X = target_df[feature_cols].reindex(columns=feature_cols, fill_value=0).fillna(0)
    target_df['pred_score'] = model.predict(X).values
    logger.info("Predictions done")

    # ── 4. レースごとにシミュレーション ──
    strategy_totals = defaultdict(lambda: {
        'bet': 0, 'payout': 0, 'hits': 0, 'races': 0, 'type': '',
        'payouts_when_hit': [],
    })

    race_ids = target_df['race_id'].unique()
    total_races = len(race_ids)
    parsed = 0
    skipped = 0

    for i, race_id in enumerate(race_ids):
        if (i + 1) % 500 == 0:
            logger.info("  Progress: %d/%d races (%.0f%%)", i + 1, total_races,
                        (i + 1) / total_races * 100)

        group = target_df[target_df['race_id'] == race_id]
        if len(group) < 3:
            continue

        # 予測順にソート → 車番リスト
        ranked = group.sort_values('pred_score', ascending=False)
        pred_bikes = ranked['bike_number'].astype(int).tolist()

        if len(pred_bikes) < 3:
            continue

        # 配当パース
        payouts = parse_payouts_from_html(race_id)
        if not payouts:
            skipped += 1
            continue
        parsed += 1

        # 各戦略のシミュレーション
        strats = simulate_strategies(pred_bikes, payouts)
        for name, res in strats.items():
            s = strategy_totals[name]
            s['bet'] += res['bet']
            s['payout'] += res['payout']
            s['races'] += 1
            s['type'] = res['type']
            if res['payout'] > 0:
                s['hits'] += 1
                s['payouts_when_hit'].append(res['payout'])

    logger.info("Simulation done: %d races parsed, %d skipped (no HTML)", parsed, skipped)

    # ── 5. 結果表示 ──
    print()
    print('=' * 110)
    print(f'  全賭式ROI比較バックテスト — {args.year}年 ({parsed}レース)')
    if args.grade:
        print(f'  グレード: {args.grade}')
    print('=' * 110)

    # 賭式別にグループ化して表示
    bet_types = ['3連単', '3連複', '2連単', '2連複', 'ワイド']
    for bt in bet_types:
        strats = {k: v for k, v in sorted(strategy_totals.items()) if v['type'] == bt}
        if not strats:
            continue

        print(f'\n  ── {bt} ──')
        print(f'  {"戦略":<28} {"投資":>10} {"回収":>10} {"収支":>10} '
              f'{"回収率":>8} {"的中":>8} {"的中率":>7} {"平均配当":>8}')
        print('  ' + '-' * 104)

        for name, s in sorted(strats.items(),
                               key=lambda x: x[1]['payout'] / x[1]['bet']
                               if x[1]['bet'] > 0 else 0, reverse=True):
            if s['bet'] == 0:
                continue
            roi = s['payout'] / s['bet'] * 100
            hit_rate = s['hits'] / s['races'] * 100 if s['races'] > 0 else 0
            avg_pay = (sum(s['payouts_when_hit']) / len(s['payouts_when_hit'])
                       if s['payouts_when_hit'] else 0)
            profit = s['payout'] - s['bet']
            print(f'  {name:<28} {s["bet"]:>9,}円 {s["payout"]:>9,}円 {profit:>+9,}円 '
                  f'{roi:>7.1f}% {s["hits"]:>3}/{s["races"]:<4} {hit_rate:>5.1f}% '
                  f'{avg_pay:>7,.0f}円')

    # ── 全戦略ランキング ──
    print(f'\n{"=" * 110}')
    print(f'  ROI上位ランキング（全戦略）')
    print(f'{"=" * 110}')
    print(f'  {"順":>2} {"戦略":<28} {"賭式":<6} {"回収率":>8} {"的中率":>7} '
          f'{"投資":>10} {"収支":>10} {"平均配当":>8}')
    print('  ' + '-' * 104)

    ranked_strats = sorted(
        strategy_totals.items(),
        key=lambda x: x[1]['payout'] / x[1]['bet'] if x[1]['bet'] > 0 else 0,
        reverse=True
    )

    for rank, (name, s) in enumerate(ranked_strats, 1):
        if s['bet'] == 0:
            continue
        roi = s['payout'] / s['bet'] * 100
        hit_rate = s['hits'] / s['races'] * 100 if s['races'] > 0 else 0
        avg_pay = (sum(s['payouts_when_hit']) / len(s['payouts_when_hit'])
                   if s['payouts_when_hit'] else 0)
        profit = s['payout'] - s['bet']
        marker = ' ★' if roi >= 100 else ''
        print(f'  {rank:>2}. {name:<28} {s["type"]:<6} {roi:>7.1f}% {hit_rate:>5.1f}% '
              f'{s["bet"]:>9,}円 {profit:>+9,}円 {avg_pay:>7,.0f}円{marker}')

    # ── 賭式別サマリ ──
    print(f'\n{"=" * 110}')
    print(f'  賭式別ベスト戦略サマリ')
    print(f'{"=" * 110}')

    for bt in bet_types:
        strats = {k: v for k, v in strategy_totals.items() if v['type'] == bt}
        if not strats:
            continue
        best_name, best = max(strats.items(),
                               key=lambda x: x[1]['payout'] / x[1]['bet']
                               if x[1]['bet'] > 0 else 0)
        roi = best['payout'] / best['bet'] * 100 if best['bet'] > 0 else 0
        hit_rate = best['hits'] / best['races'] * 100 if best['races'] > 0 else 0
        profit = best['payout'] - best['bet']
        print(f'  {bt:<6} → {best_name:<28} ROI={roi:.1f}% 的中={hit_rate:.1f}% '
              f'収支={profit:+,}円')

    # ── 推奨戦略 ──
    print(f'\n{"=" * 110}')
    print(f'  総合推奨')
    print(f'{"=" * 110}')
    best_overall_name, best_overall = max(
        strategy_totals.items(),
        key=lambda x: x[1]['payout'] / x[1]['bet'] if x[1]['bet'] > 0 else 0
    )
    roi = best_overall['payout'] / best_overall['bet'] * 100
    profit = best_overall['payout'] - best_overall['bet']
    print(f'  最高ROI: {best_overall_name} ({best_overall["type"]})')
    print(f'  回収率: {roi:.1f}% | 収支: {profit:+,}円 | '
          f'的中: {best_overall["hits"]}/{best_overall["races"]}')

    # 安定性（的中率×配当バランス）
    print(f'\n  --- 安定性指標（的中率×回収率のバランス） ---')
    for name, s in sorted(strategy_totals.items()):
        if s['bet'] == 0 or s['races'] == 0:
            continue
        roi = s['payout'] / s['bet'] * 100
        hit_rate = s['hits'] / s['races'] * 100
        # シャープレシオ的な指標: (ROI-75) / volatility
        if s['payouts_when_hit']:
            std = np.std(s['payouts_when_hit']) if len(s['payouts_when_hit']) > 1 else 0
        else:
            std = 0
    # Just show ROI >= 70% strategies
    print(f'\n  ROI 70%以上の戦略:')
    for name, s in sorted(ranked_strats,
                           key=lambda x: x[1]['payout'] / x[1]['bet']
                           if x[1]['bet'] > 0 else 0, reverse=True):
        if s['bet'] == 0:
            continue
        roi = s['payout'] / s['bet'] * 100
        if roi < 70:
            break
        hit_rate = s['hits'] / s['races'] * 100 if s['races'] > 0 else 0
        avg_pay = (sum(s['payouts_when_hit']) / len(s['payouts_when_hit'])
                   if s['payouts_when_hit'] else 0)
        per_race = (s['payout'] - s['bet']) / s['races'] if s['races'] > 0 else 0
        print(f'    {name:<28} ROI={roi:>6.1f}% 的中率={hit_rate:>5.1f}% '
              f'1R平均損益={per_race:>+6.0f}円 平均配当={avg_pay:>7,.0f}円')


if __name__ == '__main__':
    main()
