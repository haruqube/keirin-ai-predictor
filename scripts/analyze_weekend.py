# -*- coding: utf-8 -*-
"""週末の予測 vs 実績 分析 + 購入戦略シミュレーション"""

import sys
import io
import re
import sqlite3
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup
from config import CACHE_DIR, DB_PATH


def get_all_paybacks(race_id):
    """Get full payback data from cached HTML"""
    html_files = []
    for f in CACHE_DIR.glob('*.html'):
        if race_id in f.name and 'result' in f.name:
            html_files = [f]
            break
    if not html_files:
        return {}

    html = html_files[0].read_text(encoding='utf-8')
    soup = BeautifulSoup(html, 'lxml')
    pay_table = soup.select_one('.Payout_Detail_Table')
    if not pay_table:
        return {}

    result = {}
    text = pay_table.get_text(' ', strip=True)

    for bet_type in ['２車複', '２車単', '３連複', '３連単']:
        pattern = rf'{re.escape(bet_type)}\s+(\d+[->\s]+\d+(?:[->\s]+\d+)?)\s+([\d,]+)円\s+(\d+)人気'
        m = re.search(pattern, text)
        if m:
            result[bet_type] = {
                'combo': m.group(1),
                'amount': int(m.group(2).replace(',', '')),
                'popularity': int(m.group(3)),
            }

    # ワイド: multiple combos
    wide_matches = re.findall(r'(\d+-\d+)\s+([\d,]+)円\s+(\d+)人気', text)
    # Filter to only wide combos (after 'ワイド' keyword)
    wide_start = text.find('ワイド')
    sanren_start = text.find('３連複')
    if wide_start >= 0:
        wide_text = text[wide_start:sanren_start] if sanren_start > wide_start else text[wide_start:]
        wides = re.findall(r'(\d+-\d+)\s+([\d,]+)円\s+(\d+)人気', wide_text)
        result['ワイド'] = [
            {'combo': w[0], 'amount': int(w[1].replace(',', '')), 'pop': int(w[2])}
            for w in wides
        ]

    return result


def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # All races from this week (3/10 - 3/15)
    week_prefixes = ('20260310', '20260311', '20260312', '20260313', '20260314', '20260315')
    race_ids = [
        r['race_id'] for r in conn.execute(
            'SELECT DISTINCT race_id FROM predictions ORDER BY race_id'
        ).fetchall()
        if any(r['race_id'].startswith(p) for p in week_prefixes)
    ]

    # Gather all data
    race_data = []
    for race_id in sorted(race_ids):
        date_str = f"{race_id[4:6]}/{race_id[6:8]}"
        race_num = int(race_id[10:12])

        preds = conn.execute('''
            SELECT p.rider_id, p.predicted_rank, p.predicted_score, e.bike_number, e.rider_name
            FROM predictions p
            JOIN entries e ON p.race_id = e.race_id AND p.rider_id = e.rider_id
            WHERE p.race_id = ?
            ORDER BY p.predicted_rank
        ''', (race_id,)).fetchall()

        results = conn.execute('''
            SELECT rider_id, finish_position, bike_number, rider_name
            FROM race_results WHERE race_id = ? AND finish_position IS NOT NULL
            ORDER BY finish_position
        ''', (race_id,)).fetchall()

        if not preds or not results:
            continue

        payback = get_all_paybacks(race_id)
        pred_bikes = [p['bike_number'] for p in preds[:5]]
        actual_bikes = [r['bike_number'] for r in results[:3]]
        scores = [p['predicted_score'] for p in preds[:5]]
        score_gap = scores[0] - scores[1] if len(scores) > 1 else 0
        confidence = 'HIGH' if score_gap > 0.5 else ('MED' if score_gap > 0.2 else 'LOW')

        # Get velodrome name from races table
        race_info = conn.execute(
            'SELECT velodrome, grade FROM races WHERE race_id = ?', (race_id,)
        ).fetchone()
        velodrome = race_info['velodrome'] if race_info else '?'
        grade = race_info['grade'] if race_info else '?'

        race_data.append({
            'date': date_str, 'num': race_num, 'race_id': race_id,
            'velodrome': velodrome, 'grade': grade,
            'pred': pred_bikes, 'actual': actual_bikes,
            'scores': scores, 'gap': score_gap, 'conf': confidence,
            'payback': payback,
        })

    # ===== Strategy simulations =====
    strategies = {}

    def add_result(name, bet, payout, desc=''):
        if name not in strategies:
            strategies[name] = {'bet': 0, 'payout': 0, 'hits': 0, 'races': 0, 'desc': desc}
        strategies[name]['bet'] += bet
        strategies[name]['payout'] += payout
        strategies[name]['races'] += 1
        if payout > 0:
            strategies[name]['hits'] += 1

    for rd in race_data:
        pb = rd['payback']
        pred3 = set(rd['pred'][:3])
        pred5 = set(rd['pred'][:5])

        # ---- Strategy A: ワイド ◎○▲ BOX (3点) ----
        if 'ワイド' in pb:
            bet = 300
            payout = 0
            for w in pb['ワイド']:
                combo_bikes = set(int(b) for b in w['combo'].split('-'))
                if combo_bikes.issubset(pred3):
                    payout += w['amount']
            add_result('A. ワイド◎○▲BOX(3点)', bet, payout, '100円x3通り')

        # ---- Strategy B: 2車複 ◎○▲ BOX (3点) ----
        if '２車複' in pb:
            bet = 300
            payout = 0
            actual_2fuku = set(int(b) for b in re.findall(r'\d+', pb['２車複']['combo']))
            if actual_2fuku.issubset(pred3):
                payout = pb['２車複']['amount']
            add_result('B. 2車複◎○▲BOX(3点)', bet, payout, '100円x3通り')

        # ---- Strategy C: 2車複 ◎流し→○▲△△ (4点) ----
        if '２車複' in pb:
            bet = 400
            payout = 0
            actual_2fuku = set(int(b) for b in re.findall(r'\d+', pb['２車複']['combo']))
            if rd['pred'][0] in actual_2fuku and actual_2fuku.issubset(pred5):
                payout = pb['２車複']['amount']
            add_result('C. 2車複◎流し(4点)', bet, payout, '◎→○▲△△ 100円x4')

        # ---- Strategy D: 3連複 ◎○▲ (1点) ----
        if '３連複' in pb:
            bet = 100
            payout = 0
            actual_3fuku = set(int(b) for b in re.findall(r'\d+', pb['３連複']['combo']))
            if actual_3fuku == pred3:
                payout = pb['３連複']['amount']
            add_result('D. 3連複◎○▲(1点)', bet, payout, '100円x1通り')

        # ---- Strategy E: 3連複 Top5BOX (10点) ----
        if '３連複' in pb:
            bet = 1000
            payout = 0
            actual_3fuku = set(int(b) for b in re.findall(r'\d+', pb['３連複']['combo']))
            if actual_3fuku.issubset(pred5):
                payout = pb['３連複']['amount']
            add_result('E. 3連複Top5BOX(10点)', bet, payout, '100円x10通り')

        # ---- Strategy F: 信頼度フィルタ + ワイドBOX ----
        if 'ワイド' in pb:
            if rd['conf'] == 'HIGH':
                multiplier = 3
            elif rd['conf'] == 'MED':
                multiplier = 1
            else:
                multiplier = 0  # skip LOW
            if multiplier > 0:
                bet = 300 * multiplier
                payout = 0
                for w in pb['ワイド']:
                    combo_bikes = set(int(b) for b in w['combo'].split('-'))
                    if combo_bikes.issubset(pred3):
                        payout += w['amount'] * multiplier
                add_result('F. ワイドBOX信頼度調整', bet, payout, 'HIGH=300円 MED=100円 LOW=見送り')

        # ---- Strategy G: 信頼度フィルタ + 2車複BOX ----
        if '２車複' in pb:
            if rd['conf'] == 'HIGH':
                multiplier = 3
            elif rd['conf'] == 'MED':
                multiplier = 1
            else:
                multiplier = 0
            if multiplier > 0:
                bet = 300 * multiplier
                payout = 0
                actual_2fuku = set(int(b) for b in re.findall(r'\d+', pb['２車複']['combo']))
                if actual_2fuku.issubset(pred3):
                    payout = pb['２車複']['amount'] * multiplier
                add_result('G. 2車複BOX信頼度調整', bet, payout, 'HIGH=300円 MED=100円 LOW=見送り')

        # ---- Strategy H: HIGH信頼度のみ 2車複BOX ----
        if '２車複' in pb and rd['conf'] == 'HIGH':
            bet = 500
            payout = 0
            actual_2fuku = set(int(b) for b in re.findall(r'\d+', pb['２車複']['combo']))
            if actual_2fuku.issubset(pred3):
                payout = pb['２車複']['amount'] * 5
            add_result('H. HIGHのみ2車複(500円)', bet, payout, 'HIGH信頼度のみ500円、他見送り')

        # ---- Strategy I: ワイド◎軸流し + 信頼度 ----
        if 'ワイド' in pb and rd['conf'] != 'LOW':
            multiplier = 2 if rd['conf'] == 'HIGH' else 1
            bet = 200 * multiplier  # ◎-○, ◎-▲ の2点
            payout = 0
            for w in pb['ワイド']:
                combo_bikes = set(int(b) for b in w['combo'].split('-'))
                if rd['pred'][0] in combo_bikes and combo_bikes.issubset(pred3):
                    payout += w['amount'] * multiplier
            add_result('I. ワイド◎軸流し+信頼度', bet, payout, '◎-○,◎-▲ HIGH=200円 MED=100円')

    # ===== Print detailed race results by venue =====
    # Group by venue
    from collections import defaultdict
    venue_groups = defaultdict(list)
    for rd in race_data:
        venue_groups[f"{rd['velodrome']}({rd['grade']})"].append(rd)

    for venue_key, venue_races in sorted(venue_groups.items()):
        dates_in_venue = sorted(set(rd['date'] for rd in venue_races))
        print(f"\n{'=' * 100}")
        print(f"  {venue_key} ({', '.join(dates_in_venue)}) - {len(venue_races)}レース")
        print(f"{'=' * 100}")
        print(f"{'日付':>5} {'R':>3} {'◎':>1} {'信頼':>4} | {'予測':>7} | {'実際':>7} | {'一致':>4} | {'2車複':>17} | {'3連複':>17}")
        print('-' * 100)

        venue_top1 = 0
        venue_top3 = 0
        for rd in venue_races:
            pred3 = rd['pred'][:3]
            actual3 = rd['actual']
            overlap = len(set(pred3) & set(actual3))
            hit = 'o' if pred3[0] == actual3[0] else 'x'
            if pred3[0] == actual3[0]:
                venue_top1 += 1
            venue_top3 += overlap

            pb = rd['payback']
            nisha = pb.get('２車複', {})
            sanren = pb.get('３連複', {})

            nisha_str = f"{nisha['amount']:>6,}円({nisha['popularity']:>2}人気)" if nisha else '           -'
            sanren_str = f"{sanren['amount']:>6,}円({sanren['popularity']:>2}人気)" if sanren else '           -'

            print(f"{rd['date']:>5} {rd['num']:>3}R [{hit}] {rd['conf']:>4} | "
                  f"{pred3[0]}-{pred3[1]}-{pred3[2]:>1} | "
                  f"{actual3[0]}-{actual3[1]}-{actual3[2]:>1} | "
                  f"{overlap}/3  | {nisha_str} | {sanren_str}")

        n = len(venue_races)
        print(f"  >>> {venue_key} 小計: ◎的中={venue_top1}/{n} ({venue_top1/n*100:.1f}%) "
              f"Top3一致={venue_top3}/{n*3} ({venue_top3/(n*3)*100:.1f}%)")

    # ===== Print strategy comparison =====
    # Venue summary
    print(f"\n{'=' * 100}")
    print(f"  会場別サマリ")
    print(f"{'=' * 100}")
    for venue_key, venue_races in sorted(venue_groups.items()):
        n = len(venue_races)
        t1 = sum(1 for rd in venue_races if rd['pred'][0] == rd['actual'][0])
        t3 = sum(len(set(rd['pred'][:3]) & set(rd['actual'])) for rd in venue_races)
        print(f"  {venue_key:<15} {n:>3}R | ◎={t1:>2}/{n} ({t1/n*100:>5.1f}%) | Top3={t3:>3}/{n*3} ({t3/(n*3)*100:>5.1f}%)")

    print(f"\n{'=' * 100}")
    print(f"  購入戦略比較シミュレーション ({len(race_data)}レース)")
    print(f"{'=' * 100}")
    print(f"{'戦略':<32} {'投資':>8} {'回収':>8} {'収支':>8} {'回収率':>7} {'的中':>6}")
    print('-' * 100)

    for name in sorted(strategies.keys()):
        s = strategies[name]
        if s['bet'] > 0:
            roi = s['payout'] / s['bet'] * 100
            profit = s['payout'] - s['bet']
            print(f"{name:<30} {s['bet']:>7,}円 {s['payout']:>7,}円 {profit:>+7,}円 {roi:>6.1f}% {s['hits']:>2}/{s['races']}")
            print(f"  ({s['desc']})")

    # ===== Optimal strategy recommendation =====
    print(f"\n{'=' * 90}")
    print("  最適戦略の推奨")
    print(f"{'=' * 90}")

    best = max(strategies.items(), key=lambda x: x[1]['payout'] / x[1]['bet'] if x[1]['bet'] > 0 else 0)
    print(f"\n回収率最高: {best[0]}")
    s = best[1]
    roi = s['payout'] / s['bet'] * 100 if s['bet'] > 0 else 0
    print(f"  回収率: {roi:.1f}% | 的中: {s['hits']}/{s['races']} | 収支: {s['payout'] - s['bet']:+,}円")

    # Analysis
    print(f"\n--- 予測精度サマリ ---")
    top1_hits = sum(1 for rd in race_data if rd['pred'][0] == rd['actual'][0])
    top3_overlap = sum(len(set(rd['pred'][:3]) & set(rd['actual'])) for rd in race_data)
    top5_cover = sum(1 for rd in race_data if set(rd['actual']).issubset(set(rd['pred'][:5])))
    print(f"◎的中率: {top1_hits}/{len(race_data)} ({top1_hits/len(race_data)*100:.1f}%)")
    print(f"Top3一致: {top3_overlap}/{len(race_data)*3} ({top3_overlap/(len(race_data)*3)*100:.1f}%)")
    print(f"Top5内に3着以内全員: {top5_cover}/{len(race_data)} ({top5_cover/len(race_data)*100:.1f}%)")

    print(f"\n--- 信頼度別 ◎的中率 ---")
    for conf in ['HIGH', 'MED', 'LOW']:
        conf_races = [rd for rd in race_data if rd['conf'] == conf]
        if conf_races:
            hits = sum(1 for rd in conf_races if rd['pred'][0] == rd['actual'][0])
            print(f"  {conf}: {hits}/{len(conf_races)} ({hits/len(conf_races)*100:.1f}%) "
                  f"- スコア差平均: {sum(rd['gap'] for rd in conf_races)/len(conf_races):.3f}")

    conn.close()


if __name__ == '__main__':
    main()
