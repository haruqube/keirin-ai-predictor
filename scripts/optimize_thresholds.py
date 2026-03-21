# -*- coding: utf-8 -*-
"""本命スコア × 信頼度の閾値・点数を最適化"""

import sys, io, re, logging, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from config import RESULTS_DIR, CACHE_DIR
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker

logging.basicConfig(level=logging.WARNING)

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
    m = re.search(r'２車単\s+(\d+[->\s]+\d+)\s+([\d,]+)円\s+(\d+)人気', text)
    if not m: return None
    bikes = [int(b) for b in re.findall(r'\d+', m.group(1))]
    return {'bikes': bikes, 'amount': int(m.group(2).replace(',', ''))}

def main():
    model = LGBMRanker()
    model.load(str(RESULTS_DIR / 'model_lgbm.pkl'))
    feature_cols = model.feature_names

    builder = FeatureBuilder()
    print("データ構築中...")
    all_df = builder.build_dataset(2024, 2025)
    all_df = all_df.dropna(subset=['finish_position'])

    X = all_df[feature_cols].reindex(columns=feature_cols, fill_value=0).fillna(0)
    all_df['pred_score'] = model.predict(X).values

    races = []
    for race_id, group in all_df.groupby('race_id'):
        if len(group) < 5: continue
        ranked = group.sort_values('pred_score', ascending=False)
        bikes = ranked['bike_number'].astype(int).tolist()
        scores = ranked['pred_score'].tolist()
        year = race_id[:4]

        payout = parse_nisyatan(race_id)
        if not payout: continue

        a1, a2 = payout['bikes'][0], payout['bikes'][1]
        hit_at = None
        if bikes[0] == a1:
            for i in range(1, min(5, len(bikes))):
                if bikes[i] == a2:
                    hit_at = i
                    break

        races.append({
            'race_id': race_id, 'year': year,
            'honmei': scores[0] - scores[1],
            'shinrai': scores[0] - scores[2],
            'hit_at': hit_at,
            'payout': payout['amount'],
        })

    print(f"解析対象: {len(races)}レース (2024: {sum(1 for r in races if r['year']=='2024')}, 2025: {sum(1 for r in races if r['year']=='2025')})")

    # ========================================
    # Phase 1: 本命スコア閾値の最適化
    # ========================================
    print(f"\n{'='*90}")
    print(f"  Phase 1: 本命スコア閾値（参加/見送り境界）— 全レース4点500円")
    print(f"{'='*90}")
    print(f"\n{'本命閾値':>8}  {'レース数':>6}  {'的中率':>7}  {'投資':>12}  {'回収':>12}  {'ROI':>7}  {'利益':>12}")
    for h_th in [0.0, 0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75, 0.89, 1.0, 1.2]:
        sub = [r for r in races if r['honmei'] >= h_th]
        if len(sub) < 50: continue
        n = len(sub)
        hits = sum(1 for r in sub if r['hit_at'] is not None and r['hit_at'] <= 4)
        bet = n * 4 * 500
        ret = sum(r['payout'] * 5 for r in sub if r['hit_at'] is not None and r['hit_at'] <= 4)
        roi = ret / bet * 100
        print(f"  >={h_th:<5.2f}  {n:>6}  {hits/n*100:>6.1f}%  {bet:>11,}円  {ret:>11,.0f}円  {roi:>6.1f}%  {ret-bet:>+11,.0f}円")

    # ========================================
    # Phase 2: 信頼度閾値（HIGH=2pt, 他=4pt）
    # ========================================
    print(f"\n{'='*90}")
    print(f"  Phase 2: 信頼度閾値（HIGH→2pt, 他→4pt）— 本命>=0.35")
    print(f"{'='*90}")
    target = [r for r in races if r['honmei'] >= 0.35]
    print(f"\n{'信頼度閾値':>10}  {'HIGH件':>6}  {'他件':>5}  {'投資':>12}  {'回収':>12}  {'ROI':>7}  {'利益':>12}")
    for s_th in [1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 3.0, 3.5, 4.0, 5.0]:
        hi = [r for r in target if r['shinrai'] >= s_th]
        lo = [r for r in target if r['shinrai'] < s_th]
        bet = len(hi)*2*500 + len(lo)*4*500
        ret = sum(r['payout']*5 for r in hi if r['hit_at'] is not None and r['hit_at']<=2) + \
              sum(r['payout']*5 for r in lo if r['hit_at'] is not None and r['hit_at']<=4)
        roi = ret/bet*100 if bet>0 else 0
        print(f"    >={s_th:<4.1f}  {len(hi):>6}  {len(lo):>5}  {bet:>11,}円  {ret:>11,.0f}円  {roi:>6.1f}%  {ret-bet:>+11,.0f}円")

    # ========================================
    # Phase 3: 全組み合わせグリッドサーチ
    # ========================================
    print(f"\n{'='*90}")
    print(f"  Phase 3: グリッドサーチ（本命閾値 × 信頼度閾値 × 点数）")
    print(f"{'='*90}")

    all_results = []
    for h_th in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75, 0.89]:
        for s_th in [1.0, 1.5, 2.0, 2.3, 2.5, 3.0, 3.5, 4.0, 5.0]:
            for hi_pts in [1, 2, 3]:
                for lo_pts in [3, 4]:
                    hi = [r for r in races if r['honmei'] >= h_th and r['shinrai'] >= s_th]
                    lo = [r for r in races if r['honmei'] >= h_th and r['shinrai'] < s_th]
                    total_n = len(hi) + len(lo)
                    if total_n < 100: continue

                    bet = len(hi)*hi_pts*500 + len(lo)*lo_pts*500
                    ret = sum(r['payout']*5 for r in hi if r['hit_at'] is not None and r['hit_at']<=hi_pts) + \
                          sum(r['payout']*5 for r in lo if r['hit_at'] is not None and r['hit_at']<=lo_pts)
                    roi = ret/bet*100 if bet>0 else 0

                    all_results.append({
                        'h_th': h_th, 's_th': s_th,
                        'hi_pts': hi_pts, 'lo_pts': lo_pts,
                        'n_hi': len(hi), 'n_lo': len(lo),
                        'bet': bet, 'ret': ret,
                        'roi': roi, 'profit': ret - bet
                    })

    # ROI上位
    all_results.sort(key=lambda x: -x['roi'])
    print(f"\n--- ROI上位20 ---")
    print(f"{'本命':>5} {'信頼度':>6} {'H点':>3} {'他点':>3} {'H件':>5} {'他件':>5} {'投資':>12} {'回収':>12} {'ROI':>7} {'利益':>12}")
    for r in all_results[:20]:
        print(f"  >={r['h_th']:<4.2f} >={r['s_th']:<4.1f} {r['hi_pts']:>3} {r['lo_pts']:>3} {r['n_hi']:>5} {r['n_lo']:>5} {r['bet']:>11,}円 {r['ret']:>11,.0f}円 {r['roi']:>6.1f}% {r['profit']:>+11,.0f}円")

    # 利益上位
    all_results.sort(key=lambda x: -x['profit'])
    print(f"\n--- 利益上位20 ---")
    print(f"{'本命':>5} {'信頼度':>6} {'H点':>3} {'他点':>3} {'H件':>5} {'他件':>5} {'投資':>12} {'回収':>12} {'ROI':>7} {'利益':>12}")
    for r in all_results[:20]:
        print(f"  >={r['h_th']:<4.2f} >={r['s_th']:<4.1f} {r['hi_pts']:>3} {r['lo_pts']:>3} {r['n_hi']:>5} {r['n_lo']:>5} {r['bet']:>11,}円 {r['ret']:>11,.0f}円 {r['roi']:>6.1f}% {r['profit']:>+11,.0f}円")

    # ========================================
    # Phase 4: 年別安定性チェック
    # ========================================
    print(f"\n{'='*90}")
    print(f"  Phase 4: 年別安定性（ROI上位5 + 利益上位5の重複排除）")
    print(f"{'='*90}")

    all_results.sort(key=lambda x: -x['roi'])
    seen = set()
    top_cfgs = []
    for r in all_results:
        key = (r['h_th'], r['s_th'], r['hi_pts'], r['lo_pts'])
        if key not in seen:
            seen.add(key)
            top_cfgs.append(r)
        if len(top_cfgs) >= 5: break
    all_results.sort(key=lambda x: -x['profit'])
    for r in all_results:
        key = (r['h_th'], r['s_th'], r['hi_pts'], r['lo_pts'])
        if key not in seen:
            seen.add(key)
            top_cfgs.append(r)
        if len(top_cfgs) >= 10: break

    # 現行戦略も追加
    curr = [r for r in races if r['honmei'] >= 0.89]
    curr_bet = len(curr)*4*500
    curr_ret = sum(r['payout']*5 for r in curr if r['hit_at'] is not None and r['hit_at']<=4)
    top_cfgs.insert(0, {'h_th': 0.89, 's_th': 999, 'hi_pts': 0, 'lo_pts': 4, 'label': '現行(本命>=0.89, 全4pt)'})

    for cfg in top_cfgs:
        h_th = cfg['h_th']
        s_th = cfg['s_th']
        hi_pts = cfg['hi_pts']
        lo_pts = cfg['lo_pts']
        label = cfg.get('label', f"本命>={h_th:.2f} / 信頼度>={s_th:.1f}->{hi_pts}pt, 他->{lo_pts}pt")
        print(f"\n  [{label}]")
        for year in ['2024', '2025', '2024-2025']:
            yr = [r for r in races if (year == '2024-2025' or r['year'] == year)]
            if s_th >= 999:
                # 現行: 全部同じ点数
                sub = [r for r in yr if r['honmei'] >= h_th]
                bet = len(sub)*lo_pts*500
                ret = sum(r['payout']*5 for r in sub if r['hit_at'] is not None and r['hit_at']<=lo_pts)
            else:
                hi = [r for r in yr if r['honmei'] >= h_th and r['shinrai'] >= s_th]
                lo_r = [r for r in yr if r['honmei'] >= h_th and r['shinrai'] < s_th]
                bet = len(hi)*hi_pts*500 + len(lo_r)*lo_pts*500
                ret = sum(r['payout']*5 for r in hi if r['hit_at'] is not None and r['hit_at']<=hi_pts) + \
                      sum(r['payout']*5 for r in lo_r if r['hit_at'] is not None and r['hit_at']<=lo_pts)
                sub = hi + lo_r
            n = len(sub)
            roi = ret/bet*100 if bet>0 else 0
            hits_n = sum(1 for r in sub if r['hit_at'] is not None and r['hit_at']<=(hi_pts if s_th<999 and r['shinrai']>=s_th else lo_pts))
            print(f"    {year:>9}: {n:>5}R  的中{hits_n:>4} ({hits_n/n*100:.1f}%)  投資{bet:>10,}円  ROI {roi:.1f}%  利益{ret-bet:>+10,.0f}円")

if __name__ == '__main__':
    main()
