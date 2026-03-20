# -*- coding: utf-8 -*-
"""戦略の堅牢性を多角的に検証"""

import sys, io, re, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from config import RESULTS_DIR, CACHE_DIR
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from db.schema import get_connection

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
    # 2連単
    m = re.search(r'２車単\s+(\d+[->＞\s]+\d+)\s+([\d,]+)円\s+(\d+)人気', text)
    if not m: return None
    bikes = [int(b) for b in re.findall(r'\d+', m.group(1))]
    amount_nisyatan = int(m.group(2).replace(',', ''))
    # 2連複
    m2 = re.search(r'２車複\s+(\d+[->＞\s—=]+\d+)\s+([\d,]+)円', text)
    amount_nishafuku = int(m2.group(2).replace(',', '')) if m2 else None
    # ワイド (multiple results possible)
    wides = re.findall(r'ワイド\s+(\d+)[->＞\s—=]+(\d+)\s+([\d,]+)円', text)
    return {
        'bikes': bikes,
        'amount': amount_nisyatan,
        'amount_nishafuku': amount_nishafuku,
        'wides': [(int(w[0]), int(w[1]), int(w[2].replace(',',''))) for w in wides],
    }

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

    # グレード情報を取得
    conn = get_connection()
    race_info = pd.read_sql("SELECT race_id, grade, velodrome, date FROM races", conn)
    conn.close()

    races = []
    for race_id, group in all_df.groupby('race_id'):
        if len(group) < 5: continue
        ranked = group.sort_values('pred_score', ascending=False)
        bikes = ranked['bike_number'].astype(int).tolist()
        scores = ranked['pred_score'].tolist()

        payout = parse_nisyatan(race_id)
        if not payout: continue

        a1, a2 = payout['bikes'][0], payout['bikes'][1]
        hit_at = None
        if bikes[0] == a1:
            for i in range(1, min(6, len(bikes))):
                if bikes[i] == a2:
                    hit_at = i
                    break

        # 2連複: 着順不問で◎が1-2着に入り、相手も1-2着
        nishafuku_hit = (a1 in bikes[:2] and a2 in bikes[:2]) if len(bikes) >= 2 else False

        # ワイド: ◎と相手が両方3着以内
        wide_hits = []
        for w_a, w_b, w_amt in (payout.get('wides') or []):
            for i in range(1, min(4, len(bikes))):
                if bikes[0] in (w_a, w_b) and bikes[i] in (w_a, w_b):
                    wide_hits.append({'partner_rank': i, 'amount': w_amt})
                    break

        info = race_info[race_info['race_id'] == race_id]
        grade = info['grade'].iloc[0] if len(info) > 0 else ''
        velodrome = info['velodrome'].iloc[0] if len(info) > 0 else ''
        date = info['date'].iloc[0] if len(info) > 0 else ''

        races.append({
            'race_id': race_id,
            'date': date,
            'year': race_id[:4],
            'month': date[5:7] if date and len(date) >= 7 else '',
            'year_month': date[:7] if date and len(date) >= 7 else '',
            'grade': grade,
            'velodrome': velodrome,
            'honmei': scores[0] - scores[1],
            'shinrai': scores[0] - scores[2],
            'hit_at': hit_at,
            'payout': payout['amount'],
            'nishafuku_hit': nishafuku_hit,
            'nishafuku_payout': payout.get('amount_nishafuku'),
            'wide_hits': wide_hits,
            'top1_actual': a1 == bikes[0],
        })

    print(f"解析対象: {len(races)}レース")

    # ========================================
    # 検証1: 月別ROI推移（安定性チェック）
    # ========================================
    print(f"\n{'='*90}")
    print(f"  検証1: 月別ROI推移（本命>=0.89, 3点, 500円）")
    print(f"{'='*90}")
    print(f"{'月':>8}  {'レース数':>6}  {'的中':>4}  {'的中率':>7}  {'投資':>10}  {'回収':>12}  {'ROI':>7}  {'利益':>12}")

    monthly_rois = []
    for ym in sorted(set(r['year_month'] for r in races if r['year_month'])):
        sub = [r for r in races if r['year_month'] == ym and r['honmei'] >= 0.89]
        if len(sub) < 3: continue
        n = len(sub)
        hits = sum(1 for r in sub if r['hit_at'] is not None and r['hit_at'] <= 3)
        bet = n * 3 * 500
        ret = sum(r['payout'] * 5 for r in sub if r['hit_at'] is not None and r['hit_at'] <= 3)
        roi = ret / bet * 100 if bet > 0 else 0
        monthly_rois.append(roi)
        print(f"  {ym:>7}  {n:>6}  {hits:>4}  {hits/n*100:>6.1f}%  {bet:>9,}円  {ret:>11,.0f}円  {roi:>6.1f}%  {ret-bet:>+11,.0f}円")

    if monthly_rois:
        print(f"\n  月ROI中央値: {np.median(monthly_rois):.1f}%  平均: {np.mean(monthly_rois):.1f}%  最低: {np.min(monthly_rois):.1f}%  最高: {np.max(monthly_rois):.1f}%")
        profitable_months = sum(1 for r in monthly_rois if r > 100)
        print(f"  黒字月: {profitable_months}/{len(monthly_rois)} ({profitable_months/len(monthly_rois)*100:.0f}%)")

    # ========================================
    # 検証2: 配当分布（外れ値の影響）
    # ========================================
    print(f"\n{'='*90}")
    print(f"  検証2: 配当分布分析（本命>=0.89, 3pt的中時）")
    print(f"{'='*90}")

    hit_payouts = [r['payout'] for r in races if r['honmei'] >= 0.89 and r['hit_at'] is not None and r['hit_at'] <= 3]
    if hit_payouts:
        arr = np.array(hit_payouts)
        total_ret = arr.sum() * 5
        print(f"  的中件数: {len(arr)}")
        print(f"  配当分布: 中央値 {np.median(arr):,.0f}円  平均 {np.mean(arr):,.0f}円  最大 {np.max(arr):,.0f}円")
        print(f"  25%ile: {np.percentile(arr,25):,.0f}円  75%ile: {np.percentile(arr,75):,.0f}円")

        # 上位10件を除いた場合のROI
        all_high = [r for r in races if r['honmei'] >= 0.89]
        bet_total = len(all_high) * 3 * 500
        sorted_payouts = sorted(hit_payouts, reverse=True)
        for remove_n in [0, 3, 5, 10, 20]:
            remaining = sorted_payouts[remove_n:]
            ret = sum(remaining) * 5
            ret_removed = sum(sorted_payouts[:remove_n]) * 5 if remove_n > 0 else 0
            roi = (ret + ret_removed) / bet_total * 100
            roi_without = ret / bet_total * 100
            if remove_n == 0:
                print(f"\n  全的中含む: ROI {roi:.1f}%")
            else:
                print(f"  上位{remove_n:>2}件除外: ROI {roi_without:.1f}% (除外分 {ret_removed:,.0f}円)")

    # ========================================
    # 検証3: グレード別分析
    # ========================================
    print(f"\n{'='*90}")
    print(f"  検証3: グレード別ROI（本命>=0.89, 3pt）")
    print(f"{'='*90}")
    print(f"{'グレード':>10}  {'レース数':>6}  {'的中率':>7}  {'ROI':>7}  {'平均配当':>10}")

    for grade in ['FII', 'FI', 'GIII', 'GII', 'GI', 'GP']:
        sub = [r for r in races if r['honmei'] >= 0.89 and r['grade'] == grade]
        if len(sub) < 10: continue
        hits = [r for r in sub if r['hit_at'] is not None and r['hit_at'] <= 3]
        bet = len(sub) * 3 * 500
        ret = sum(r['payout'] * 5 for r in hits)
        roi = ret / bet * 100 if bet > 0 else 0
        avg_pay = np.mean([r['payout'] for r in hits]) if hits else 0
        print(f"  {grade:>8}  {len(sub):>6}  {len(hits)/len(sub)*100:>6.1f}%  {roi:>6.1f}%  {avg_pay:>9,.0f}円")

    # ========================================
    # 検証4: 2連単 vs 2連複 vs ワイド
    # ========================================
    print(f"\n{'='*90}")
    print(f"  検証4: 賭式比較（本命>=0.89）")
    print(f"{'='*90}")

    high_races = [r for r in races if r['honmei'] >= 0.89]
    n = len(high_races)

    # 2連単 3pt
    nisyatan_hits = [r for r in high_races if r['hit_at'] is not None and r['hit_at'] <= 3]
    nisyatan_bet = n * 3 * 500
    nisyatan_ret = sum(r['payout'] * 5 for r in nisyatan_hits)

    # 2連複: ◎軸流し3点（○▲△との組み合わせ）
    # 的中条件: ◎が1-2着かつ2-4位予測のいずれかが1-2着
    nishafuku_hits_3pt = []
    for r in high_races:
        if r['nishafuku_hit'] and r['nishafuku_payout']:
            nishafuku_hits_3pt.append(r)
    nishafuku_bet = n * 3 * 500
    nishafuku_ret = sum(r['nishafuku_payout'] * 5 for r in nishafuku_hits_3pt)

    # ワイド: ◎軸流し3点
    wide_hits_3pt = []
    wide_ret = 0
    for r in high_races:
        for wh in r.get('wide_hits', []):
            if wh['partner_rank'] <= 3:
                wide_ret += wh['amount'] * 5
                if r not in wide_hits_3pt:
                    wide_hits_3pt.append(r)
    wide_bet = n * 3 * 500

    print(f"\n  {'賭式':>8}  {'レース':>5}  {'的中':>4}  {'的中率':>7}  {'投資':>11}  {'回収':>12}  {'ROI':>7}")
    print(f"  {'2連単3pt':>8}  {n:>5}  {len(nisyatan_hits):>4}  {len(nisyatan_hits)/n*100:>6.1f}%  {nisyatan_bet:>10,}円  {nisyatan_ret:>11,.0f}円  {nisyatan_ret/nisyatan_bet*100:>6.1f}%")
    print(f"  {'2連複3pt':>8}  {n:>5}  {len(nishafuku_hits_3pt):>4}  {len(nishafuku_hits_3pt)/n*100:>6.1f}%  {nishafuku_bet:>10,}円  {nishafuku_ret:>11,.0f}円  {nishafuku_ret/nishafuku_bet*100:>6.1f}%")
    print(f"  {'ワイド3pt':>8}  {n:>5}  {len(wide_hits_3pt):>4}  {len(wide_hits_3pt)/n*100:>6.1f}%  {wide_bet:>10,}円  {wide_ret:>11,.0f}円  {wide_ret/wide_bet*100:>6.1f}%")

    # ========================================
    # 検証5: ◎的中率の真のパフォーマンス
    # ========================================
    print(f"\n{'='*90}")
    print(f"  検証5: ◎1着精度の時系列推移")
    print(f"{'='*90}")
    print(f"{'月':>8}  {'レース数':>6}  {'◎1着':>5}  {'◎1着率':>7}  {'◎1-2着':>6}  {'◎1-2着率':>8}")

    for ym in sorted(set(r['year_month'] for r in races if r['year_month'])):
        sub = [r for r in races if r['year_month'] == ym and r['honmei'] >= 0.89]
        if len(sub) < 3: continue
        top1 = sum(1 for r in sub if r['top1_actual'])
        top12 = sum(1 for r in sub if r['hit_at'] is not None)  # ◎1着で2着に相手がいる = top1かつ的中
        top12_count = sum(1 for r in sub if r['top1_actual'])
        print(f"  {ym:>7}  {len(sub):>6}  {top1:>5}  {top1/len(sub)*100:>6.1f}%  {top12_count:>6}  {top12_count/len(sub)*100:>7.1f}%")

    # ========================================
    # 検証6: Bootstrap信頼区間
    # ========================================
    print(f"\n{'='*90}")
    print(f"  検証6: Bootstrap 95%信頼区間（本命>=0.89, 3pt）")
    print(f"{'='*90}")

    high_races_arr = high_races.copy()
    n_boot = 1000
    boot_rois = []
    np.random.seed(42)
    for _ in range(n_boot):
        sample = np.random.choice(len(high_races_arr), size=len(high_races_arr), replace=True)
        sampled = [high_races_arr[i] for i in sample]
        bet = len(sampled) * 3 * 500
        ret = sum(r['payout'] * 5 for r in sampled if r['hit_at'] is not None and r['hit_at'] <= 3)
        boot_rois.append(ret / bet * 100)

    boot_rois = np.array(boot_rois)
    ci_lower = np.percentile(boot_rois, 2.5)
    ci_upper = np.percentile(boot_rois, 97.5)
    print(f"  サンプル数: {len(high_races_arr)}レース, Bootstrap {n_boot}回")
    print(f"  ROI: {np.mean(boot_rois):.1f}% (95%CI: {ci_lower:.1f}% - {ci_upper:.1f}%)")
    print(f"  ROI < 100% の確率: {(boot_rois < 100).mean()*100:.1f}%")

    # 3pt vs 4pt のBootstrap比較
    boot_diff = []
    for _ in range(n_boot):
        sample = np.random.choice(len(high_races_arr), size=len(high_races_arr), replace=True)
        sampled = [high_races_arr[i] for i in sample]
        bet3 = len(sampled) * 3 * 500
        ret3 = sum(r['payout'] * 5 for r in sampled if r['hit_at'] is not None and r['hit_at'] <= 3)
        bet4 = len(sampled) * 4 * 500
        ret4 = sum(r['payout'] * 5 for r in sampled if r['hit_at'] is not None and r['hit_at'] <= 4)
        boot_diff.append(ret3/bet3*100 - ret4/bet4*100)

    boot_diff = np.array(boot_diff)
    print(f"\n  3pt vs 4pt ROI差: {np.mean(boot_diff):+.1f}% (95%CI: {np.percentile(boot_diff,2.5):+.1f}% - {np.percentile(boot_diff,97.5):+.1f}%)")
    print(f"  3pt > 4pt の確率: {(boot_diff > 0).mean()*100:.1f}%")

    # ========================================
    # 検証7: 最大ドローダウン
    # ========================================
    print(f"\n{'='*90}")
    print(f"  検証7: 最大ドローダウン分析（本命>=0.89, 3pt, 日別）")
    print(f"{'='*90}")

    daily_pnl = {}
    for r in high_races:
        d = r['date']
        if d not in daily_pnl:
            daily_pnl[d] = {'bet': 0, 'ret': 0}
        daily_pnl[d]['bet'] += 3 * 500
        if r['hit_at'] is not None and r['hit_at'] <= 3:
            daily_pnl[d]['ret'] += r['payout'] * 5

    dates_sorted = sorted(daily_pnl.keys())
    cumulative = 0
    peak = 0
    max_dd = 0
    dd_start = dd_end = None
    consecutive_loss = 0
    max_consecutive_loss = 0
    daily_profits = []

    for d in dates_sorted:
        pnl = daily_pnl[d]['ret'] - daily_pnl[d]['bet']
        daily_profits.append(pnl)
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
        if pnl < 0:
            consecutive_loss += 1
            max_consecutive_loss = max(max_consecutive_loss, consecutive_loss)
        else:
            consecutive_loss = 0

    print(f"  運用日数: {len(dates_sorted)}日")
    print(f"  最大ドローダウン: {max_dd:,.0f}円")
    print(f"  最大連敗日数: {max_consecutive_loss}日")
    print(f"  日別損益: 中央値 {np.median(daily_profits):+,.0f}円  平均 {np.mean(daily_profits):+,.0f}円")
    loss_days = sum(1 for p in daily_profits if p < 0)
    print(f"  赤字日数: {loss_days}/{len(daily_profits)} ({loss_days/len(daily_profits)*100:.0f}%)")

    # ========================================
    # 検証8: オッズとの関係
    # ========================================
    print(f"\n{'='*90}")
    print(f"  検証8: 本命オッズ別ROI（本命>=0.89, 3pt）")
    print(f"{'='*90}")

    # DBからオッズ情報を取得
    conn = get_connection()
    odds_df = pd.read_sql("SELECT race_id, bike_number, odds FROM race_results WHERE odds IS NOT NULL", conn)
    conn.close()

    for r in high_races:
        rid = r['race_id']
        bike1 = all_df[all_df['race_id'] == rid].sort_values('pred_score', ascending=False).iloc[0]['bike_number']
        o = odds_df[(odds_df['race_id'] == rid) & (odds_df['bike_number'] == int(bike1))]
        r['honmei_odds'] = float(o['odds'].iloc[0]) if len(o) > 0 else None

    print(f"{'オッズ帯':>10}  {'レース数':>6}  {'的中率':>7}  {'ROI':>7}  {'平均配当':>10}")
    odds_bins = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 50)]
    for lo, hi in odds_bins:
        sub = [r for r in high_races if r.get('honmei_odds') and lo <= r['honmei_odds'] < hi]
        if len(sub) < 10: continue
        hits = [r for r in sub if r['hit_at'] is not None and r['hit_at'] <= 3]
        bet = len(sub) * 3 * 500
        ret = sum(r['payout'] * 5 for r in hits)
        roi = ret / bet * 100 if bet > 0 else 0
        avg_pay = np.mean([r['payout'] for r in hits]) if hits else 0
        print(f"  {lo:.0f}-{hi:.0f}倍  {len(sub):>6}  {len(hits)/len(sub)*100:>6.1f}%  {roi:>6.1f}%  {avg_pay:>9,.0f}円")

if __name__ == '__main__':
    main()
