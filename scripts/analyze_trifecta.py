"""3連単戦略分析スクリプト"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from db.schema import get_connection
from collections import defaultdict

conn = get_connection()
cur = conn.cursor()

# 予測 vs 実績を取得（結果のあるレースのみ）
cur.execute('''
    SELECT p.race_id, r.velodrome, r.race_number, r.grade, r.date,
           p.rider_id, p.predicted_rank, p.predicted_score, p.mark,
           rr.finish_position,
           e.bike_number, ri.name, ri.class
    FROM predictions p
    JOIN races r ON p.race_id = r.race_id
    JOIN riders ri ON p.rider_id = ri.rider_id
    LEFT JOIN entries e ON p.race_id = e.race_id AND p.rider_id = e.rider_id
    LEFT JOIN race_results rr ON p.race_id = rr.race_id AND p.rider_id = rr.rider_id
    WHERE rr.finish_position IS NOT NULL
    ORDER BY r.date, r.race_id, p.predicted_rank
''')

rows = cur.fetchall()

races = defaultdict(list)
for row in rows:
    race_id, velo, rnum, grade, date, rid, pred_rank, pred_score, mark, finish, bike, name, cls = row
    races[race_id].append({
        'velodrome': velo, 'race_number': rnum, 'grade': grade, 'date': date,
        'rider_id': rid, 'pred_rank': pred_rank, 'pred_score': pred_score,
        'mark': mark, 'finish': finish, 'bike': bike, 'name': name, 'class': cls
    })

print("=== 3連単戦略分析 ===\n")

results = []
for race_id, entries in sorted(races.items()):
    top3_pred = [e for e in entries if e['pred_rank'] <= 3]
    top3_scores = [e['pred_score'] for e in top3_pred]
    rest_scores = [e['pred_score'] for e in entries if e['pred_rank'] > 3]

    # 信頼度: top3の最低スコア - 4位以下の最高スコア
    if top3_scores and rest_scores:
        confidence = min(top3_scores) - max(rest_scores)
    else:
        confidence = 0

    # top1スコアと2位スコアの差
    score_gap_12 = top3_scores[0] - top3_scores[1] if len(top3_scores) >= 2 else 0

    top1_hit = entries[0]['finish'] == 1

    pred_top3 = set(e['rider_id'] for e in entries if e['pred_rank'] <= 3)
    pred_top4 = set(e['rider_id'] for e in entries if e['pred_rank'] <= 4)
    pred_top5 = set(e['rider_id'] for e in entries if e['pred_rank'] <= 5)
    actual_top3 = set(e['rider_id'] for e in entries if e['finish'] and e['finish'] <= 3)

    overlap3 = len(pred_top3 & actual_top3)
    overlap4 = len(pred_top4 & actual_top3)
    overlap5 = len(pred_top5 & actual_top3)

    top3_members_hit = pred_top3 == actual_top3

    # 3連単的中チェック
    trifecta_hit = False
    if top3_members_hit:
        pred_order = [(e['rider_id'], e['pred_rank']) for e in entries if e['pred_rank'] <= 3]
        actual_order = [(e['rider_id'], e['finish']) for e in entries if e['finish'] and e['finish'] <= 3]
        pred_sorted = [rid for rid, _ in sorted(pred_order, key=lambda x: x[1])]
        actual_sorted = [rid for rid, _ in sorted(actual_order, key=lambda x: x[1])]
        trifecta_hit = pred_sorted == actual_sorted

    # 3連単ボックス的中（TOP3メンバー一致 = ボックスなら当たる）
    box3_hit = top3_members_hit
    # TOP4ボックス（4人から3人選ぶ = 24通り）
    box4_hit = overlap4 == 3
    # TOP5ボックス（5人から3人選ぶ = 60通り）
    box5_hit = overlap5 == 3

    actual_order_list = sorted([e for e in entries if e['finish'] and e['finish'] <= 3], key=lambda x: x['finish'])
    actual_str = '-'.join([str(e['bike']) for e in actual_order_list])
    pred_str = '-'.join([str(e['bike']) for e in entries if e['pred_rank'] <= 3])

    results.append({
        'race_id': race_id, 'velo': entries[0]['velodrome'], 'rnum': entries[0]['race_number'],
        'grade': entries[0]['grade'], 'date': entries[0]['date'],
        'confidence': confidence, 'score_gap_12': score_gap_12,
        'top1_hit': top1_hit, 'top3_members': top3_members_hit,
        'trifecta': trifecta_hit, 'box3': box3_hit, 'box4': box4_hit, 'box5': box5_hit,
        'overlap3': overlap3, 'overlap4': overlap4, 'overlap5': overlap5,
        'actual': actual_str, 'pred': pred_str
    })

# 信頼度順にソート
results.sort(key=lambda x: -x['confidence'])

header = "{:<14} {:<4} {:>6}  {:>8} {:>8}  {:>4} {:>5} {:>5} {:>5} {:>5}".format(
    'レース', 'Grade', '信頼度', '予測', '実際', 'TOP1', 'BOX3', 'BOX4', 'BOX5', '3連単')
print(header)
print('-' * 85)

for r in results:
    t1 = 'O' if r['top1_hit'] else 'X'
    b3 = 'O' if r['box3'] else 'X'
    b4 = 'O' if r['box4'] else 'X'
    b5 = 'O' if r['box5'] else 'X'
    tri = 'O' if r['trifecta'] else 'X'
    line = "{:<10} {:<4} {:>6.2f}  {:>8} {:>8}    {} {:>4} {:>4} {:>4} {:>4}".format(
        r['velo'] + str(r['rnum']) + 'R', r['grade'], r['confidence'],
        r['pred'], r['actual'], t1, b3, b4, b5, tri)
    print(line)

print()
print("=== 全体統計 ===")
total = len(results)
top1_cnt = sum(1 for r in results if r['top1_hit'])
box3_cnt = sum(1 for r in results if r['box3'])
box4_cnt = sum(1 for r in results if r['box4'])
box5_cnt = sum(1 for r in results if r['box5'])
tri_cnt = sum(1 for r in results if r['trifecta'])

print(f"レース数: {total}")
print(f"1着的中:           {top1_cnt}/{total} ({100*top1_cnt/total:.1f}%)")
print(f"3連単的中(順序一致): {tri_cnt}/{total} ({100*tri_cnt/total:.1f}%)")
print(f"BOX3的中(TOP3内):   {box3_cnt}/{total} ({100*box3_cnt/total:.1f}%) → 6通り購入")
print(f"BOX4的中(TOP4内):   {box4_cnt}/{total} ({100*box4_cnt/total:.1f}%) → 24通り購入")
print(f"BOX5的中(TOP5内):   {box5_cnt}/{total} ({100*box5_cnt/total:.1f}%) → 60通り購入")

# 信頼度別分析
print()
print("=== 信頼度別分析 ===")
thresholds = [
    ('高信頼 (>=1.0)', lambda r: r['confidence'] >= 1.0),
    ('中高信頼 (0.5-1.0)', lambda r: 0.5 <= r['confidence'] < 1.0),
    ('中信頼 (0.0-0.5)', lambda r: 0.0 <= r['confidence'] < 0.5),
    ('低信頼 (<0.0)', lambda r: r['confidence'] < 0.0),
]

for label, cond in thresholds:
    group = [r for r in results if cond(r)]
    if not group:
        print(f"{label}: 該当なし")
        continue
    n = len(group)
    t1 = sum(1 for r in group if r['top1_hit'])
    b3 = sum(1 for r in group if r['box3'])
    b4 = sum(1 for r in group if r['box4'])
    b5 = sum(1 for r in group if r['box5'])
    tri = sum(1 for r in group if r['trifecta'])
    print(f"{label}: {n}レース")
    print(f"  1着={t1}/{n}({100*t1/n:.0f}%) BOX3={b3}/{n}({100*b3/n:.0f}%) BOX4={b4}/{n}({100*b4/n:.0f}%) BOX5={b5}/{n}({100*b5/n:.0f}%) 3連単={tri}/{n}({100*tri/n:.0f}%)")

# Grade別分析
print()
print("=== Grade別分析 ===")
grades = sorted(set(r['grade'] for r in results))
for grade in grades:
    group = [r for r in results if r['grade'] == grade]
    n = len(group)
    t1 = sum(1 for r in group if r['top1_hit'])
    b3 = sum(1 for r in group if r['box3'])
    b4 = sum(1 for r in group if r['box4'])
    tri = sum(1 for r in group if r['trifecta'])
    print(f"{grade}: {n}レース | 1着={t1}/{n}({100*t1/n:.0f}%) | BOX3={b3}/{n}({100*b3/n:.0f}%) | BOX4={b4}/{n}({100*b4/n:.0f}%) | 3連単={tri}/{n}({100*tri/n:.0f}%)")

# 収支シミュレーション
print()
print("=== 収支シミュレーション (100円/通りで計算) ===")
print()

# 各戦略のシミュレーション
# 3連単ボックスのコスト
strategies = [
    ('戦略A: 全レース BOX3 (6通り)', 6, lambda r: True, lambda r: r['box3']),
    ('戦略B: 全レース BOX4 (24通り)', 24, lambda r: True, lambda r: r['box4']),
    ('戦略C: 高信頼のみ BOX3', 6, lambda r: r['confidence'] >= 0.5, lambda r: r['box3']),
    ('戦略D: 高信頼のみ BOX4', 24, lambda r: r['confidence'] >= 0.5, lambda r: r['box4']),
    ('戦略E: 中高以上 BOX3', 6, lambda r: r['confidence'] >= 0.0, lambda r: r['box3']),
    ('戦略F: フォーメーション(◎→○▲→○▲△△) 12通り', 12, lambda r: True, lambda r: r['box4']),  # 近似
]

# 3連単の平均配当を推定（一般的に7000-15000円程度）
avg_payouts = [5000, 8000, 12000, 20000]

print("※ 3連単平均配当は実データがないため推定値で計算")
print()

for label, cost_per_race, filter_fn, hit_fn in strategies:
    target = [r for r in results if filter_fn(r)]
    hits = sum(1 for r in target if hit_fn(r))
    races_bet = len(target)
    total_cost = races_bet * cost_per_race * 100
    hit_rate = 100 * hits / races_bet if races_bet else 0
    print(f"{label}")
    print(f"  対象: {races_bet}レース | 的中: {hits}回 ({hit_rate:.1f}%) | 投資: {total_cost:,}円")
    for payout in avg_payouts:
        revenue = hits * payout
        profit = revenue - total_cost
        roi = 100 * revenue / total_cost if total_cost > 0 else 0
        print(f"    配当{payout:,}円想定: 回収{revenue:,}円 | 収支{profit:+,}円 | 回収率{roi:.0f}%")
    print()

conn.close()
