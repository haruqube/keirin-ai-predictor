"""予測精度追跡（改善版）

改善点:
- 回収率（ROI）シミュレーション追加
- グレード別精度分析
- 詳細統計（レース数、的中率、回収率）
"""

import logging
from collections import defaultdict
from db.schema import get_connection

logger = logging.getLogger(__name__)


def evaluate_predictions(date: str | None = None, verbose: bool = True) -> dict:
    """予測と実際の結果を照合して精度を計算"""
    conn = get_connection()

    query = """
        SELECT DISTINCT p.race_id
        FROM predictions p
        JOIN races r ON p.race_id = r.race_id
    """
    params = []
    if date:
        query += " WHERE r.date = ?"
        params.append(date)
    query += " ORDER BY r.date, p.race_id"

    pred_races = conn.execute(query, params).fetchall()

    total_races = 0
    top1_hits = 0
    top3_overlap = 0
    roi_bets = 0
    roi_returns = 0.0

    # グレード別統計
    grade_stats = defaultdict(lambda: {"races": 0, "top1": 0, "top3": 0})

    for pr in pred_races:
        race_id = pr["race_id"]

        results = conn.execute("""
            SELECT rider_id, finish_position, odds
            FROM race_results WHERE race_id = ? AND finish_position IS NOT NULL
            ORDER BY finish_position
        """, (race_id,)).fetchall()

        preds = conn.execute("""
            SELECT rider_id, predicted_rank, mark
            FROM predictions WHERE race_id = ?
            ORDER BY predicted_rank
        """, (race_id,)).fetchall()

        race = conn.execute(
            "SELECT grade FROM races WHERE race_id = ?", (race_id,)
        ).fetchone()

        if not results or not preds:
            continue

        total_races += 1
        grade = race["grade"] if race else "unknown"

        actual_top1 = results[0]["rider_id"]
        actual_top1_odds = results[0]["odds"] or 0
        actual_top3 = set(r["rider_id"] for r in results[:3])
        pred_top1 = preds[0]["rider_id"]
        pred_top3 = set(p["rider_id"] for p in preds[:3])

        # Top1
        if pred_top1 == actual_top1:
            top1_hits += 1
            grade_stats[grade]["top1"] += 1
            roi_returns += actual_top1_odds  # 単勝払い戻し

        # Top3
        overlap = len(pred_top3 & actual_top3)
        top3_overlap += overlap
        grade_stats[grade]["top3"] += overlap
        grade_stats[grade]["races"] += 1

        # ROI: 本命（◎）単勝100円ベット
        roi_bets += 100

    conn.close()

    result = {
        "total_races": total_races,
        "top1_hits": top1_hits,
        "top1_accuracy": top1_hits / total_races if total_races > 0 else 0.0,
        "top3_overlap": top3_overlap,
        "top3_rate": top3_overlap / (total_races * 3) if total_races > 0 else 0.0,
        "roi_bets": roi_bets,
        "roi_returns": roi_returns,
        "roi_rate": (roi_returns / roi_bets * 100) if roi_bets > 0 else 0.0,
        "grade_stats": dict(grade_stats),
    }

    if verbose and total_races > 0:
        print(f"\n{'='*50}")
        print(f" 予測精度レポート ({total_races} races)")
        print(f"{'='*50}")
        print(f" Top1 的中率: {top1_hits}/{total_races} ({result['top1_accuracy']*100:.1f}%)")
        print(f" Top3 一致率: {top3_overlap}/{total_races*3} ({result['top3_rate']*100:.1f}%)")
        print(f" 単勝回収率:  {result['roi_rate']:.1f}% (投資: {roi_bets}円 / 回収: {roi_returns:.0f}円)")
        print()

        # グレード別
        if grade_stats:
            print(f" {'グレード':>8}  {'レース':>5}  {'Top1':>8}  {'Top3率':>8}")
            print(f" {'-'*8}  {'-'*5}  {'-'*8}  {'-'*8}")
            for grade in sorted(grade_stats.keys()):
                s = grade_stats[grade]
                t1_rate = s['top1'] / s['races'] * 100 if s['races'] > 0 else 0
                t3_rate = s['top3'] / (s['races'] * 3) * 100 if s['races'] > 0 else 0
                print(f" {grade or 'N/A':>8}  {s['races']:>5}  {t1_rate:>7.1f}%  {t3_rate:>7.1f}%")
        print(f"{'='*50}\n")

    return result
