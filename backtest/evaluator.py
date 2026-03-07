"""予測精度追跡"""

import logging
from db.schema import get_connection

logger = logging.getLogger(__name__)


def evaluate_predictions(date: str | None = None):
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

        if not results or not preds:
            continue

        total_races += 1
        actual_top1 = results[0]["rider_id"]
        actual_top3 = set(r["rider_id"] for r in results[:3])
        pred_top1 = preds[0]["rider_id"]
        pred_top3 = set(p["rider_id"] for p in preds[:3])

        if pred_top1 == actual_top1:
            top1_hits += 1
        top3_overlap += len(pred_top3 & actual_top3)

    conn.close()

    if total_races > 0:
        print(f"Evaluated: {total_races} races")
        print(f"Top1 accuracy: {top1_hits}/{total_races} ({top1_hits/total_races*100:.1f}%)")
        print(f"Top3 overlap:  {top3_overlap}/{total_races*3} ({top3_overlap/(total_races*3)*100:.1f}%)")
    else:
        print("No races to evaluate")

    return {
        "total_races": total_races,
        "top1_hits": top1_hits,
        "top3_overlap": top3_overlap,
    }
