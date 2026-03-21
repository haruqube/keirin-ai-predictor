"""予測レポート生成スクリプト

DBの予測データからnote.com用マークダウン記事とXティーザーを生成。
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR
from db.schema import get_connection
from publishing.note_formatter import NoteFormatter

MARKS = ["◎", "○", "▲", "△", "△"]


def get_predictions_for_date(date: str) -> list[dict]:
    """指定日の予測データをDBから取得してレース単位にまとめる"""
    conn = get_connection()

    races_rows = conn.execute("""
        SELECT DISTINCT r.race_id, r.velodrome, r.race_number, r.race_name,
               r.grade, r.rider_count
        FROM races r
        JOIN predictions p ON r.race_id = p.race_id
        WHERE r.date = ?
        ORDER BY r.velodrome, r.race_number
    """, (date,)).fetchall()

    races = []
    for race_row in races_rows:
        race_id = race_row["race_id"]

        preds = conn.execute("""
            SELECT p.rider_id, p.predicted_rank, p.predicted_score, p.mark,
                   p.confidence,
                   rd.name, rd.class
            FROM predictions p
            LEFT JOIN riders rd ON p.rider_id = rd.rider_id
            WHERE p.race_id = ?
            ORDER BY p.predicted_rank
        """, (race_id,)).fetchall()

        riders = []
        for pred in preds:
            riders.append({
                "mark": pred["mark"] or "",
                "bike_number": "",  # will be filled from entries
                "name": pred["name"] or "",
                "class": pred["class"] or "",
                "score": pred["predicted_score"] or 0.0,
            })

        # 車番を出走表から取得
        entries = conn.execute("""
            SELECT rider_id, bike_number FROM entries WHERE race_id = ?
        """, (race_id,)).fetchall()
        bike_map = {e["rider_id"]: e["bike_number"] for e in entries}

        for i, pred in enumerate(preds):
            riders[i]["bike_number"] = bike_map.get(pred["rider_id"], "")

        # 信頼度・推奨賭け金
        confidence = preds[0]["confidence"] if preds else 0.0
        grade = race_row["grade"] or ""
        if grade == "F2":
            bet_label, bet_rec = "SKIP", "見送り（F2）"
        elif confidence >= 1.00:
            bet_label, bet_rec = "HIGH", "500円×4点=2,000円"
        elif confidence >= 0.80:
            bet_label, bet_rec = "MED+", "200円×4点=800円"
        elif confidence >= 0.50:
            bet_label, bet_rec = "MED", "100円×4点=400円"
        else:
            bet_label, bet_rec = "LOW", "見送り"

        races.append({
            "race_id": race_id,
            "velodrome": race_row["velodrome"],
            "race_number": race_row["race_number"],
            "race_name": race_row["race_name"] or "",
            "grade": grade,
            "rider_count": race_row["rider_count"],
            "riders": riders,
            "confidence": confidence or 0.0,
            "bet_label": bet_label,
            "bet_rec": bet_rec,
        })

    conn.close()
    return races


def build_top_races(races: list[dict], n: int = 3) -> list[dict]:
    """X投稿用に注目レース（本命スコア上位）を抽出"""
    scored = []
    for race in races:
        if not race["riders"]:
            continue
        top_rider = race["riders"][0]
        scored.append({
            "velodrome": race["velodrome"],
            "race_number": race["race_number"],
            "honmei_mark": "◎",
            "honmei_bike": top_rider["bike_number"],
            "honmei_name": top_rider["name"],
            "top_score": top_rider["score"],
        })
    scored.sort(key=lambda x: x["top_score"], reverse=True)
    return scored[:n]


def generate_article(date: str, last_week_accuracy: str = "集計中"):
    """予測記事を生成して保存"""
    formatted_date = f"{date[:4]}/{int(date[4:6])}/{int(date[6:8])}"
    # 曜日を取得
    from datetime import datetime
    dt = datetime.strptime(date, "%Y%m%d")
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    date_display = f"{formatted_date}({weekdays[dt.weekday()]})"

    races = get_predictions_for_date(
        f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    )

    if not races:
        print(f"[記事生成] {date} の予測データがありません")
        return

    # 開催場一覧
    velodromes = sorted(set(r["velodrome"] for r in races))
    venue_display = "・".join(velodromes)

    formatter = NoteFormatter()

    # note.com記事生成
    article = formatter.generate_article(
        date_display=date_display,
        venue_display=venue_display,
        races=races,
        last_week_accuracy=last_week_accuracy,
    )
    filename = f"article_{date}.md"
    path = formatter.save_article(article, filename)
    print(f"[記事生成] note記事保存: {path}")

    # Xティーザー生成
    top_races = build_top_races(races)
    teaser = formatter.generate_x_teaser(
        date_display=date_display,
        venue_display=venue_display,
        top_races=top_races,
    )
    teaser_path = RESULTS_DIR / f"x_teaser_{date}.txt"
    teaser_path.write_text(teaser, encoding="utf-8")
    print(f"[記事生成] Xティーザー保存: {teaser_path}")
    print(f"\n--- Xティーザー ---\n{teaser}\n---")

    return {"article_path": path, "teaser_path": str(teaser_path), "teaser": teaser}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="予測記事生成")
    parser.add_argument("--date", required=True, help="対象日 (YYYYMMDD)")
    parser.add_argument("--accuracy", default="集計中", help="先週的中率の表示テキスト")
    args = parser.parse_args()
    generate_article(args.date, args.accuracy)
