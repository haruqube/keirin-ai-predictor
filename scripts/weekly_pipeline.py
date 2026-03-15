"""週間予測パイプライン

指定週の開催レースを取得し、出走表データに基づく予測を生成する。
学習済みモデルがある場合はそれを使用し、なければ競走得点・勝率ベースの
ヒューリスティック予測を行う。

Usage:
    python scripts/weekly_pipeline.py                    # 今週の予測
    python scripts/weekly_pipeline.py --date 20260316    # 指定日を含む週
    python scripts/weekly_pipeline.py --demo             # デモモード（サンプルデータ）
"""

import sys
import logging
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR, VELODROME_CODES, CLASS_MAP, GRADE_MAP

try:
    from data.scraper import KeirinScraper
except ImportError:
    KeirinScraper = None

logger = logging.getLogger(__name__)

MARKS = ["◎", "○", "▲", "△", "△"]


def get_week_dates(ref_date: str | None = None) -> list[str]:
    """月曜〜日曜のYYYYMMDD日付リストを返す"""
    if ref_date:
        d = datetime.strptime(ref_date, "%Y%m%d")
    else:
        d = datetime.now()
    monday = d - timedelta(days=d.weekday())
    return [(monday + timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]


def heuristic_score(entry: dict) -> float:
    """出走表データからヒューリスティックスコアを算出

    競走得点(50%) + 勝率(20%) + 3連対率(15%) + 級班(15%)
    """
    score = 0.0

    # 競走得点 (max ~120, normalize to 0-100)
    comp_score = entry.get("avg_competition_score")
    if comp_score and comp_score > 0:
        score += comp_score * 0.5
    else:
        score += 40 * 0.5  # デフォルト

    # 勝率 (0-100%)
    win_rate = entry.get("win_rate")
    if win_rate and win_rate > 0:
        score += win_rate * 0.2

    # 3連対率 (0-100%)
    top3_rate = entry.get("top3_rate")
    if top3_rate and top3_rate > 0:
        score += top3_rate * 0.15

    # 級班ボーナス
    rider_class = entry.get("class", "")
    class_bonus = {
        "SS": 15, "S1": 12, "S2": 10,
        "A1": 7, "A2": 4, "A3": 2,
    }
    score += class_bonus.get(rider_class, 0)

    return score


def predict_day(scraper: KeirinScraper, date: str) -> list[dict]:
    """1日分の全レース予測を生成"""
    formatted = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    day_predictions = []

    race_ids = scraper.scrape_race_list(date)
    if not race_ids:
        return day_predictions

    logger.info("%s: %d races found", formatted, len(race_ids))

    for race_id in race_ids:
        try:
            entry_data = scraper.scrape_race_entry(race_id)
            entries = entry_data.get("entries", [])
            if not entries:
                continue

            # スコア計算 & ソート
            for e in entries:
                e["pred_score"] = heuristic_score(e)
            entries.sort(key=lambda e: e["pred_score"], reverse=True)

            # 予測マーク付与
            for i, e in enumerate(entries):
                e["mark"] = MARKS[i] if i < len(MARKS) else ""
                e["predicted_rank"] = i + 1

            race_pred = {
                "race_id": race_id,
                "date": formatted,
                "velodrome": entry_data.get("velodrome", ""),
                "race_number": entry_data.get("race_number"),
                "race_name": entry_data.get("race_name", ""),
                "grade": entry_data.get("grade"),
                "entries": entries,
            }
            day_predictions.append(race_pred)

        except Exception as e:
            logger.warning("Error predicting %s: %s", race_id, e)
            continue

    return day_predictions


def format_race_prediction(race: dict) -> str:
    """1レースの予測をフォーマット"""
    lines = []
    velodrome = race["velodrome"]
    rnum = race.get("race_number", "?")
    grade = race.get("grade", "")
    race_name = race.get("race_name", "")

    header = f"  {velodrome} {rnum}R"
    if grade:
        header += f" [{grade}]"
    if race_name:
        header += f" {race_name}"
    lines.append(header)

    entries = race.get("entries", [])
    for e in entries[:5]:  # Top 5
        mark = e.get("mark", "")
        name = e.get("rider_name", "?")
        cls = e.get("class", "")
        pref = e.get("prefecture", "")
        score = e.get("pred_score", 0)
        bike = e.get("bike_number", "?")
        lines.append(f"    {mark} {bike}番 {name}({pref}/{cls}) スコア:{score:.1f}")

    return "\n".join(lines)


def format_weekly_report(weekly: dict[str, list[dict]]) -> str:
    """週間予測レポートをフォーマット"""
    output = []
    output.append("=" * 60)
    output.append("  競輪AI予測 週間レポート")
    output.append("=" * 60)

    total_races = 0
    for date_str in sorted(weekly.keys()):
        races = weekly[date_str]
        if not races:
            continue

        formatted_date = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}"
        weekday_names = ["月", "火", "水", "木", "金", "土", "日"]
        d = datetime.strptime(date_str, "%Y%m%d")
        weekday = weekday_names[d.weekday()]

        output.append("")
        output.append(f"■ {formatted_date}({weekday}) - {len(races)}レース")
        output.append("-" * 50)

        # 競輪場ごとにグループ化
        by_velodrome: dict[str, list[dict]] = {}
        for race in races:
            v = race.get("velodrome", "不明")
            by_velodrome.setdefault(v, []).append(race)

        for velodrome, v_races in by_velodrome.items():
            v_races.sort(key=lambda r: r.get("race_number", 0) or 0)
            for race in v_races:
                output.append(format_race_prediction(race))
                output.append("")

        total_races += len(races)

    output.append("=" * 60)
    output.append(f"  合計: {total_races}レース")
    output.append("  ※ 競走得点・勝率ベースのヒューリスティック予測")
    output.append("=" * 60)

    return "\n".join(output)


def generate_demo_data(dates: list[str]) -> dict[str, list[dict]]:
    """デモ用サンプルデータを生成（外部アクセス不可時）"""
    random.seed(42)

    # 実在の選手名・府県・級班を模したサンプルデータ
    sample_riders = [
        {"rider_name": "松浦悠士", "prefecture": "広島", "class": "SS", "base_score": 117.5},
        {"rider_name": "脇本雄太", "prefecture": "福井", "class": "SS", "base_score": 119.2},
        {"rider_name": "古性優作", "prefecture": "大阪", "class": "SS", "base_score": 116.8},
        {"rider_name": "郡司浩平", "prefecture": "神奈川", "class": "SS", "base_score": 115.3},
        {"rider_name": "清水裕友", "prefecture": "山口", "class": "SS", "base_score": 114.9},
        {"rider_name": "新山響平", "prefecture": "青森", "class": "SS", "base_score": 113.7},
        {"rider_name": "平原康多", "prefecture": "埼玉", "class": "SS", "base_score": 112.4},
        {"rider_name": "佐藤慎太郎", "prefecture": "福島", "class": "SS", "base_score": 111.8},
        {"rider_name": "守澤太志", "prefecture": "秋田", "class": "SS", "base_score": 111.2},
        {"rider_name": "吉田拓矢", "prefecture": "茨城", "class": "S1", "base_score": 110.5},
        {"rider_name": "宿口陽一", "prefecture": "埼玉", "class": "S1", "base_score": 109.8},
        {"rider_name": "和田健太郎", "prefecture": "千葉", "class": "S1", "base_score": 109.1},
        {"rider_name": "山口拳矢", "prefecture": "岐阜", "class": "S1", "base_score": 108.3},
        {"rider_name": "嘉永泰斗", "prefecture": "熊本", "class": "S1", "base_score": 107.6},
        {"rider_name": "眞杉匠", "prefecture": "栃木", "class": "S1", "base_score": 107.0},
        {"rider_name": "坂井洋", "prefecture": "栃木", "class": "S1", "base_score": 106.5},
        {"rider_name": "小林優香", "prefecture": "福岡", "class": "S1", "base_score": 105.8},
        {"rider_name": "犬伏湧也", "prefecture": "徳島", "class": "S1", "base_score": 105.2},
        {"rider_name": "太田竜馬", "prefecture": "徳島", "class": "S1", "base_score": 104.7},
        {"rider_name": "北井佑季", "prefecture": "神奈川", "class": "S1", "base_score": 104.1},
        {"rider_name": "渡邉雄太", "prefecture": "静岡", "class": "S2", "base_score": 101.3},
        {"rider_name": "小原太樹", "prefecture": "神奈川", "class": "S2", "base_score": 100.8},
        {"rider_name": "町田太我", "prefecture": "広島", "class": "S2", "base_score": 100.2},
        {"rider_name": "山田庸平", "prefecture": "佐賀", "class": "S2", "base_score": 99.5},
        {"rider_name": "菊池岳仁", "prefecture": "長野", "class": "S2", "base_score": 98.7},
        {"rider_name": "河端朋之", "prefecture": "岡山", "class": "S2", "base_score": 98.0},
        {"rider_name": "松岡健介", "prefecture": "兵庫", "class": "S2", "base_score": 97.3},
        {"rider_name": "島川将貴", "prefecture": "徳島", "class": "A1", "base_score": 93.5},
        {"rider_name": "高橋晋也", "prefecture": "福島", "class": "A1", "base_score": 92.8},
        {"rider_name": "田中誠", "prefecture": "福岡", "class": "A1", "base_score": 91.5},
        {"rider_name": "鈴木裕也", "prefecture": "千葉", "class": "A1", "base_score": 90.2},
        {"rider_name": "伊藤颯馬", "prefecture": "沖縄", "class": "A1", "base_score": 89.5},
        {"rider_name": "佐々木悠葵", "prefecture": "群馬", "class": "A2", "base_score": 85.3},
        {"rider_name": "中村一将", "prefecture": "三重", "class": "A2", "base_score": 84.0},
        {"rider_name": "木村弘", "prefecture": "青森", "class": "A2", "base_score": 82.5},
    ]

    # 週間の開催パターン（主要場を日別に配置）
    velodrome_schedule = {
        0: [("25", "大宮"), ("41", "名古屋"), ("74", "松山")],        # 月
        1: [("25", "大宮"), ("41", "名古屋"), ("74", "松山")],        # 火
        2: [("25", "大宮"), ("41", "名古屋"), ("74", "松山")],        # 水
        3: [("28", "立川"), ("34", "平塚"), ("81", "小倉")],          # 木
        4: [("28", "立川"), ("34", "平塚"), ("81", "小倉")],          # 金
        5: [("28", "立川"), ("22", "前橋"), ("55", "岸和田")],        # 土
        6: [("28", "立川"), ("22", "前橋"), ("55", "岸和田")],        # 日
    }

    weekly = {}
    for date in dates:
        d = datetime.strptime(date, "%Y%m%d")
        weekday = d.weekday()
        velodromes = velodrome_schedule.get(weekday, [])
        day_races = []

        for jyo_cd, velodrome_name in velodromes:
            num_races = random.randint(7, 12)
            for race_num in range(1, num_races + 1):
                # 9人のレース
                rider_count = 9
                race_riders = random.sample(sample_riders, rider_count)

                entries = []
                for bike_num, rider in enumerate(race_riders, 1):
                    noise = random.gauss(0, 3)
                    comp_score = rider["base_score"] + noise
                    win_rate = max(0, min(50, (comp_score - 80) * 0.8 + random.gauss(0, 5)))
                    top3_rate = max(0, min(80, win_rate * 2.2 + random.gauss(0, 5)))

                    entry = {
                        "rider_name": rider["rider_name"],
                        "prefecture": rider["prefecture"],
                        "class": rider["class"],
                        "bike_number": bike_num,
                        "frame_number": bike_num,
                        "avg_competition_score": round(comp_score, 1),
                        "win_rate": round(win_rate, 1),
                        "top3_rate": round(top3_rate, 1),
                        "gear_ratio": round(random.uniform(3.57, 3.93), 2),
                    }
                    entry["pred_score"] = heuristic_score(entry)
                    entries.append(entry)

                entries.sort(key=lambda e: e["pred_score"], reverse=True)
                for i, e in enumerate(entries):
                    e["mark"] = MARKS[i] if i < len(MARKS) else ""
                    e["predicted_rank"] = i + 1

                grade = "F1" if race_num <= 6 else "F2"
                race_name_parts = ["予選", "一次予選", "二次予選", "準決勝", "特選", "選抜"]
                race_name = random.choice(race_name_parts)

                race_id = f"{date}{jyo_cd}{race_num:02d}"
                day_races.append({
                    "race_id": race_id,
                    "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                    "velodrome": velodrome_name,
                    "race_number": race_num,
                    "race_name": race_name,
                    "grade": grade,
                    "entries": entries,
                })

        weekly[date] = day_races

    return weekly


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="競輪週間予測")
    parser.add_argument("--date", help="基準日 (YYYYMMDD), デフォルト=今日")
    parser.add_argument("--demo", action="store_true",
                        help="デモモード（サンプルデータで予測生成）")
    args = parser.parse_args()

    dates = get_week_dates(args.date)
    logger.info("予測期間: %s 〜 %s", dates[0], dates[-1])

    use_demo = args.demo
    if not use_demo:
        # スクレイピングを試みる
        try:
            from data.scraper import KeirinScraper
            scraper = KeirinScraper()
            # 接続テスト
            test_ids = scraper.scrape_race_list(dates[0])
        except Exception as e:
            logger.warning("外部接続エラー: %s", e)
            logger.info("デモモードに切り替えます")
            use_demo = True

    if use_demo:
        weekly = generate_demo_data(dates)
    else:
        weekly = {}
        for date in dates:
            logger.info("Processing %s ...", date)
            predictions = predict_day(scraper, date)
            weekly[date] = predictions

    report = format_weekly_report(weekly)
    print(report)

    # ファイル出力
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    week_start = dates[0]
    output_path = RESULTS_DIR / f"weekly_forecast_{week_start}.txt"
    output_path.write_text(report, encoding="utf-8")
    logger.info("レポート出力: %s", output_path)


if __name__ == "__main__":
    main()
