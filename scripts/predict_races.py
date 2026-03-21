"""レース予測スクリプト

指定日のレースを取得し、学習済みモデルで着順予測を行う。
race_programページから効率的にレースIDを取得。
"""

import sys
import logging
import argparse
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    NETKEIRIN_BASE_URL, VELODROME_CODES, RESULTS_DIR, CLASS_MAP,
    MARKS, LINE_BONUS, STRONGEST_LINE_BONUS, get_bet_category,
)
from db.schema import get_connection, insert_race, insert_rider, insert_entry, insert_prediction
from data.scraper import KeirinScraper
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker

logger = logging.getLogger(__name__)

MAJOR_CODES = ["22", "25", "27", "28", "31", "34", "35", "42", "54", "56", "75", "81"]


def apply_line_bonus(df, entries, line_formation):
    """ライン構成に基づくボーナススコアを計算"""
    # 車番→rider_idマッピング
    bike_to_rider = {e["bike_number"]: e["rider_id"] for e in entries if e.get("bike_number") and e.get("rider_id")}
    # rider_id→class
    rider_class = {e["rider_id"]: e.get("class", "") for e in entries if e.get("rider_id")}

    # 各ラインの強度スコアを計算
    line_scores = []
    for line in line_formation:
        score = 0
        for member in line:
            bn = member["bike_number"]
            rid = bike_to_rider.get(bn, "")
            cls = rider_class.get(rid, "")
            cls_num = CLASS_MAP.get(cls, 6)
            score += (7 - cls_num)
        line_scores.append(score)

    strongest_idx = line_scores.index(max(line_scores)) if line_scores else -1

    # rider_id → (ラインサイズ, 役割, 最強ライン所属)
    rider_line_info = {}
    for line_idx, line in enumerate(line_formation):
        line_size = len(line)
        for pos, member in enumerate(line):
            bn = member["bike_number"]
            rid = bike_to_rider.get(bn, "")
            if not rid:
                continue
            if pos == 0:
                role = "自力"
            elif pos == 1:
                role = "番手"
            else:
                role = "3番手"
            is_strongest = (line_idx == strongest_idx)
            rider_line_info[rid] = (line_size, role, is_strongest)

    bonuses = []
    for _, row in df.iterrows():
        rid = row["rider_id"]
        info = rider_line_info.get(rid)
        if info is None:
            bonuses.append(0.0)
            continue
        line_size, role, is_strongest = info
        bonus = LINE_BONUS.get((line_size, role), 0.0)
        if is_strongest:
            bonus += STRONGEST_LINE_BONUS
        bonuses.append(bonus)

    return bonuses


def get_today_jyo_codes(scraper: KeirinScraper) -> list[str]:
    """netkeirinトップページから本日開催中の場コードを自動取得"""
    url = f"{NETKEIRIN_BASE_URL}/top/"
    html = scraper._get(url)
    codes = re.findall(r"jyo_cd=(\d+)&rf=toptodayrace", html)
    result = sorted(set(codes))
    if result:
        names = [VELODROME_CODES.get(c, f"?({c})") for c in result]
        logger.info("Auto-detected %d velodromes: %s", len(result), ", ".join(names))
    return result


def get_race_ids_for_date(scraper: KeirinScraper, date: str, jyo_codes: list[str]) -> list[str]:
    """race_programページから指定日の全race_idを効率取得

    kaisai_group_idは開催初日+場コードなので、2日目以降は前日以前の
    kaisai_group_idからしか取得できない。最大4日前まで遡って検索する。
    race_programで見つからない場合はapi_race_listをフォールバックとして使用。
    """
    from datetime import datetime, timedelta

    dt = datetime.strptime(date, "%Y%m%d")
    # 当日を含め最大5日前まで遡る（4日間開催が最長）
    check_dates = [(dt - timedelta(days=d)).strftime("%Y%m%d") for d in range(5)]

    all_ids = []
    for jyo_cd in jyo_codes:
        found = False
        for check_date in check_dates:
            kaisai_group_id = f"{check_date}{jyo_cd}"
            cache_key = f"race_program_{kaisai_group_id}"
            cached = scraper._get_json_cache(cache_key)
            if cached is not None:
                # 対象日のrace_idのみ抽出
                matching = [rid for rid in cached if rid.startswith(date)]
                all_ids.extend(matching)
                if matching:
                    found = True
                    break  # この場コードはヒットしたので次の場へ
                continue

            url = f"{NETKEIRIN_BASE_URL}/db/race_program/?kaisai_group_id={kaisai_group_id}"
            html = scraper._get(url)
            race_ids = sorted(set(re.findall(r"race_id=(\d{12})", html)))
            scraper._set_json_cache(cache_key, race_ids)
            # 対象日のrace_idのみ抽出
            matching = [rid for rid in race_ids if rid.startswith(date)]
            all_ids.extend(matching)
            if matching:
                found = True
                break  # この場コードはヒットしたので次の場へ

        # フォールバック: api_race_listで直接検索
        if not found:
            cache_key = f"api_race_list_{date}_{jyo_cd}"
            cached = scraper._get_json_cache(cache_key)
            if cached is not None:
                all_ids.extend(cached)
            else:
                url = f"{NETKEIRIN_BASE_URL}/race/api_race_list.html?kaisai_date={date}&jyocd={jyo_cd}"
                html = scraper._get(url)
                race_ids = sorted(set(re.findall(r"race_id=(\d{12})", html)))
                scraper._set_json_cache(cache_key, race_ids)
                all_ids.extend(race_ids)
                if race_ids:
                    logger.info("Found %d races via api_race_list for jyo_cd=%s", len(race_ids), jyo_cd)

    return sorted(set(all_ids))


def predict_races(date: str, velodromes: str = "major"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_path = str(RESULTS_DIR / "model_lgbm.pkl")
    model = LGBMRanker()
    model.load(model_path)
    logger.info("Model loaded")

    scraper = KeirinScraper()
    builder = FeatureBuilder()
    conn = get_connection()

    # 場コード選択
    if velodromes == "auto":
        jyo_codes = get_today_jyo_codes(scraper)
        if not jyo_codes:
            logger.warning("Auto-detect failed, falling back to all")
            jyo_codes = list(VELODROME_CODES.keys())
    elif velodromes == "all":
        jyo_codes = list(VELODROME_CODES.keys())
    elif velodromes == "major":
        jyo_codes = MAJOR_CODES
    else:
        jyo_codes = [c.strip() for c in velodromes.split(",")]

    race_ids = get_race_ids_for_date(scraper, date, jyo_codes)
    logger.info("%s: %d races found", date, len(race_ids))

    if not race_ids:
        logger.info("No races found for %s", date)
        conn.close()
        return

    all_predictions = []

    for race_id in race_ids:
        try:
            # 出走表を取得
            entry_data = scraper.scrape_race_entry(race_id)
            entries = entry_data.get("entries", [])
            if not entries:
                logger.debug("No entries for %s", race_id)
                continue

            # DBに保存
            formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            race_data = {
                "race_id": race_id,
                "date": formatted_date,
                "velodrome": entry_data.get("velodrome", ""),
                "race_number": entry_data.get("race_number"),
                "race_name": entry_data.get("race_name"),
                "grade": entry_data.get("grade"),
                "rider_count": len(entries),
                "start_time": entry_data.get("start_time"),
            }
            insert_race(conn, race_data)

            for e in entries:
                if e.get("rider_id"):
                    insert_rider(conn, {
                        "rider_id": e["rider_id"],
                        "name": e.get("rider_name", ""),
                        "class": e.get("class"),
                        "prefecture": e.get("prefecture"),
                    })
                    insert_entry(conn, e)

            conn.commit()

            # 特徴量構築 & 予測
            df = builder.build_race_features(race_id, formatted_date)
            if df.empty:
                continue

            feature_cols = builder.feature_names
            X = df[feature_cols].fillna(0)
            df["pred_score"] = model.predict(X).values

            # ライン補正スコアを加算
            line_formation = entry_data.get("line_formation", [])
            if line_formation:
                df["line_bonus"] = apply_line_bonus(df, entries, line_formation)
                df["pred_score"] = df["pred_score"] + df["line_bonus"]

            df = df.sort_values("pred_score", ascending=False).reset_index(drop=True)

            # 信頼度（◎-○スコア差）→ 推奨賭け金判定
            score_gap = 0.0
            if len(df) >= 2:
                score_gap = df.iloc[0]["pred_score"] - df.iloc[1]["pred_score"]

            # F2レースは賭け対象外（バックテストで回収率64.5%）
            race_grade = entry_data.get("grade", "")
            is_f2 = (race_grade == "F2")

            gap_label, bet_rec, _ = get_bet_category(score_gap, race_grade)

            velodrome = entry_data.get("velodrome", "?")
            rnum = entry_data.get("race_number", "?")

            # 選手名マップ
            name_map = {e["rider_id"]: e.get("rider_name", "") for e in entries if e.get("rider_id")}
            class_map = {e["rider_id"]: e.get("class", "") for e in entries if e.get("rider_id")}
            bike_map = {e["rider_id"]: e.get("bike_number", "") for e in entries if e.get("rider_id")}

            # 2連単オッズ取得（◎→○,▲,△,△ の4点）
            exacta_odds_map = {}  # rider_id → exacta_odds
            torigami_warn = ""
            if gap_label in ("HIGH", "MED+", "MED"):
                odds_data = scraper.scrape_exacta_odds(race_id)
                if odds_data:
                    honmei_bike = bike_map.get(df.iloc[0]["rider_id"], "")
                    if honmei_bike:
                        honmei_str = str(honmei_bike).zfill(2)
                        combo_odds = []
                        for rank_i in range(1, min(5, len(df))):
                            target_rid = df.iloc[rank_i]["rider_id"]
                            target_bike = str(bike_map.get(target_rid, "")).zfill(2)
                            combo_key = f"{honmei_str}{target_bike}"
                            odds_val = odds_data.get(combo_key)
                            if odds_val is not None:
                                exacta_odds_map[target_rid] = odds_val
                                combo_odds.append(odds_val)
                        # トリガミ判定: 全4点のオッズが4.0未満なら確実に損
                        if combo_odds and max(combo_odds) < 4.0:
                            torigami_warn = " *** トリガミ注意 ***"

            print(f"\n{'='*50}")
            print(f"  {velodrome} {rnum}R")
            print(f"  信頼度: {score_gap:.2f} ({gap_label}) → {bet_rec}{torigami_warn}")
            print(f"{'='*50}")

            for i, row in df.iterrows():
                rank = i + 1
                mark = MARKS[i] if i < len(MARKS) else "  "
                rider_id = row["rider_id"]
                rider_name = name_map.get(rider_id, "")
                rider_class = class_map.get(rider_id, "")
                bike_num = bike_map.get(rider_id, "")

                odds_str = ""
                e_odds = exacta_odds_map.get(rider_id)
                if e_odds is not None:
                    odds_str = f" odds={e_odds:.1f}x"

                print(f"  {mark} {bike_num}番 {rider_name} ({rider_class}) score={row['pred_score']:.3f}{odds_str}")

                insert_prediction(conn, {
                    "race_id": race_id,
                    "rider_id": rider_id,
                    "predicted_score": row["pred_score"],
                    "predicted_rank": rank,
                    "mark": mark,
                    "confidence": score_gap,
                    "exacta_odds": e_odds,
                })

                all_predictions.append({
                    "velodrome": velodrome,
                    "race_number": rnum,
                    "rank": rank,
                    "mark": mark,
                    "bike_number": bike_num,
                    "rider_name": rider_name,
                    "rider_class": rider_class,
                    "score": row["pred_score"],
                })

            conn.commit()

        except Exception as e:
            logger.warning("Error predicting %s: %s", race_id, e)
            continue

    conn.close()
    print(f"\n{'='*50}")
    print(f"  Total: {len(race_ids)} races predicted")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競輪レース予測")
    parser.add_argument("--date", required=True, help="予測日 (YYYYMMDD)")
    parser.add_argument("--velodromes", default="auto",
                        help="'auto'(トップページから自動検出) or 場コード(カンマ区切り) or 'all' or 'major'")
    args = parser.parse_args()
    predict_races(args.date, args.velodromes)
