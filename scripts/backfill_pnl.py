"""過去レースのバックフィル（エントリー・結果・配当・予測）

2026-01-01以降の全レースについて:
1. レース発見（DB既存 or スクレイピング）
2. 出走表取得 → DB保存
3. 結果・配当取得 → DB保存
4. 予測実行 → DB保存
5. Supabase同期

Usage:
    python scripts/backfill_pnl.py                        # 全期間
    python scripts/backfill_pnl.py --from 20260201        # 2/1以降のみ
    python scripts/backfill_pnl.py --from 20260101 --to 20260110  # 範囲指定
    python scripts/backfill_pnl.py --sync-only            # Supabase同期のみ
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DB_PATH, VELODROME_CODES, RESULTS_DIR, MARKS,
    SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY,
)
from db.schema import (
    get_connection, insert_race, insert_rider,
    insert_entry, insert_result, insert_payout, insert_prediction,
)
from data.scraper import KeirinScraper
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from scripts.predict_races import apply_line_bonus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def get_date_range(from_date: str, to_date: str) -> list[str]:
    """YYYYMMDD形式の日付リストを生成"""
    start = datetime.strptime(from_date, "%Y%m%d")
    end = datetime.strptime(to_date, "%Y%m%d")
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return dates


def discover_race_ids(conn, scraper, date: str) -> list[str]:
    """指定日のrace_idを取得（DB優先、なければスクレイピング）"""
    formatted = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

    # DB既存チェック
    existing = conn.execute(
        "SELECT race_id FROM races WHERE date = ?", (formatted,)
    ).fetchall()
    if existing:
        return [r[0] for r in existing]

    # スクレイピングで発見（主要12場のみ = 実運用と同じ）
    from scripts.predict_races import get_race_ids_for_date, MAJOR_CODES
    race_ids = get_race_ids_for_date(scraper, date, MAJOR_CODES)
    return race_ids


def ensure_entries(conn, scraper, race_id: str) -> dict | None:
    """出走表を確保（DB or スクレイピング）。成功時にentry_dataを返す。"""
    existing = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE race_id = ?", (race_id,)
    ).fetchone()[0]

    if existing > 0:
        # DBにあるのでJSONキャッシュから取得してline_formationを返す
        cached = scraper._get_json_cache(f"race_entry_{race_id}")
        return cached  # Noneでも大丈夫

    try:
        entry_data = scraper.scrape_race_entry(race_id)
        entries = entry_data.get("entries", [])
        if not entries:
            return None

        formatted = f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"
        race_data = {
            "race_id": race_id,
            "date": formatted,
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
        return entry_data
    except Exception as e:
        logger.debug("Entry fetch failed for %s: %s", race_id, e)
        return None


def ensure_results_and_payouts(conn, scraper, race_id: str):
    """結果と配当を確保"""
    has_results = conn.execute(
        "SELECT COUNT(*) FROM race_results WHERE race_id = ?", (race_id,)
    ).fetchone()[0]
    has_payout = conn.execute(
        "SELECT COUNT(*) FROM race_payouts WHERE race_id = ?", (race_id,)
    ).fetchone()[0]

    if has_results > 0 and has_payout > 0:
        return True

    try:
        # 既存JSONキャッシュにpayoutsがない場合はクリアして再取得
        from config import CACHE_DIR
        cache_key = f"race_result_{race_id}"
        cached = scraper._get_json_cache(cache_key)
        if cached and not cached.get("payouts"):
            json_path = CACHE_DIR / f"{cache_key}.json"
            if json_path.exists():
                json_path.unlink()

        result_data = scraper.scrape_race_result(race_id)
        if not result_data:
            return False

        # 結果保存
        if has_results == 0:
            formatted = f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"
            race_data = {
                "race_id": race_id,
                "date": formatted,
                "velodrome": result_data.get("velodrome", ""),
                "race_number": result_data.get("race_number"),
                "race_name": result_data.get("race_name"),
                "grade": result_data.get("grade"),
                "rider_count": result_data.get("rider_count"),
            }
            insert_race(conn, race_data)

            for rr in result_data.get("results", []):
                rr["race_id"] = race_id
                if rr.get("rider_id"):
                    insert_rider(conn, {
                        "rider_id": rr["rider_id"],
                        "name": rr.get("rider_name", ""),
                        "class": rr.get("class"),
                        "prefecture": rr.get("prefecture"),
                    })
                insert_result(conn, rr)

        # 配当保存
        if has_payout == 0:
            payouts = result_data.get("payouts", {})
            if payouts.get("nisyatan_payout"):
                payouts["race_id"] = race_id
                insert_payout(conn, payouts)

        conn.commit()
        return True
    except Exception as e:
        logger.debug("Result/payout fetch failed for %s: %s", race_id, e)
        return False


def run_prediction(conn, builder, model, race_id: str, entry_data: dict | None):
    """予測実行（未実行の場合のみ）"""
    existing = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE race_id = ?", (race_id,)
    ).fetchone()[0]
    if existing > 0:
        return True

    try:
        formatted = f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"
        df = builder.build_race_features(race_id, formatted)
        if df.empty:
            return False

        feature_cols = builder.feature_names
        X = df[feature_cols].fillna(0)
        df["pred_score"] = model.predict(X).values

        # ライン補正
        if entry_data:
            line_formation = entry_data.get("line_formation", [])
            entries = entry_data.get("entries", [])
            if line_formation and entries:
                df["line_bonus"] = apply_line_bonus(df, entries, line_formation)
                df["pred_score"] = df["pred_score"] + df["line_bonus"]

        df = df.sort_values("pred_score", ascending=False).reset_index(drop=True)

        score_gap = 0.0
        if len(df) >= 2:
            score_gap = df.iloc[0]["pred_score"] - df.iloc[1]["pred_score"]

        for i, row in df.iterrows():
            rank = i + 1
            mark = MARKS[i] if i < len(MARKS) else ""
            insert_prediction(conn, {
                "race_id": race_id,
                "rider_id": row["rider_id"],
                "predicted_score": row["pred_score"],
                "predicted_rank": rank,
                "mark": mark,
                "confidence": score_gap,
            })
        conn.commit()
        return True
    except Exception as e:
        logger.debug("Prediction failed for %s: %s", race_id, e)
        return False


def sync_to_supabase():
    """全データをSupabaseに同期"""
    supabase_key = SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    if not SUPABASE_URL or not supabase_key:
        logger.warning("Supabase未設定 — 同期スキップ")
        return

    from supabase import create_client
    from scripts.sync_to_supabase import sync_since, get_sqlite_conn

    client = create_client(SUPABASE_URL, supabase_key)
    conn = get_sqlite_conn()
    try:
        logger.info("Supabase同期中...")
        sync_since(client, conn, "2026-01-01")
        logger.info("Supabase同期完了")
    except Exception as e:
        logger.warning("Supabase同期エラー: %s", e)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="過去レースバックフィル")
    parser.add_argument("--from", dest="from_date", default="20260101",
                        help="開始日 (YYYYMMDD)")
    parser.add_argument("--to", dest="to_date", default=None,
                        help="終了日 (YYYYMMDD) ※省略時は昨日")
    parser.add_argument("--sync-only", action="store_true",
                        help="Supabase同期のみ")
    parser.add_argument("--no-sync", action="store_true",
                        help="Supabase同期をスキップ")
    args = parser.parse_args()

    if args.sync_only:
        sync_to_supabase()
        return

    if not args.to_date:
        yesterday = datetime.now() - timedelta(days=1)
        args.to_date = yesterday.strftime("%Y%m%d")

    dates = get_date_range(args.from_date, args.to_date)
    logger.info("バックフィル開始: %s 〜 %s (%d日間)", args.from_date, args.to_date, len(dates))

    # モデルロード
    model_path = str(RESULTS_DIR / "model_lgbm.pkl")
    model = LGBMRanker()
    model.load(model_path)
    builder = FeatureBuilder()
    scraper = KeirinScraper()
    conn = get_connection()

    total_races = 0
    total_entries = 0
    total_results = 0
    total_payouts = 0
    total_predictions = 0
    start_time = time.time()

    for date_idx, date in enumerate(dates):
        formatted = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

        # レースID取得
        race_ids = discover_race_ids(conn, scraper, date)
        if not race_ids:
            continue

        day_entries = 0
        day_results = 0
        day_payouts = 0
        day_preds = 0

        for race_id in race_ids:
            total_races += 1

            # 1. 出走表確保
            entry_data = ensure_entries(conn, scraper, race_id)
            if entry_data:
                day_entries += 1

            # 2. 結果・配当確保
            if ensure_results_and_payouts(conn, scraper, race_id):
                day_results += 1
                has_pay = conn.execute(
                    "SELECT COUNT(*) FROM race_payouts WHERE race_id = ?", (race_id,)
                ).fetchone()[0]
                if has_pay > 0:
                    day_payouts += 1

            # 3. 予測実行
            if entry_data and run_prediction(conn, builder, model, race_id, entry_data):
                day_preds += 1

        total_entries += day_entries
        total_results += day_results
        total_payouts += day_payouts
        total_predictions += day_preds

        elapsed = time.time() - start_time
        remaining_dates = len(dates) - date_idx - 1
        if date_idx > 0:
            avg_per_date = elapsed / (date_idx + 1)
            eta_min = remaining_dates * avg_per_date / 60
        else:
            eta_min = 0

        if len(race_ids) > 0:
            logger.info(
                "[%d/%d] %s: %dR (entry=%d result=%d payout=%d pred=%d) ETA=%.0fm",
                date_idx + 1, len(dates), formatted, len(race_ids),
                day_entries, day_results, day_payouts, day_preds, eta_min,
            )

    elapsed_total = (time.time() - start_time) / 60
    logger.info("=" * 60)
    logger.info("バックフィル完了 (%.1f分)", elapsed_total)
    logger.info("  レース数: %d", total_races)
    logger.info("  出走表: %d", total_entries)
    logger.info("  結果: %d", total_results)
    logger.info("  配当: %d", total_payouts)
    logger.info("  予測: %d", total_predictions)
    logger.info("=" * 60)

    conn.close()

    # Supabase同期
    if not args.no_sync:
        sync_to_supabase()


if __name__ == "__main__":
    main()
