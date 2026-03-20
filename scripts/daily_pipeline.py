"""日次自動パイプライン

朝(--morning): 予測 → Supabase同期
夕(--evening): 結果取得 → 精度評価 → Supabase同期

Usage:
    python scripts/daily_pipeline.py --morning
    python scripts/daily_pipeline.py --evening
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY, RESULTS_DIR, VELODROME_CODES

LOG_DIR = RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MARKS = ["◎", "○", "▲", "△", "△"]


def setup_logging(mode: str, date_str: str):
    """ファイル+コンソールの両方にログ出力"""
    log_file = LOG_DIR / f"daily_{date_str}_{mode}.log"
    handlers = [
        logging.FileHandler(str(log_file), encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_file


def create_supabase_client():
    """Supabaseクライアントを作成。未設定ならNone。"""
    supabase_key = SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    if not SUPABASE_URL or not supabase_key:
        logging.warning("Supabase未設定 — 同期スキップ")
        return None
    from supabase import create_client
    return create_client(SUPABASE_URL, supabase_key)


def run_morning(date_str: str):
    """朝パイプライン: 予測 → Supabase同期"""
    logger = logging.getLogger(__name__)
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    # 1. 予測実行
    logger.info("=== 朝パイプライン開始: %s ===", date_str)
    logger.info("[Step 1] レース予測...")
    from scripts.predict_races import predict_races
    predict_races(date_str, velodromes="auto")
    logger.info("予測完了")

    # 2. Supabase同期
    logger.info("[Step 2] Supabase同期...")
    client = create_supabase_client()
    if client:
        from scripts.sync_to_supabase import sync_since, get_sqlite_conn
        conn = get_sqlite_conn()
        try:
            sync_since(client, conn, formatted_date)
            logger.info("Supabase同期完了")
        finally:
            conn.close()

    logger.info("=== 朝パイプライン完了 ===")


def generate_detail_report(conn, formatted_date: str, date_str: str) -> str:
    """予測 vs 結果の詳細比較レポートを生成しファイルに保存"""
    race_rows = conn.execute("""
        SELECT DISTINCT r.race_id, r.velodrome, r.race_number, r.grade
        FROM predictions p
        JOIN races r ON p.race_id = r.race_id
        WHERE r.date = ?
        ORDER BY r.race_id
    """, (formatted_date,)).fetchall()

    if not race_rows:
        return ""

    lines = []
    lines.append("=" * 60)
    lines.append(f"  予測 vs 結果 詳細レポート: {formatted_date}")
    lines.append("=" * 60)

    total_races = 0
    top1_hits = 0
    top3_total = 0
    top3_hits = 0

    for race in race_rows:
        race_id = race["race_id"]
        velodrome = race["velodrome"] or "?"
        race_num = race["race_number"] or "?"
        grade = race["grade"] or ""

        preds = conn.execute("""
            SELECT p.rider_id, p.predicted_rank, p.predicted_score, p.mark,
                   e.bike_number, e.rider_name, e.class
            FROM predictions p
            LEFT JOIN entries e ON p.race_id = e.race_id AND p.rider_id = e.rider_id
            WHERE p.race_id = ?
            ORDER BY p.predicted_rank
        """, (race_id,)).fetchall()

        results = conn.execute("""
            SELECT rider_id, finish_position, odds
            FROM race_results WHERE race_id = ? AND finish_position IS NOT NULL
            ORDER BY finish_position
        """, (race_id,)).fetchall()

        if not preds or not results:
            continue

        total_races += 1
        result_map = {r["rider_id"]: r for r in results}
        actual_top3 = set(r["rider_id"] for r in results[:3])

        grade_label = f" [{grade}]" if grade else ""
        lines.append(f"\n── {velodrome} {race_num}R{grade_label} " + "─" * (40 - len(velodrome) - len(str(race_num)) - len(grade_label)))
        lines.append(f"  {'予測':>4}  {'車番':>4}  {'選手名':<8}  {'級班':>4}  {'スコア':>8}  │ {'着順':>4}  {'判定'}")
        lines.append(f"  {'----':>4}  {'----':>4}  {'--------':<8}  {'----':>4}  {'--------':>8}  │ {'----':>4}  {'----'}")

        race_top1_hit = False
        race_top3_hits = 0

        for pred in preds:
            rider_id = pred["rider_id"]
            rank = pred["predicted_rank"]
            mark = pred["mark"] or "  "
            bike = pred["bike_number"] or "?"
            name = (pred["rider_name"] or "")[:8]
            cls = pred["class"] or ""
            score = pred["predicted_score"] or 0

            res = result_map.get(rider_id)
            if res:
                finish = res["finish_position"]
                # 判定
                if rank == 1 and finish == 1:
                    judge = "◎的中!"
                    race_top1_hit = True
                elif rank <= 3 and finish <= 3:
                    judge = "○Top3"
                    race_top3_hits += 1
                elif finish <= 3:
                    judge = "△見逃"
                else:
                    judge = ""
                finish_str = f"{finish}着"
            else:
                finish_str = "---"
                judge = ""

            lines.append(
                f"  {mark:>2}{rank:>2}位  {bike:>4}番  {name:<8}  {cls:>4}  {score:>8.3f}  │ {finish_str:>4}  {judge}"
            )

        if race_top1_hit:
            top1_hits += 1
        # Top3: 予測Top3のうち実際にTop3に入った数
        pred_top3_riders = set(p["rider_id"] for p in preds[:3])
        t3 = len(pred_top3_riders & actual_top3)
        top3_hits += t3
        top3_total += min(3, len(actual_top3))

        hit_mark = "✓" if race_top1_hit else "✗"
        lines.append(f"  → ◎{hit_mark}  Top3一致: {t3}/3")

    # サマリー
    lines.append(f"\n{'=' * 60}")
    lines.append(f"  サマリー ({formatted_date})")
    lines.append(f"{'=' * 60}")
    lines.append(f"  対象レース数: {total_races}")
    if total_races > 0:
        lines.append(f"  ◎的中率: {top1_hits}/{total_races} ({top1_hits / total_races * 100:.1f}%)")
        lines.append(f"  Top3一致率: {top3_hits}/{top3_total} ({top3_hits / top3_total * 100:.1f}%)" if top3_total > 0 else "")
    lines.append("=" * 60)

    report = "\n".join(lines)

    # ファイルに保存
    report_file = LOG_DIR / f"daily_{date_str}_report.txt"
    report_file.write_text(report, encoding="utf-8")

    return str(report_file)


def run_evening(date_str: str):
    """夕パイプライン: 結果取得 → 精度評価 → Supabase同期"""
    logger = logging.getLogger(__name__)
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    logger.info("=== 夕パイプライン開始: %s ===", date_str)

    # 1. 結果取得
    logger.info("[Step 1] 結果取得...")
    from data.scraper import KeirinScraper
    from db.schema import get_connection, insert_result

    scraper = KeirinScraper()
    conn = get_connection()

    pred_races = conn.execute("""
        SELECT DISTINCT p.race_id
        FROM predictions p
        JOIN races r ON p.race_id = r.race_id
        WHERE r.date = ?
    """, (formatted_date,)).fetchall()

    if not pred_races:
        logger.info("予測データなし — 結果取得スキップ")
        conn.close()
        logger.info("=== 夕パイプライン完了（対象なし） ===")
        return

    fetched = 0
    for pr in pred_races:
        race_id = pr["race_id"] if hasattr(pr, "__getitem__") else pr[0]
        existing = conn.execute(
            "SELECT COUNT(*) FROM race_results WHERE race_id = ?",
            (race_id,)
        ).fetchone()[0]
        if existing > 0:
            continue
        try:
            result_data = scraper.scrape_race_result(race_id)
            if result_data:
                for rr in result_data.get("results", []):
                    rr["race_id"] = race_id
                    insert_result(conn, rr)
                conn.commit()
                fetched += 1
        except Exception as e:
            logger.warning("結果取得エラー %s: %s", race_id, e)

    logger.info("結果取得完了: %d/%dレース新規取得", fetched, len(pred_races))

    # 2. 精度評価
    logger.info("[Step 2] 精度評価...")
    from backtest.evaluator import evaluate_predictions
    stats = evaluate_predictions(formatted_date)

    if stats and stats.get("total_races", 0) > 0:
        top1_pct = stats["top1_hits"] / stats["total_races"] * 100
        logger.info(
            "精度: ◎的中率=%.1f%% (%d/%dR)",
            top1_pct, stats["top1_hits"], stats["total_races"]
        )

    # 3. 詳細比較レポート生成
    logger.info("[Step 3] 詳細比較レポート生成...")
    report_file = generate_detail_report(conn, formatted_date, date_str)
    if report_file:
        logger.info("レポート保存: %s", report_file)

    conn.close()

    # 4. Supabase同期
    logger.info("[Step 4] Supabase同期...")
    client = create_supabase_client()
    if client:
        from scripts.sync_to_supabase import sync_since, get_sqlite_conn
        sync_conn = get_sqlite_conn()
        try:
            sync_since(client, sync_conn, formatted_date)
            logger.info("Supabase同期完了")
        finally:
            sync_conn.close()

    logger.info("=== 夕パイプライン完了 ===")


def main():
    parser = argparse.ArgumentParser(description="競輪AI 日次パイプライン")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--morning", action="store_true", help="朝パイプライン（予測+同期）")
    group.add_argument("--evening", action="store_true", help="夕パイプライン（結果+評価+同期）")
    parser.add_argument("--date", help="対象日 (YYYYMMDD) ※省略時は今日")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    mode = "morning" if args.morning else "evening"

    log_file = setup_logging(mode, date_str)
    logger = logging.getLogger(__name__)
    logger.info("ログファイル: %s", log_file)

    try:
        if args.morning:
            run_morning(date_str)
        else:
            run_evening(date_str)
    except Exception:
        logger.exception("パイプライン異常終了")
        sys.exit(1)


if __name__ == "__main__":
    main()
