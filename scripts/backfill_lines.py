"""キャッシュHTMLからライン情報をDB復元

data/cache/keirin_netkeiba_com_db_result__race_id_*.html を全件パースし、
race_results の line_group / line_role を UPDATE する。
ネットワーク不要（ローカルI/Oのみ）。
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup
from config import CACHE_DIR
from db.schema import get_connection
from data.scraper import KeirinScraper

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # キャッシュHTMLを全件取得
    html_files = sorted(CACHE_DIR.glob("keirin_netkeiba_com_db_result__race_id_*.html"))
    logger.info("Found %d cached HTML files", len(html_files))

    if not html_files:
        logger.warning("No cached HTML files found in %s", CACHE_DIR)
        return

    conn = get_connection()
    updated_races = 0
    updated_rows = 0
    skipped = 0

    for i, html_file in enumerate(html_files):
        # race_id をファイル名から抽出
        # ファイル名: keirin_netkeiba_com_db_result__race_id_XXXXXXXXXXXX.html
        name = html_file.stem  # 拡張子なし
        parts = name.split("race_id_")
        if len(parts) < 2:
            continue
        race_id = parts[1]
        if not race_id.isdigit() or len(race_id) != 12:
            continue

        # HTMLをパース
        try:
            html = html_file.read_text(encoding="utf-8")
            soup = BeautifulSoup(html, "lxml")
        except Exception as e:
            logger.warning("Failed to parse %s: %s", html_file.name, e)
            continue

        # ライン情報をパース
        line_data = KeirinScraper._parse_line_formation(soup)
        if not line_data:
            skipped += 1
            continue

        # bike_number → (line_group, line_role) マッピング構築
        bike_to_line = {}
        for line_group_idx, members in enumerate(line_data, 1):
            line_group = str(line_group_idx)
            for pos, member in enumerate(members):
                bike_num = member["bike_number"]
                if pos == 0:
                    role = "自力"
                elif pos == 1:
                    role = "番手"
                else:
                    role = "3番手"
                bike_to_line[bike_num] = (line_group, role)

        # race_results を UPDATE
        for bike_num, (line_group, line_role) in bike_to_line.items():
            cur = conn.execute(
                """UPDATE race_results
                   SET line_group = ?, line_role = ?
                   WHERE race_id = ? AND bike_number = ?""",
                (line_group, line_role, race_id, bike_num),
            )
            updated_rows += cur.rowcount

        if bike_to_line:
            updated_races += 1

        # 500レースごとにcommit + 進捗ログ
        if (i + 1) % 500 == 0:
            conn.commit()
            logger.info(
                "Progress: %d/%d files, %d races updated, %d rows updated, %d skipped",
                i + 1, len(html_files), updated_races, updated_rows, skipped,
            )

    conn.commit()
    conn.close()

    logger.info(
        "Complete: %d/%d files processed, %d races updated, %d rows updated, %d skipped (no DeployYoso)",
        len(html_files), len(html_files), updated_races, updated_rows, skipped,
    )

    # 検証クエリ
    conn = get_connection()
    total = conn.execute("SELECT count(*) FROM race_results").fetchone()[0]
    with_line = conn.execute(
        "SELECT count(*) FROM race_results WHERE line_group IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    logger.info("Verification: %d/%d results now have line_group (%.1f%%)",
                with_line, total, with_line / total * 100 if total else 0)


if __name__ == "__main__":
    main()
