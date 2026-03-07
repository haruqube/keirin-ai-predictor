"""keirin.jpスクレイパー（キャッシュ付き）"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    CACHE_DIR, KEIRIN_JP_BASE_URL,
    REQUEST_HEADERS, SCRAPE_DELAY, VELODROME_CODES,
)

logger = logging.getLogger(__name__)


class KeirinScraper:
    """keirin.jpからレース情報・結果をスクレイピング"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str) -> str:
        cache_key = re.sub(r"[^a-zA-Z0-9_\-]", "_", url.replace("https://", ""))
        cache_file = CACHE_DIR / f"{cache_key}.html"

        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        time.sleep(SCRAPE_DELAY)
        resp = self.session.get(url, timeout=30)
        resp.encoding = resp.apparent_encoding or "utf-8"
        html = resp.text
        cache_file.write_text(html, encoding="utf-8")
        logger.debug("Fetched %s", url)
        return html

    def _get_json_cache(self, key: str) -> dict | list | None:
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return None

    def _set_json_cache(self, key: str, data):
        cache_file = CACHE_DIR / f"{key}.json"
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def scrape_race_list(self, date: str) -> list[str]:
        """指定日のレースID一覧を取得

        keirin.jpのレース一覧ページから各レースのIDを抽出。
        date: YYYYMMDD形式
        """
        cache_key = f"race_list_{date}"
        cached = self._get_json_cache(cache_key)
        if cached:
            return cached

        # keirin.jp の開催一覧ページ
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        url = f"{KEIRIN_JP_BASE_URL}/race/schedule?date={formatted_date}"
        html = self._get(url)
        soup = BeautifulSoup(html, "lxml")

        race_ids = []
        # レース詳細へのリンクからrace_idを抽出
        for link in soup.select("a[href*='/race/result'], a[href*='/race/detail']"):
            href = link.get("href", "")
            m = re.search(r"race_id=(\d+)", href)
            if m and m.group(1) not in race_ids:
                race_ids.append(m.group(1))

        race_ids = sorted(set(race_ids))
        self._set_json_cache(cache_key, race_ids)
        return race_ids

    def scrape_race_result(self, race_id: str) -> dict:
        """レース結果ページをパース"""
        cache_key = f"race_result_{race_id}"
        cached = self._get_json_cache(cache_key)
        if cached:
            return cached

        url = f"{KEIRIN_JP_BASE_URL}/race/result?race_id={race_id}"
        html = self._get(url)
        soup = BeautifulSoup(html, "lxml")

        race_info = self._parse_race_info(soup, race_id)
        results = self._parse_result_table(soup, race_id)
        race_info["results"] = results
        race_info["rider_count"] = len(results)

        self._set_json_cache(cache_key, race_info)
        return race_info

    def scrape_race_entry(self, race_id: str) -> dict:
        """出走表ページをパース（レース前）"""
        cache_key = f"race_entry_{race_id}"
        cached = self._get_json_cache(cache_key)
        if cached:
            return cached

        url = f"{KEIRIN_JP_BASE_URL}/race/detail?race_id={race_id}"
        html = self._get(url)
        soup = BeautifulSoup(html, "lxml")

        race_info = self._parse_race_info(soup, race_id)
        entries = self._parse_entry_table(soup, race_id)
        race_info["entries"] = entries
        race_info["rider_count"] = len(entries)

        self._set_json_cache(cache_key, race_info)
        return race_info

    def _parse_race_info(self, soup: BeautifulSoup, race_id: str) -> dict:
        """レース情報をパース"""
        info = {"race_id": race_id}

        # レース名
        title = soup.select_one("h3, .race-name, .raceName")
        info["race_name"] = title.get_text(strip=True) if title else ""

        # レース番号
        rnum_el = soup.select_one(".race-number, .raceNumber")
        if rnum_el:
            m = re.search(r"(\d+)", rnum_el.get_text())
            info["race_number"] = int(m.group(1)) if m else None
        else:
            info["race_number"] = None

        # 競輪場
        velodrome_el = soup.select_one(".velodrome, .placeName")
        info["velodrome"] = velodrome_el.get_text(strip=True) if velodrome_el else ""

        # グレード
        info["grade"] = None
        grade_el = soup.select_one("[class*='grade'], [class*='Grade']")
        if grade_el:
            text = grade_el.get_text(strip=True)
            for g in ["GP", "G1", "G2", "G3", "F1", "F2"]:
                if g in text:
                    info["grade"] = g
                    break

        return info

    def _parse_result_table(self, soup: BeautifulSoup, race_id: str) -> list[dict]:
        """レース結果テーブルをパース"""
        results = []

        # テーブルを探す（keirin.jpの構造に合わせる）
        table = soup.select_one("table.resultTable, table.raceResult, table")
        if not table:
            return results

        for row in table.select("tbody tr, tr"):
            cells = row.select("td")
            if len(cells) < 5:
                continue
            try:
                # 着順が数値でない行はスキップ
                pos_text = cells[0].get_text(strip=True)
                if not pos_text.isdigit():
                    continue

                # 選手IDをリンクから抽出
                rider_link = row.select_one("a[href*='player'], a[href*='rider']")
                rider_id = ""
                if rider_link:
                    m = re.search(r"(?:player|rider)[_=]?(\d+)", rider_link.get("href", ""))
                    rider_id = m.group(1) if m else ""

                result = {
                    "race_id": race_id,
                    "rider_id": rider_id,
                    "finish_position": self._safe_int(pos_text),
                    "frame_number": self._safe_int(cells[1].get_text(strip=True)) if len(cells) > 1 else None,
                    "bike_number": self._safe_int(cells[2].get_text(strip=True)) if len(cells) > 2 else None,
                    "rider_name": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                    "class": cells[4].get_text(strip=True) if len(cells) > 4 else None,
                    "prefecture": cells[5].get_text(strip=True) if len(cells) > 5 else None,
                    "gear_ratio": self._safe_float(cells[6].get_text(strip=True)) if len(cells) > 6 else None,
                    "finish_time": cells[7].get_text(strip=True) if len(cells) > 7 else None,
                    "margin": cells[8].get_text(strip=True) if len(cells) > 8 else None,
                    "odds": self._safe_float(cells[9].get_text(strip=True)) if len(cells) > 9 else None,
                    "popularity": self._safe_int(cells[10].get_text(strip=True)) if len(cells) > 10 else None,
                }
                results.append(result)
            except (IndexError, ValueError) as e:
                logger.warning("Result row parse error (race_id=%s): %s", race_id, e)
                continue

        return results

    def _parse_entry_table(self, soup: BeautifulSoup, race_id: str) -> list[dict]:
        """出走表テーブルをパース"""
        entries = []
        table = soup.select_one("table.entryTable, table.shutubaTable, table")
        if not table:
            return entries

        for row in table.select("tbody tr, tr"):
            cells = row.select("td")
            if len(cells) < 5:
                continue
            try:
                rider_link = row.select_one("a[href*='player'], a[href*='rider']")
                rider_id = ""
                if rider_link:
                    m = re.search(r"(?:player|rider)[_=]?(\d+)", rider_link.get("href", ""))
                    rider_id = m.group(1) if m else ""

                if not rider_id:
                    continue

                entry = {
                    "race_id": race_id,
                    "rider_id": rider_id,
                    "frame_number": self._safe_int(cells[0].get_text(strip=True)),
                    "bike_number": self._safe_int(cells[1].get_text(strip=True)) if len(cells) > 1 else None,
                    "rider_name": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                    "class": cells[3].get_text(strip=True) if len(cells) > 3 else None,
                    "prefecture": cells[4].get_text(strip=True) if len(cells) > 4 else None,
                    "gear_ratio": self._safe_float(cells[5].get_text(strip=True)) if len(cells) > 5 else None,
                }
                entries.append(entry)
            except (IndexError, ValueError) as e:
                logger.warning("Entry row parse error (race_id=%s): %s", race_id, e)
                continue

        return entries

    # ── ユーティリティ ──

    @staticmethod
    def _safe_int(text: str) -> int | None:
        try:
            return int(re.sub(r"[^\d]", "", text))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(text: str) -> float | None:
        try:
            return float(text.replace(",", ""))
        except (ValueError, TypeError):
            return None
