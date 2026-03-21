"""netkeirinスクレイパー（キャッシュ付き）

データソース: https://keirin.netkeiba.com
race_id形式: YYYYMMDD + 競輪場コード(2桁) + レース番号(2桁) = 12桁
  例: 202603057511 = 2026/03/05, 松山(75), 11R
"""

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
    CACHE_DIR, NETKEIRIN_BASE_URL,
    REQUEST_HEADERS, SCRAPE_DELAY, VELODROME_CODES,
)

logger = logging.getLogger(__name__)


class KeirinScraper:
    """netkeirinからレース情報・結果をスクレイピング"""

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

    def scrape_race_list_for_velodrome(self, jyo_cd: str, date: str) -> list[str]:
        """指定競輪場・日付のレースID一覧を取得

        jyo_cd: 競輪場コード (e.g. "75")
        date: YYYYMMDD形式
        戻り値: race_idのリスト
        """
        cache_key = f"race_list_{jyo_cd}_{date}"
        cached = self._get_json_cache(cache_key)
        if cached is not None:
            return cached

        url = f"{NETKEIRIN_BASE_URL}/race/course/?jyo_cd={jyo_cd}"
        html = self._get(url)

        # race_idパターン: YYYYMMDD + jyo_cd + RR
        pattern = re.compile(rf"race_id=({re.escape(date)}{re.escape(jyo_cd)}\d{{2}})")
        race_ids = sorted(set(pattern.findall(html)))

        self._set_json_cache(cache_key, race_ids)
        return race_ids

    def scrape_race_list(self, date: str) -> list[str]:
        """指定日の全競輪場のレースID一覧を取得

        date: YYYYMMDD形式
        """
        cache_key = f"race_list_all_{date}"
        cached = self._get_json_cache(cache_key)
        if cached is not None:
            return cached

        all_ids = []
        for jyo_cd in VELODROME_CODES:
            ids = self.scrape_race_list_for_velodrome(jyo_cd, date)
            all_ids.extend(ids)

        all_ids = sorted(set(all_ids))
        self._set_json_cache(cache_key, all_ids)
        return all_ids

    def scrape_race_result(self, race_id: str) -> dict:
        """レース結果ページをパース

        URL: /race/result/?race_id=XXXXXXXXXXXXXX
        テーブル列: 着, 枠番, 車番, 選手名, 着差, 上り, 決, SB
        選手リンク: /db/profile/?id=XXXXX
        """
        cache_key = f"race_result_{race_id}"
        cached = self._get_json_cache(cache_key)
        if cached:
            return cached

        url = f"{NETKEIRIN_BASE_URL}/race/result/?race_id={race_id}"
        html = self._get(url)
        soup = BeautifulSoup(html, "lxml")

        race_info = self._parse_race_info(soup, race_id)
        results = self._parse_result_table(soup, race_id)

        # ライン並び予想をパースして結果に付与
        line_data = self._parse_line_formation(soup)
        if line_data:
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
                    bike_to_line[bike_num] = {"line_group": line_group, "line_role": role}

            for result in results:
                bn = result.get("bike_number")
                if bn and bn in bike_to_line:
                    result["line_group"] = bike_to_line[bn]["line_group"]
                    result["line_role"] = bike_to_line[bn]["line_role"]

        race_info["results"] = results
        race_info["rider_count"] = len(results)

        # 配当情報パース
        race_info["payouts"] = self._parse_payout_table(soup)

        self._set_json_cache(cache_key, race_info)
        return race_info

    def scrape_race_entry(self, race_id: str) -> dict:
        """出走表ページをパース

        URL: /race/entry/?race_id=XXXXXXXXXXXXXX
        Fixed table (RaceCard_Simple_Table_Fixed): 枠, 車, 予想印
        Static table (RaceCard_Simple_Table_Static): 本紙, 選手名, 競走得点,
            脚質, S, B, 逃げ, まくり, 差し, マーク, 1着~着外, 勝率~3連対率, ギヤ倍数, コメント
        """
        cache_key = f"race_entry_{race_id}"
        cached = self._get_json_cache(cache_key)
        if cached:
            return cached

        url = f"{NETKEIRIN_BASE_URL}/race/entry/?race_id={race_id}"
        html = self._get(url)
        soup = BeautifulSoup(html, "lxml")

        race_info = self._parse_race_info(soup, race_id)
        entries = self._parse_entry_table(soup, race_id)

        # ライン並び予想をパース
        line_data = self._parse_line_formation(soup)
        # 車番→ライン情報のマッピングをentriesに付与
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
                bike_to_line[bike_num] = {"line_group": line_group, "line_role": role}

        for entry in entries:
            bn = entry.get("bike_number")
            if bn and bn in bike_to_line:
                entry["line_group"] = bike_to_line[bn]["line_group"]
                entry["line_role"] = bike_to_line[bn]["line_role"]

        race_info["entries"] = entries
        race_info["rider_count"] = len(entries)
        race_info["line_formation"] = line_data

        self._set_json_cache(cache_key, race_info)
        return race_info

    def _parse_race_info(self, soup: BeautifulSoup, race_id: str) -> dict:
        """レース情報をパース（結果ページ・出走表共通）"""
        info = {"race_id": race_id}

        # race_idから基本情報を抽出: YYYYMMDD(8) + JJ(2) + RR(2) = 12桁
        info["date"] = f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"
        jyo_cd = race_id[8:10]
        info["velodrome_code"] = jyo_cd
        info["velodrome"] = VELODROME_CODES.get(jyo_cd, jyo_cd)
        info["race_number"] = int(race_id[10:12]) if len(race_id) >= 12 else None

        # ページ全文からレース情報を抽出
        page_text = soup.get_text(" ", strip=True)

        # レース名: "Ｓ級 一次予選" のようなパターン
        race_name = ""
        # RaceList_NameBox_inner内の最初のテキスト要素を取得
        name_inner = soup.select_one(".RaceList_NameBox_inner")
        if name_inner:
            # 最初の意味のあるテキストを取得
            for child in name_inner.stripped_strings:
                race_name = child
                break
        if not race_name:
            name_box = soup.select_one(".RaceName, .RaceList_NameBox")
            if name_box:
                race_name = name_box.get_text(strip=True)[:50]
        info["race_name"] = race_name

        # 距離・周回数: "2025m 5周" パターンをpage全体から検索
        d_match = re.search(r"(\d{3,4})m\s*(\d+)周", page_text)
        if d_match:
            info["distance"] = int(d_match.group(1))
            info["laps"] = int(d_match.group(2))
        else:
            d_match2 = re.search(r"(\d{3,4})m", page_text)
            info["distance"] = int(d_match2.group(1)) if d_match2 else None
            info["laps"] = None

        # 発走時刻: "発走 20:40" パターン
        time_match = re.search(r"発走\s*(\d{1,2}:\d{2})", page_text)
        info["start_time"] = time_match.group(1) if time_match else None

        # グレード: GIII, GII, GI, GP, FI, FII をページテキストから検索
        info["grade"] = None
        grade_patterns = [
            (r"GP|グランプリ", "GP"),
            (r"GI(?!I)|ＧＩ(?!Ｉ)", "G1"),
            (r"GII(?!I)|ＧＩＩ(?!Ｉ)", "G2"),
            (r"GIII|ＧＩＩＩgr", "G3"),
        ]
        for pattern, grade in grade_patterns:
            if re.search(pattern, page_text):
                info["grade"] = grade
                break
        if not info["grade"]:
            if "FII" in page_text or "ＦＩＩ" in page_text:
                info["grade"] = "F2"
            elif "FI" in page_text or "ＦＩ" in page_text:
                info["grade"] = "F1"

        return info

    def _parse_result_table(self, soup: BeautifulSoup, race_id: str) -> list[dict]:
        """レース結果テーブルをパース

        netkeirin結果テーブル列:
        [0]着 [1]枠番 [2]車番 [3]選手名(+リンク) [4]着差 [5]上り [6]決 [7]SB
        選手リンク: /db/profile/?id=XXXXX → player_id
        """
        results = []

        # TableSlideArea内のテーブルを探す
        table_area = soup.select_one(".TableSlideArea")
        if not table_area:
            # フォールバック: 最初のtableを試す
            table_area = soup

        for row in table_area.select("tbody tr, table tr"):
            cells = row.select("td")
            if len(cells) < 4:
                continue

            try:
                pos_text = cells[0].get_text(strip=True)
                if not pos_text or not pos_text[0].isdigit():
                    continue

                # 選手ID: player_img_XXXXX or /db/profile/?id=XXXXX
                rider_id = ""
                img_el = row.select_one("[id^='player_img_']")
                if img_el:
                    m = re.search(r"player_img_(\d+)", img_el.get("id", ""))
                    rider_id = m.group(1) if m else ""
                if not rider_id:
                    link = row.select_one("a[href*='/db/profile/']")
                    if link:
                        m = re.search(r"id=(\d+)", link.get("href", ""))
                        rider_id = m.group(1) if m else ""

                # 選手名セル(Player_Info): "晝田宗一郎岡山 26歳115期 Ｓ2"
                # /db/result/ では "お気に入り選手" テキストが含まれる場合がある
                player_cell = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                player_cell = player_cell.replace("お気に入り選手", " ")
                parsed = self._parse_player_info(player_cell)
                clean_name = parsed["name"]
                prefecture = parsed["prefecture"]

                # 着順テキストから数値のみ抽出 ("1着" -> 1)
                finish_pos = self._safe_int(pos_text)

                result = {
                    "race_id": race_id,
                    "rider_id": rider_id,
                    "finish_position": finish_pos,
                    "frame_number": self._safe_int(cells[1].get_text(strip=True)) if len(cells) > 1 else None,
                    "bike_number": self._safe_int(cells[2].get_text(strip=True)) if len(cells) > 2 else None,
                    "rider_name": clean_name,
                    "class": parsed["class"],
                    "prefecture": prefecture,
                    "margin": cells[4].get_text(strip=True) if len(cells) > 4 else None,
                    "last_1lap": self._safe_float(cells[5].get_text(strip=True)) if len(cells) > 5 else None,
                    "winning_move": cells[6].get_text(strip=True) if len(cells) > 6 else None,
                }
                results.append(result)
            except (IndexError, ValueError) as e:
                logger.warning("Result row parse error (race_id=%s): %s", race_id, e)
                continue

        return results

    def _parse_payout_table(self, soup: BeautifulSoup) -> dict:
        """配当テーブルをパース（2連単・2連複）"""
        payout = {}
        pay_el = soup.select_one('.Payout_Detail_Table')
        if not pay_el:
            return payout
        text = pay_el.get_text(' ', strip=True)

        # 2連単
        m = re.search(r'２車単\s+(\d+[->＞\s]+\d+)\s+([\d,]+)円\s+(\d+)人気', text)
        if m:
            payout['nisyatan_combo'] = m.group(1).strip()
            payout['nisyatan_payout'] = int(m.group(2).replace(',', ''))
            payout['nisyatan_popularity'] = int(m.group(3))

        # 2連複
        m2 = re.search(r'２車複\s+(\d+[->＞\s—=]+\d+)\s+([\d,]+)円', text)
        if m2:
            payout['nishafuku_combo'] = m2.group(1).strip()
            payout['nishafuku_payout'] = int(m2.group(2).replace(',', ''))

        return payout

    def _parse_entry_table(self, soup: BeautifulSoup, race_id: str) -> list[dict]:
        """出走表テーブルをパース

        netkeirinの出走表は2つのRaceCard_Simple_Tableがある。
        2番目のテーブル（詳細版）を使用:
        列: 枠, 車, チェック, 本紙, 選手名(Player_Info), 競走得点, 脚質, S, B,
            逃げ, まくり, 差し, マーク, 1着, 2着, 3着, 着外, 勝率, 2連対率, 3連対率, ギヤ倍数, コメント
        """
        entries = []

        tables = soup.select("table.RaceCard_Simple_Table")
        if len(tables) < 2:
            return entries

        # 2番目のテーブル（詳細版）を使用
        detail_table = tables[1]

        for row in detail_table.select("tbody tr, tr"):
            cells = row.select("td")
            if len(cells) < 6:
                continue

            try:
                # 選手IDを抽出
                # 出走表では img src="...players/player_XXXXX.jpg" から取得
                rider_id = ""
                img_el = row.select_one("img[src*='player_']")
                if img_el:
                    m = re.search(r"player_(\d+)", img_el.get("src", ""))
                    rider_id = m.group(1) if m else ""
                if not rider_id:
                    img_el2 = row.select_one("[id^='player_img_'], [id^='entry_player_photo_']")
                    if img_el2:
                        m = re.search(r"player_(\d+)", img_el2.get("src", "") or img_el2.get("id", ""))
                        rider_id = m.group(1) if m else ""
                if not rider_id:
                    link = row.select_one("a[href*='/db/profile/']")
                    if link:
                        m = re.search(r"id=(\d+)", link.get("href", ""))
                        rider_id = m.group(1) if m else ""

                if not rider_id:
                    continue

                # Player_Infoセル (index 4)
                # "ヨシダタクヤ吉田拓矢お気に入り選手茨城 30歳107期 ＳＳ"
                player_text = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                # "お気に入り選手" を除去
                player_text = player_text.replace("お気に入り選手", " ")
                # カタカナ読みを除去（先頭のカタカナ）
                player_text = re.sub(r"^[ァ-ヴー]+", "", player_text)

                parsed = self._parse_player_info(player_text)

                # 競走得点 (index 5)
                score = self._safe_float(cells[5].get_text(strip=True)) if len(cells) > 5 else None
                # 脚質 (index 6)
                leg_type = cells[6].get_text(strip=True) if len(cells) > 6 else None

                # S, B, 逃げ, まくり, 差し, マーク (index 7-12)
                # 1着, 2着, 3着, 着外 (index 13-16)
                # 勝率, 2連対率, 3連対率 (index 17-19)
                win_rate = self._safe_float(cells[17].get_text(strip=True)) if len(cells) > 17 else None
                place_rate = self._safe_float(cells[18].get_text(strip=True)) if len(cells) > 18 else None
                top3_rate = self._safe_float(cells[19].get_text(strip=True)) if len(cells) > 19 else None
                # ギヤ倍数 (index 20)
                gear_ratio = self._safe_float(cells[20].get_text(strip=True)) if len(cells) > 20 else None
                # コメント (index 21, 最後)
                comment = cells[21].get_text(strip=True) if len(cells) > 21 else ""

                entry = {
                    "race_id": race_id,
                    "rider_id": rider_id,
                    "frame_number": self._safe_int(cells[0].get_text(strip=True)),
                    "bike_number": self._safe_int(cells[1].get_text(strip=True)),
                    "rider_name": parsed["name"],
                    "class": parsed["class"],
                    "prefecture": parsed["prefecture"],
                    "age": parsed["age"],
                    "period": parsed["period"],
                    "avg_competition_score": score,
                    "leg_type": leg_type,
                    "win_rate": win_rate,
                    "place_rate": place_rate,
                    "top3_rate": top3_rate,
                    "gear_ratio": gear_ratio,
                    "comment": comment,
                }
                entries.append(entry)
            except (IndexError, ValueError) as e:
                logger.warning("Entry row parse error (race_id=%s): %s", race_id, e)
                continue

        return entries

    # ── ライン並び予想 ──

    @staticmethod
    def _parse_line_formation(soup: BeautifulSoup) -> list[list[dict]]:
        """DeployYosoセクションからライン並びをパース

        戻り値: [[{"bike_number": 7, "name": "深谷知"}, ...], [...], ...]
        各リストが1つのライン。ライン内の順序が自力→番手→3番手。
        """
        deploy = soup.select_one(".DeployYoso")
        if not deploy:
            return []

        boxes = deploy.select(".DeployInBox")
        if not boxes:
            return []

        lines = []
        current_line = []

        for box in boxes:
            sep = box.select_one(".WakuSeparat")
            num_el = box.select_one(".Shaban_Num")

            if sep and not num_el:
                # セパレータ = ラインの区切り
                if current_line:
                    lines.append(current_line)
                    current_line = []
            elif num_el:
                try:
                    bike_num = int(num_el.get_text(strip=True))
                except ValueError:
                    continue
                name_el = box.select_one(".Name")
                name = name_el.get_text(strip=True) if name_el else ""
                current_line.append({"bike_number": bike_num, "name": name})

        if current_line:
            lines.append(current_line)

        return lines

    # ── ユーティリティ ──

    _PREF_PATTERN = re.compile(
        r"(北海道|青森|岩手|宮城|秋田|山形|福島|茨城|栃木|群馬|"
        r"埼玉|千葉|東京|神奈川|新潟|富山|石川|福井|山梨|長野|"
        r"岐阜|静岡|愛知|三重|滋賀|京都|大阪|兵庫|奈良|和歌山|"
        r"鳥取|島根|岡山|広島|山口|徳島|香川|愛媛|高知|福岡|"
        r"佐賀|長崎|熊本|大分|宮崎|鹿児島|沖縄)"
    )
    _CLASS_PATTERN = re.compile(r"(SS|Ｓ[Ｓ12]|S[S12]|Ａ[123]|A[123])")

    @classmethod
    def _parse_player_info(cls, text: str) -> dict:
        """Player_Infoセルのテキストをパース

        入力例: "晝田宗一郎岡山 26歳115期 Ｓ2"
        """
        result = {"name": "", "prefecture": None, "class": None, "age": None, "period": None}

        if not text:
            return result

        # 級班
        class_match = cls._CLASS_PATTERN.search(text)
        if class_match:
            raw_class = class_match.group(1)
            # 全角→半角正規化
            result["class"] = raw_class.replace("Ｓ", "S").replace("Ａ", "A")

        # 年齢
        age_match = re.search(r"(\d{2,3})歳", text)
        result["age"] = int(age_match.group(1)) if age_match else None

        # 期
        period_match = re.search(r"(\d{2,3})期", text)
        result["period"] = int(period_match.group(1)) if period_match else None

        # 府県: 名前の直後に続く
        pref_match = cls._PREF_PATTERN.search(text)
        result["prefecture"] = pref_match.group(1) if pref_match else None

        # 名前: 府県の前の部分 or スペースまで
        if pref_match:
            result["name"] = text[:pref_match.start()].strip()
        else:
            # スペースで区切って最初の部分
            result["name"] = text.split()[0] if text.split() else text

        return result

    def scrape_exacta_odds(self, race_id: str) -> dict[str, float]:
        """2連単（Exacta）オッズをAPI経由で取得

        戻り値: {"XXYY": odds, ...}  XX=1着車番, YY=2着車番（ゼロパディング2桁）
        例: {"0102": 3.7, "0103": 5.2, ...}
        取得失敗時は空dict
        """
        cache_key = f"exacta_odds_{race_id}"
        cached = self._get_json_cache(cache_key)
        if cached is not None:
            return cached

        url = (
            f"{NETKEIRIN_BASE_URL}/api/race/"
            f"?class=AplRaceOdds&method=get&race_id={race_id}&compress=0"
        )
        try:
            time.sleep(SCRAPE_DELAY)
            resp = self.session.get(url, timeout=30, headers={
                **REQUEST_HEADERS,
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": f"{NETKEIRIN_BASE_URL}/race/odds/",
            })
            data = resp.json()
            if data.get("status") != "OK":
                logger.debug("Odds API NG for %s: %s", race_id, data.get("reason", ""))
                return {}

            odds_key = f"nkrace_odds::{race_id}"
            odds_data = data.get("data", {}).get(odds_key, {})
            list_6 = odds_data.get("list_6", [])  # 2連単

            result = {}
            for item in list_6:
                combo = item[0]   # "XXYY"
                try:
                    odds_val = float(item[1])
                except (ValueError, IndexError):
                    continue
                result[combo] = odds_val

            self._set_json_cache(cache_key, result)
            return result

        except Exception as e:
            logger.debug("Odds fetch error for %s: %s", race_id, e)
            return {}

    @staticmethod
    def _safe_int(text: str) -> int | None:
        try:
            return int(re.sub(r"[^\d]", "", text))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(text: str) -> float | None:
        try:
            return float(text.replace(",", "").replace("%", "").strip())
        except (ValueError, TypeError):
            return None
