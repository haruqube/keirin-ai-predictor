"""開催カレンダー取得（netkeirin）"""

import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from config import NETKEIRIN_BASE_URL, REQUEST_HEADERS, SCRAPE_DELAY, VELODROME_CODES


def get_today_and_tomorrow_dates() -> list[str]:
    """今日と明日の日付をYYYYMMDD形式で返す"""
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    return [today.strftime("%Y%m%d"), tomorrow.strftime("%Y%m%d")]


def get_kaisai_dates(year: int, month: int) -> list[str]:
    """指定年月の開催日一覧を取得

    netkeirinの各競輪場ページからrace_idに含まれる日付を収集する。
    """
    dates = set()
    # 全競輪場を回すのはコストが高いので、主要場のみチェック
    # race_idのパターン: YYYYMMDD + jyo_cd + RR
    target_prefix = f"{year}{month:02d}"

    for jyo_cd in VELODROME_CODES:
        url = f"{NETKEIRIN_BASE_URL}/race/course/?jyo_cd={jyo_cd}"
        time.sleep(SCRAPE_DELAY)
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
            resp.encoding = "utf-8"
            # race_idからYYYYMMDD部分を抽出
            for m in re.finditer(r"race_id=(\d{8})" + re.escape(jyo_cd) + r"\d{2}", resp.text):
                date_str = m.group(1)
                if date_str.startswith(target_prefix):
                    dates.add(date_str)
        except Exception:
            continue

    return sorted(dates)


def get_active_velodromes(date: str) -> list[str]:
    """指定日に開催中の競輪場コードを返す

    全場のレース一覧ページを確認し、該当日のrace_idが存在する場をリストアップ。
    """
    active = []
    for jyo_cd in VELODROME_CODES:
        url = f"{NETKEIRIN_BASE_URL}/race/course/?jyo_cd={jyo_cd}"
        time.sleep(SCRAPE_DELAY)
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
            resp.encoding = "utf-8"
            pattern = f"race_id={date}{jyo_cd}"
            if pattern in resp.text:
                active.append(jyo_cd)
        except Exception:
            continue
    return active
