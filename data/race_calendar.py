"""開催カレンダー取得"""

import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from config import KEIRIN_JP_BASE_URL, REQUEST_HEADERS, SCRAPE_DELAY


def get_today_and_tomorrow_dates() -> list[str]:
    """今日と明日の日付をYYYYMMDD形式で返す"""
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    return [today.strftime("%Y%m%d"), tomorrow.strftime("%Y%m%d")]


def get_kaisai_dates(year: int, month: int) -> list[str]:
    """keirin.jpカレンダーから開催日一覧を取得"""
    url = f"{KEIRIN_JP_BASE_URL}/race/calendar?date={year}-{month:02d}-01"
    time.sleep(SCRAPE_DELAY)
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")

        dates = []
        # カレンダーのリンクから日付を抽出
        for link in soup.select("a[href*='date=']"):
            href = link.get("href", "")
            m = re.search(r"date=(\d{4}-\d{2}-\d{2})", href)
            if m:
                date_str = m.group(1).replace("-", "")
                if date_str not in dates and date_str.startswith(f"{year}{month:02d}"):
                    dates.append(date_str)

        return sorted(dates)
    except Exception:
        return []
