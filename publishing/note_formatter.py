"""Jinja2テンプレートでnote.com記事を生成"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from config import TEMPLATES_DIR, RESULTS_DIR


class NoteFormatter:
    """note.com向け記事フォーマッター"""

    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            keep_trailing_newline=True,
        )

    def generate_article(
        self,
        date_display: str,
        venue_display: str,
        races: list[dict],
        last_week_accuracy: str = "集計中",
    ) -> str:
        """note.com向けマークダウン記事を生成

        Args:
            date_display: 表示用日付 (例: "2026/3/9(日)")
            venue_display: 開催場 (例: "前橋・川崎・小倉")
            races: レース情報リスト
            last_week_accuracy: 先週の的中率表示
        """
        template = self.env.get_template("note_article.md.j2")
        return template.render(
            date_display=date_display,
            venue_display=venue_display,
            race_count=len(races),
            races=races,
            last_week_accuracy=last_week_accuracy,
        )

    def generate_x_teaser(
        self,
        date_display: str,
        venue_display: str,
        top_races: list[dict],
        note_url: str = "",
    ) -> str:
        """X(Twitter)用ティーザーを生成"""
        template = self.env.get_template("x_teaser.j2")
        return template.render(
            date_display=date_display,
            venue_display=venue_display,
            top_races=top_races[:3],
            note_url=note_url,
        )

    def save_article(self, content: str, filename: str) -> str:
        """記事をファイルに保存"""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / filename
        path.write_text(content, encoding="utf-8")
        return str(path)
