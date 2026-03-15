"""X(Twitter)投稿"""

import tweepy
from config import X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET


class XPoster:
    """Tweepyを使ったX投稿"""

    def __init__(self):
        self.client = None
        if all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET]):
            self.client = tweepy.Client(
                consumer_key=X_API_KEY,
                consumer_secret=X_API_SECRET,
                access_token=X_ACCESS_TOKEN,
                access_token_secret=X_ACCESS_SECRET,
            )

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def post(self, text: str) -> dict | None:
        """テキストをXに投稿"""
        if not self.client:
            print("[X] API未設定。投稿スキップ。")
            print(f"[X] 投稿内容:\n{text}")
            return None

        if len(text) > 280:
            text = text[:277] + "..."

        response = self.client.create_tweet(text=text)
        tweet_id = response.data["id"]
        print(f"[X] 投稿完了: https://x.com/i/status/{tweet_id}")
        return response.data

    def post_prediction(self, date_display: str, venue_display: str,
                        top_races: list[dict], note_url: str = "") -> dict | None:
        """予測ティーザーを投稿"""
        from publishing.note_formatter import NoteFormatter
        formatter = NoteFormatter()
        text = formatter.generate_x_teaser(date_display, venue_display, top_races, note_url)
        return self.post(text)

    def post_result(self, results_text: str) -> dict | None:
        """結果報告を投稿"""
        return self.post(results_text)
