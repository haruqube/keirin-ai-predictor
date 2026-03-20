"""競輪予想AI設定"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── パス ──
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = BASE_DIR / "db" / "keirin.db"
RESULTS_DIR = BASE_DIR / "results"
TEMPLATES_DIR = BASE_DIR / "publishing" / "templates"

# ── X (Twitter) ──
X_API_KEY = os.getenv("X_API_KEY", "")
X_API_SECRET = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_SECRET = os.getenv("X_ACCESS_SECRET", "")

# ── Supabase ──
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# ── スクレイピング ──
SCRAPE_DELAY = 1.5  # 秒
NETKEIRIN_BASE_URL = "https://keirin.netkeiba.com"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# ── モデル ──
TRAIN_YEARS = [2022, 2023, 2024, 2025]
TEST_YEARS = [2025]
LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3],
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "verbose": -1,
}
LGBM_NUM_BOOST_ROUND = 1000
LGBM_EARLY_STOPPING_ROUNDS = 100

# ── 記事 ──
NOTE_PRICE_NORMAL = 200
NOTE_PRICE_G1 = 500

# ── 競輪場コード (netkeirin jyo_cd) ──
VELODROME_CODES = {
    "11": "函館", "12": "青森", "13": "いわき平",
    "21": "弥彦", "22": "前橋", "23": "取手", "24": "宇都宮", "25": "大宮",
    "26": "西武園", "27": "京王閣", "28": "立川",
    "31": "松戸", "32": "千葉", "33": "川崎", "34": "平塚",
    "35": "小田原", "36": "伊東", "37": "静岡",
    "41": "名古屋", "42": "岐阜", "43": "大垣", "44": "豊橋",
    "45": "富山", "46": "松阪", "47": "四日市",
    "51": "福井", "52": "奈良", "53": "向日町", "54": "和歌山", "55": "岸和田",
    "61": "玉野", "62": "広島", "63": "防府",
    "71": "高松", "72": "小松島", "73": "高知", "74": "松山",
    "81": "小倉", "82": "久留米", "83": "武雄", "84": "佐世保", "85": "別府",
    "86": "熊本", "87": "熊本",
}

# ── グレード ──
GRADE_MAP = {
    "GP": 1, "G1": 2, "G2": 3, "G3": 4,
    "F1": 5, "F2": 6,
}

# ── 選手級班 ──
CLASS_MAP = {
    "SS": 1, "S1": 2, "S2": 3,
    "A1": 4, "A2": 5, "A3": 6,
}

# ── バンク周長 ──
BANK_LENGTH = {
    "前橋": 335, "小田原": 333, "伊東": 333, "松阪": 333,
    "奈良": 333, "防府": 333, "小松島": 333, "高知": 333,
    "松山": 333, "佐世保": 333, "熊本": 333,
    "大宮": 500, "立川": 400, "京王閣": 400,
}
DEFAULT_BANK_LENGTH = 400
