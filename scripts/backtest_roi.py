"""改善モデルのROIバックテスト

build_dataset() → model.predict() → 2連単シミュレーションで
改善前後のモデルの回収率を比較する。

賭け戦略:
  - F2は除外
  - HIGH (信頼度≥1.10): 500円×4点 = 2,000円
  - MED  (0.50≤信頼度<0.80): 100円×4点 = 400円
  - それ以外: 見送り
  - トリガミスキップ（max combo odds < 4.0）はオフラインでは判定不可
"""

import sys
import io
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from config import RESULTS_DIR, GRADE_MAP
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from db.schema import get_connection

logger = logging.getLogger(__name__)


def run_backtest(year_start: int = 2024, year_end: int = 2026):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # モデル読み込み
    model = LGBMRanker()
    model.load(str(RESULTS_DIR / "model_lgbm.pkl"))
    logger.info("Model loaded (%d features)", len(model.feature_names))

    # 特徴量ビルド
    builder = FeatureBuilder()
    feature_cols = builder.feature_names
    logger.info("Building dataset %d-%d...", year_start, year_end)
    df = builder.build_dataset(year_start, year_end)

    if df.empty:
        logger.error("No data")
        return

    df = df.dropna(subset=["finish_position"])
    logger.info("Dataset: %d rows", len(df))

    # レースのgrade情報を取得
    conn = get_connection()
    grade_df = pd.read_sql("""
        SELECT race_id, grade, date FROM races
        WHERE date >= ? AND date <= ?
    """, conn, params=(f"{year_start}-01-01", f"{year_end}-12-31"))

    # race_payoutsを取得
    payouts_df = pd.read_sql("""
        SELECT race_id, nisyatan_combo, nisyatan_payout
        FROM race_payouts
        WHERE race_id IN (SELECT race_id FROM races WHERE date >= ? AND date <= ?)
    """, conn, params=(f"{year_start}-01-01", f"{year_end}-12-31"))
    conn.close()

    payout_map = dict(zip(payouts_df["race_id"],
                          zip(payouts_df["nisyatan_combo"], payouts_df["nisyatan_payout"])))
    grade_map = dict(zip(grade_df["race_id"], grade_df["grade"]))
    date_map = dict(zip(grade_df["race_id"], grade_df["date"]))
    logger.info("Payouts: %d races, Grades: %d races", len(payout_map), len(grade_map))

    # 予測スコア計算
    X = df[feature_cols].fillna(0)
    df["pred_score"] = model.predict(X).values
    logger.info("Predictions done")

    # レースごとにシミュレーション
    results = []
    for race_id, group in df.groupby("race_id"):
        if len(group) < 3:
            continue

        grade = grade_map.get(race_id, "")
        race_date = date_map.get(race_id, "")
        year = race_date[:4] if race_date else ""

        # F2除外
        if grade == "F2":
            continue

        # 着順予測
        pred = group.sort_values("pred_score", ascending=False).reset_index(drop=True)
        actual = group.sort_values("finish_position").reset_index(drop=True)

        # 信頼度（◎-○ スコア差）
        confidence = pred.iloc[0]["pred_score"] - pred.iloc[1]["pred_score"]

        # 賭けカテゴリ判定
        if confidence >= 1.00:
            bet_cat = "HIGH"
            bet_per_ticket = 500
        elif 0.80 <= confidence < 1.00:
            bet_cat = "MED+"
            bet_per_ticket = 200
        elif 0.50 <= confidence < 0.80:
            bet_cat = "MED"
            bet_per_ticket = 100
        else:
            bet_cat = "SKIP"
            bet_per_ticket = 0

        # ◎ の的中確認
        honmei_rid = pred.iloc[0]["rider_id"]
        honmei_bike = int(pred.iloc[0]["bike_number"]) if pd.notna(pred.iloc[0]["bike_number"]) else 0
        actual_1st_rid = actual.iloc[0]["rider_id"]
        actual_2nd_rid = actual.iloc[1]["rider_id"] if len(actual) > 1 else ""

        # 実際の1着・2着の車番
        actual_1st_bike = int(actual.iloc[0]["bike_number"]) if pd.notna(actual.iloc[0]["bike_number"]) else 0
        actual_2nd_bike = int(actual.iloc[1]["bike_number"]) if len(actual) > 1 and pd.notna(actual.iloc[1]["bike_number"]) else 0

        top1_hit = (honmei_rid == actual_1st_rid)

        # 2連単の4点買い: ◎→○, ◎→▲, ◎→△1, ◎→△2
        tickets = []
        for i in range(1, min(5, len(pred))):
            target_bike = int(pred.iloc[i]["bike_number"]) if pd.notna(pred.iloc[i]["bike_number"]) else 0
            combo = f"{honmei_bike}>{target_bike}"
            tickets.append(combo)

        # 配当チェック
        payout_info = payout_map.get(race_id)
        winning_combo = payout_info[0] if payout_info else None
        payout_amount = payout_info[1] if payout_info else 0

        exacta_hit = False
        if winning_combo and winning_combo in tickets:
            exacta_hit = True

        # 投資・回収計算
        num_tickets = len(tickets)
        investment = bet_per_ticket * num_tickets if bet_cat != "SKIP" else 0
        payout_won = 0
        if exacta_hit and bet_cat != "SKIP":
            # 配当は100円あたり → 実際の回収 = payout_amount / 100 * bet_per_ticket
            payout_won = payout_amount / 100 * bet_per_ticket

        results.append({
            "race_id": race_id,
            "year": year,
            "grade": grade,
            "confidence": confidence,
            "bet_cat": bet_cat,
            "top1_hit": top1_hit,
            "exacta_hit": exacta_hit,
            "investment": investment,
            "payout_won": payout_won,
            "payout_amount": payout_amount,
            "honmei_bike": honmei_bike,
            "tickets": tickets,
            "winning_combo": winning_combo,
        })

    res_df = pd.DataFrame(results)
    logger.info("Simulation: %d races (F2 excluded)", len(res_df))

    # === 全体サマリー ===
    print("\n" + "=" * 70)
    print("  ROI バックテスト結果")
    print("=" * 70)

    # ◎的中率
    total = len(res_df)
    top1_hits = res_df["top1_hit"].sum()
    print(f"\n◎1着的中率: {top1_hits}/{total} ({top1_hits/total*100:.1f}%)")

    # ベットしたレースのみ
    bet_df = res_df[res_df["bet_cat"] != "SKIP"]
    if len(bet_df) > 0:
        total_investment = bet_df["investment"].sum()
        total_payout = bet_df["payout_won"].sum()
        roi = total_payout / total_investment * 100 if total_investment > 0 else 0
        exacta_hits = bet_df["exacta_hit"].sum()
        print(f"\n【全ベット】")
        print(f"  対象レース: {len(bet_df)}/{total}")
        print(f"  2連単的中: {exacta_hits}/{len(bet_df)} ({exacta_hits/len(bet_df)*100:.1f}%)")
        print(f"  投資額: ¥{total_investment:,.0f}")
        print(f"  回収額: ¥{total_payout:,.0f}")
        print(f"  回収率: {roi:.1f}%")
        print(f"  収支: ¥{total_payout - total_investment:+,.0f}")

    # カテゴリ別
    for cat in ["HIGH", "MED+", "MED"]:
        cat_df = res_df[res_df["bet_cat"] == cat]
        if len(cat_df) == 0:
            continue
        inv = cat_df["investment"].sum()
        pay = cat_df["payout_won"].sum()
        roi = pay / inv * 100 if inv > 0 else 0
        hits = cat_df["exacta_hit"].sum()
        t1h = cat_df["top1_hit"].sum()
        print(f"\n【{cat}】")
        print(f"  レース数: {len(cat_df)}")
        print(f"  ◎1着: {t1h}/{len(cat_df)} ({t1h/len(cat_df)*100:.1f}%)")
        print(f"  2連単的中: {hits}/{len(cat_df)} ({hits/len(cat_df)*100:.1f}%)")
        print(f"  投資: ¥{inv:,.0f}  回収: ¥{pay:,.0f}")
        print(f"  回収率: {roi:.1f}%")
        print(f"  収支: ¥{pay - inv:+,.0f}")

    # 年別
    print(f"\n{'─' * 70}")
    print("  年別内訳")
    print(f"{'─' * 70}")
    for year in sorted(res_df["year"].unique()):
        yr_df = res_df[res_df["year"] == year]
        yr_bet = yr_df[yr_df["bet_cat"] != "SKIP"]
        if len(yr_bet) == 0:
            continue
        inv = yr_bet["investment"].sum()
        pay = yr_bet["payout_won"].sum()
        roi = pay / inv * 100 if inv > 0 else 0
        hits = yr_bet["exacta_hit"].sum()
        t1 = yr_df["top1_hit"].sum()
        print(f"\n  {year}年: {len(yr_df)} races (bet: {len(yr_bet)})")
        print(f"    ◎1着: {t1}/{len(yr_df)} ({t1/len(yr_df)*100:.1f}%)")
        print(f"    2連単的中: {hits}/{len(yr_bet)} ({hits/len(yr_bet)*100:.1f}%)")
        print(f"    投資: ¥{inv:,.0f}  回収: ¥{pay:,.0f}")
        print(f"    回収率: {roi:.1f}%  収支: ¥{pay - inv:+,.0f}")

    # グレード別
    print(f"\n{'─' * 70}")
    print("  グレード別")
    print(f"{'─' * 70}")
    for grade in sorted(res_df["grade"].unique(), key=lambda g: GRADE_MAP.get(g, 9)):
        gr_df = res_df[res_df["grade"] == grade]
        gr_bet = gr_df[gr_df["bet_cat"] != "SKIP"]
        if len(gr_bet) == 0:
            continue
        inv = gr_bet["investment"].sum()
        pay = gr_bet["payout_won"].sum()
        roi = pay / inv * 100 if inv > 0 else 0
        hits = gr_bet["exacta_hit"].sum()
        t1 = gr_df["top1_hit"].sum()
        print(f"\n  {grade}: {len(gr_df)} races (bet: {len(gr_bet)})")
        print(f"    ◎1着: {t1}/{len(gr_df)} ({t1/len(gr_df)*100:.1f}%)")
        print(f"    2連単的中: {hits}/{len(gr_bet)} ({hits/len(gr_bet)*100:.1f}%)")
        print(f"    投資: ¥{inv:,.0f}  回収: ¥{pay:,.0f}")
        print(f"    回収率: {roi:.1f}%  収支: ¥{pay - inv:+,.0f}")

    # 信頼度帯別の詳細
    print(f"\n{'─' * 70}")
    print("  信頼度帯別")
    print(f"{'─' * 70}")
    bins = [(0, 0.30), (0.30, 0.50), (0.50, 0.80), (0.80, 1.10),
            (1.10, 1.50), (1.50, 2.00), (2.00, 999)]
    for lo, hi in bins:
        band = res_df[(res_df["confidence"] >= lo) & (res_df["confidence"] < hi)]
        if len(band) == 0:
            continue
        band_bet = band[band["bet_cat"] != "SKIP"]
        t1 = band["top1_hit"].sum()
        label = f"{lo:.2f}-{hi:.2f}" if hi < 999 else f"{lo:.2f}+"
        if len(band_bet) > 0:
            inv = band_bet["investment"].sum()
            pay = band_bet["payout_won"].sum()
            roi = pay / inv * 100 if inv > 0 else 0
            hits = band_bet["exacta_hit"].sum()
            print(f"  [{label}] {len(band)}R ◎的中{t1/len(band)*100:.0f}% | bet:{len(band_bet)} hit:{hits} ROI:{roi:.0f}% 収支:¥{pay-inv:+,.0f}")
        else:
            print(f"  [{label}] {len(band)}R ◎的中{t1/len(band)*100:.0f}% | skip")

    # 2026年のみ（真のOOS）
    print(f"\n{'=' * 70}")
    print("  2026年（Out-of-Sample）詳細")
    print("=" * 70)
    oos_df = res_df[res_df["year"] == "2026"]
    if len(oos_df) > 0:
        oos_bet = oos_df[oos_df["bet_cat"] != "SKIP"]
        t1 = oos_df["top1_hit"].sum()
        print(f"  総レース: {len(oos_df)} (F2除外済)")
        print(f"  ◎1着: {t1}/{len(oos_df)} ({t1/len(oos_df)*100:.1f}%)")
        if len(oos_bet) > 0:
            inv = oos_bet["investment"].sum()
            pay = oos_bet["payout_won"].sum()
            roi = pay / inv * 100 if inv > 0 else 0
            hits = oos_bet["exacta_hit"].sum()
            print(f"  ベット対象: {len(oos_bet)}")
            print(f"  2連単的中: {hits}/{len(oos_bet)} ({hits/len(oos_bet)*100:.1f}%)")
            print(f"  投資: ¥{inv:,.0f}  回収: ¥{pay:,.0f}")
            print(f"  回収率: {roi:.1f}%")
            print(f"  収支: ¥{pay - inv:+,.0f}")

            # HIGH/MED別
            for cat in ["HIGH", "MED"]:
                c = oos_bet[oos_bet["bet_cat"] == cat]
                if len(c) == 0:
                    continue
                ci = c["investment"].sum()
                cp = c["payout_won"].sum()
                cr = cp / ci * 100 if ci > 0 else 0
                ch = c["exacta_hit"].sum()
                print(f"    {cat}: {len(c)}R 的中{ch} ROI:{cr:.0f}% 収支:¥{cp-ci:+,.0f}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ROIバックテスト")
    parser.add_argument("--start", type=int, default=2024)
    parser.add_argument("--end", type=int, default=2026)
    args = parser.parse_args()
    run_backtest(args.start, args.end)
