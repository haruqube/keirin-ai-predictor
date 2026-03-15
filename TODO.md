## TODO
## 進行中

## 完了
- [x] プロジェクト構造作成
- [x] DBスキーマ設計（races, riders, race_results, entries, rider_stats, predictions, prediction_results）
- [x] netkeirinスクレイパー実装・動作確認
- [x] /db/result/ で過去データアクセス可能を確認
- [x] /db/race_program/ で開催日一括取得の効率化
- [x] 特徴量設計（選手成績13 + レース条件7 + ライン7 = 27特徴量）
- [x] LightGBM LambdaRankモデル
- [x] Git初期化・GitHub移行
- [x] 過去データ取得完了（2024-2025年 主要12場、14,099レース）
- [x] スクレイピング並列化（ThreadPoolExecutor, 4ワーカー）
- [x] 特徴量ビルダーのバッチ最適化
- [x] モデル学習・精度検証（Top1: 38.1%, Top3: 58.8%）
- [x] 予測スクリプト動作確認（伊東 2026-03-08: Top1 25%, Top3 55.6%）
- [x] ライン情報パース実装（/race/entry/ DeployYosoセクション）
- [x] ライン補正パラメータ最適化（Top3: 55.6% → 66.7%, +11.1pp改善）
- [x] 特徴量拡張（27→32: last_1lap×3 + velodrome×2）
- [x] ハイパーパラメータチューニング（243組グリッドサーチ）
- [x] note.com記事生成テンプレート（Jinja2: note_article.md.j2, x_teaser.j2）
- [x] publishing モジュール（note_formatter.py, x_poster.py）
- [x] 記事生成スクリプト（scripts/generate_article.py）
- [x] weekly_pipeline.py（--predict / --result）
- [x] X APIキー設定（Free tier廃止のため手動コピペ運用、APIキーは.envに保存済み）

## メモ
- データソース: keirin.netkeiba.com (netkeirin)
- race_id形式: 12桁 = YYYYMMDD + jyo_cd(2) + race_num(2)
- /db/result/ は過去データにアクセス可能（/race/result/ は直近のみ）
- /db/race_program/?kaisai_group_id=YYYYMMDD+JJ で開催全race_idを一括取得
- 主要12場: 前橋,大宮,京王閣,立川,松戸,川崎,平塚,名古屋,向日町,岸和田,松山,小倉
- ライン情報: /race/entry/ のDeployYosoセクションから取得（過去データは不可、予測時のみ）
- ライン補正: 3人ライン番手+0.30が最有利、3人ライン3番手-0.30が最不利
- 最強ライン所属ボーナス: +0.20
- 特徴量重要度Top3: rider_class_num, rider_win_rate_all, rider_place_rate_all
- 新特徴量: rider_avg_last_1lap, rider_avg_last_1lap_recent5, rider_best_last_1lap, rider_velodrome_win_rate, rider_velodrome_race_count
- 最適ハイパーパラメータ: num_leaves=15, lr=0.03, min_data=50, ff=0.6, bf=0.8, rounds=800
- モデル精度: Top1=38.4%, Top3=59.3%（テストセット、ライン補正なし）
