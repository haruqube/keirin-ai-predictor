## TODO
- [ ] 過去データ取得完了待ち（2025年 主要12場、バックグラウンド実行中）
- [ ] 2024年データも取得
- [ ] モデル学習・精度検証
- [ ] 予測スクリプトの動作確認
- [ ] X APIキー設定・投稿テスト
- [ ] note.com記事生成テンプレート作成
- [x] weekly_pipeline.py 作成

## 完了
- [x] プロジェクト構造作成
- [x] DBスキーマ設計（races, riders, race_results, entries, rider_stats, predictions, prediction_results）
- [x] netkeirinスクレイパー実装・動作確認
- [x] /db/result/ で過去データアクセス可能を確認
- [x] /db/race_program/ で開催日一括取得の効率化
- [x] 特徴量設計（選手成績13 + レース条件7 + ライン7 = 27特徴量）
- [x] LightGBM LambdaRankモデル
- [x] Git初期化・GitHub移行

## メモ
- データソース: keirin.netkeiba.com (netkeirin)
- race_id形式: 12桁 = YYYYMMDD + jyo_cd(2) + race_num(2)
- /db/result/ は過去データにアクセス可能（/race/result/ は直近のみ）
- /db/race_program/?kaisai_group_id=YYYYMMDD+JJ で開催全race_idを一括取得
- 全場×全日の総当たりだがprogramが空なら1リクエストでスキップ
- 主要12場: 前橋,大宮,京王閣,立川,松戸,川崎,平塚,名古屋,向日町,岸和田,松山,小倉
- 推定所要時間: 1年分×12場 ≈ 2-3時間
