## TODO
- [ ] keirin.jpのHTML構造を実際に確認し、スクレイパーのセレクタを調整
- [ ] 過去データ取得（2023-2025、init_db.py）
- [ ] モデル学習・精度検証
- [ ] 予測スクリプトの動作確認
- [ ] X APIキー設定・投稿テスト
- [ ] note.com記事生成テンプレート作成
- [ ] weekly_pipeline.py 作成

## 完了
- [x] プロジェクト構造作成
- [x] DBスキーマ設計（races, riders, race_results, entries, rider_stats, predictions, prediction_results）
- [x] スクレイパー雛形（keirin.jp用）
- [x] 特徴量設計（選手成績13 + レース条件7 + ライン7 = 27特徴量）
- [x] LightGBM LambdaRankモデル
- [x] Git初期化・GitHub移行

## メモ
- 競輪は1レース最大9車（競馬の半分以下）→ 予測しやすい可能性
- ライン戦術が最大の特徴 → line_features.py で数値化
- keirin.jpのHTML構造は未確認 → スクレイパー調整が最初のタスク
