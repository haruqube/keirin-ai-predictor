## TODO
- [ ] 会場別補正係数の導入（取手・小倉で高精度、防府・岐阜で低精度の傾向）
- [ ] グレード別精度改善（G2: Top1=16.7%と低い → 高グレード用の特徴量追加検討）
- [ ] 出走表取得時の発走時刻を全場で取得（現在は一部のみ）
- [ ] 結果の自動取得・精度レポート自動生成
- [ ] Supabase同期のAPI権限修正（現在service_key経由のみ）

## 進行中

## 完了
- [x] プロジェクト構造作成
- [x] DBスキーマ設計（races, riders, race_results, entries, predictions）
- [x] netkeirinスクレイパー実装・動作確認
- [x] /db/result/ で過去データアクセス可能を確認
- [x] /db/race_program/ で開催日一括取得の効率化
- [x] 特徴量設計・拡張（27→38特徴量）
- [x] LightGBM LambdaRankモデル
- [x] Git初期化・GitHub移行
- [x] 過去データ取得完了（2022-2025年 63,508レース、2,673選手）
- [x] スクレイピング並列化（ThreadPoolExecutor, 4ワーカー）
- [x] 特徴量ビルダーのバッチ最適化（N+1クエリ解消）
- [x] ライン情報パース実装（/race/entry/ DeployYosoセクション）
- [x] ライン補正パラメータ最適化（グリッドサーチ）
- [x] ハイパーパラメータチューニング（243組グリッドサーチ）
- [x] note.com記事生成テンプレート（Jinja2）
- [x] publishing モジュール（note_formatter.py, x_poster.py）
- [x] 記事生成スクリプト（scripts/generate_article.py）
- [x] weekly_pipeline.py（--predict / --result）
- [x] Supabase連携（PostgreSQL同期、prediction_detailビュー）
- [x] GitHub Pagesダッシュボード（バニラJS SPA）
- [x] VELODROME_CODES全43場の正確なマッピング検証・修正
- [x] 開催場自動検出（netkeirin トップページからrf=toptodayrace解析）
- [x] 4階層ナビゲーション（日付→会場→レース一覧→詳細）
- [x] リアルタイムステータス表示（LIVE/発走前/終了）
- [x] LIVE開催場優先ソート
- [x] 信頼度スコア最適化（2,799レース6ヶ月分析、10指標比較）
  - 旧方式 min(top3)-max(残): r=0.055, H-L差+6.8%（無相関）
  - 新方式 1位vs3位差: r=0.365, H-L差+35.4%
  - 高信頼(≥2.3): Top1的中率86.7%, Top1が3着以内97.9%

## メモ
- データソース: keirin.netkeiba.com (netkeirin)
- race_id形式: 12桁 = YYYYMMDD + jyo_cd(2) + race_num(2)
- 全43場対応（VELODROME_CODES、2025年3月検証済み）
- ライン補正: 3人ライン番手+0.30が最有利、最強ライン所属+0.20
- 特徴量重要度Top3: rider_class_num, rider_win_rate_all, rider_place_rate_all
- モデル精度（6ヶ月2,799R検証）: Top1=47.8%, Top1in3=78.8%, Top3avg=1.90/3
- 信頼度指標: conf_gap13（1位vs3位スコア差）、閾値 高≥2.3, 中≥0.85
- Supabase project: yzfoauumlocqdgxojwed
- GitHub Pages: https://haruqube.github.io/keirin-ai-predictor/
