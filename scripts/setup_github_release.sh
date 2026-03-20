#!/bin/bash
# ============================================================
# GitHub Release にDB+モデルをアップロード（初回セットアップ）
#
# 使い方:
#   bash scripts/setup_github_release.sh
#
# 前提: gh CLI がインストール済み & 認証済み
# ============================================================

set -e

REPO="haruqube/keirin-ai-predictor"
TAG="data-latest"
DB_PATH="db/keirin.db"
MODEL_PATH="results/model_lgbm.pkl"

echo "=== GitHub Release セットアップ ==="

# ファイル存在チェック
for f in "$DB_PATH" "$MODEL_PATH"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f が見つかりません"
    exit 1
  fi
done

echo "DB:    $(ls -lh $DB_PATH | awk '{print $5}')"
echo "Model: $(ls -lh $MODEL_PATH | awk '{print $5}')"

# リリースが既に存在するか確認
if gh release view "$TAG" --repo "$REPO" > /dev/null 2>&1; then
  echo "Release '$TAG' は既に存在します。アセットを更新します..."
  gh release upload "$TAG" "$DB_PATH" "$MODEL_PATH" --repo "$REPO" --clobber
else
  echo "Release '$TAG' を新規作成します..."
  gh release create "$TAG" "$DB_PATH" "$MODEL_PATH" \
    --repo "$REPO" \
    --title "Data Assets (自動更新)" \
    --notes "DB + モデルファイル。GitHub Actions から自動更新されます。"
fi

echo "=== 完了 ==="
echo ""
echo "次のステップ:"
echo "  1. GitHub Settings > Secrets に以下を設定:"
echo "     - SUPABASE_URL"
echo "     - SUPABASE_SERVICE_KEY"
echo "     - SUPABASE_ANON_KEY"
echo "  2. push してワークフローを有効化"
