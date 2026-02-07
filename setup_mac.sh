#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "[1/5] macOS と Python を確認中..."
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "このスクリプトは macOS 向けです。"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 が見つかりません。"
  echo "Xcode Command Line Tools か Homebrew の Python を導入してください。"
  exit 1
fi

echo "[2/5] ffmpeg を確認中..."
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg が見つかりません。"
  echo "Homebrew を使って次を実行してください:"
  echo "  brew install ffmpeg"
  exit 1
fi

echo "[3/5] 仮想環境(.venv)を作成/更新中..."
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[4/5] pip と依存ライブラリをインストール中..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "[5/5] セットアップ完了"
echo "次は ./start_gui.command をダブルクリックして起動してください。"
