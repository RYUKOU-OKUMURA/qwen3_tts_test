#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo ".venv が見つかりません。先に ./setup_mac.sh を実行してください。"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
exec python app_gradio.py
