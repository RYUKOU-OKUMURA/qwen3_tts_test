#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "[1/6] macOS と Python を確認中..."
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "このスクリプトは macOS 向けです。"
  exit 1
fi

if [[ "$(sysctl -in sysctl.proc_translated 2>/dev/null || echo 0)" == "1" ]]; then
  echo "Rosetta上で実行されています。GPU(MPS)が不安定になるため通常のTerminalで実行してください。"
  exit 1
fi

PYTHON_CMD=""
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python3.12)"
elif [[ -x "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12" ]]; then
  PYTHON_CMD="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python3)"
fi

if [[ -z "$PYTHON_CMD" ]]; then
  echo "Python が見つかりません。"
  echo "Homebrew で次を実行してください: brew install python@3.12"
  exit 1
fi

PY_VERSION="$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VERSION" != "3.12" ]]; then
  echo "選択されたPythonは $PY_VERSION です。GPU(MPS)安定動作のため Python 3.12 を推奨します。"
  echo "Homebrew で次を実行してください: brew install python@3.12"
  exit 1
fi

echo "使用Python: $PYTHON_CMD ($PY_VERSION)"

echo "[2/6] ffmpeg を確認中..."
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg が見つかりません。"
  echo "Homebrew を使って次を実行してください:"
  echo "  brew install ffmpeg"
  exit 1
fi

echo "[3/6] 仮想環境(.venv)を作成/更新中..."
if [[ -x ".venv/bin/python" ]]; then
  VENV_PY_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$VENV_PY_VERSION" != "3.12" ]]; then
    echo "既存 .venv は Python $VENV_PY_VERSION のため再作成します。"
    rm -rf .venv
  fi
fi

if [[ ! -d ".venv" ]]; then
  "$PYTHON_CMD" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[4/6] pip と依存ライブラリをインストール中..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "[5/6] GPU(MPS) 利用可否を確認中..."
python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}")
print(f"mps_built={torch.backends.mps.is_built()}")
print(f"mps_available={torch.backends.mps.is_available()}")

if not torch.backends.mps.is_available():
    print("")
    print("MPS(GPU)が利用できません。CPU推論になってしまうため停止します。")
    print("Python 3.12 のarm64環境で再セットアップしてください。")
    print("必要なら: rm -rf .venv && ./setup_mac.sh")
    sys.exit(1)
PY

echo "[6/6] セットアップ完了"
echo "次は ./start_gui.command をダブルクリックして起動してください。"
