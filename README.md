# Qwen3 TTS Voice Clone

ローカルで動作する音声クローンGUI（uv管理）。

## セットアップ

```bash
# uvがなければインストール
brew install uv
# ffmpegがなければインストール
brew install ffmpeg
# 依存インストール
uv sync

# 開発用ツール（ruff, mypy）を含める場合
uv sync --all-extras
```

## 起動

```bash
# 推奨: エントリポイントを使用
uv run app

# または直接スクリプトを指定する場合
uv run python src/qwen3_tts_test/app_gradio.py
```

## 使い方

1. 参照音声ファイルを選択
2. 参照音声の文字起こし（ref_text）を入力
3. 読み上げたいテキストを入力
4. 「生成」ボタンをクリック

生成された音声は `outputs/` に保存されます。

## CLI使用例

```bash
uv run python src/qwen3_tts_test/voice_clone_batch.py \
  --ref-audio myvoice.mp3 \
  --ref-text-file myvoice_ref.txt \
  --text-file input.txt \
  --out out.wav
```

## 開発

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy src/
```

## 必要条件

- Python 3.10+
- macOS（MPS推奨）
- ffmpeg: `brew install ffmpeg`
