# Voice Clone GUI (macOS)

Python や長いコマンドを意識せず使えるようにした、ローカル専用の音声クローンGUIです。

## 最短セットアップ（コピペ1本）

以下をそのまま実行すると、`ffmpeg` 導入・依存インストール・GUI起動まで一気に実行できます。

```bash
cd /Users/ryukouokumura/Desktop/boss-workspace/qwen3_tts_test && \
command -v brew >/dev/null || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" && \
brew install ffmpeg && \
chmod +x setup_mac.sh start_gui.sh start_gui.command && \
./setup_mac.sh && \
./start_gui.sh
```

## 使い方（3ステップ）

1. 初回セットアップ
```bash
./setup_mac.sh
```

2. GUIを起動
- Finder から `start_gui.command` をダブルクリック
- もしくはターミナルで:
```bash
./start_gui.sh
```

3. 画面で入力して生成
- 参照音声ファイル（必須）
- 参照文字起こし `ref_text`（必須）
- 読み上げテキスト（必須）
- 保存先ディレクトリ（デフォルト: `./outputs`）
- モデルプリセット（`品質重視` / `速度重視` / `CustomVoice` / `カスタム入力`）
- 進捗バーと処理ステータス（生成中の目安を表示）
- デバイス（macOSは `mps` 推奨。`mps` 利用不可時は明示エラー）

生成後、音声プレイヤーで再生できます。ファイル名は `voiceclone_YYYYmmdd_HHMMSS.wav` で自動保存されます。

## モデルについて（重要）

- 本ツールは `qwen_tts` 経由で **Qwen3-TTS をローカル推論** で使います。
- 初回実行時にモデルがローカルキャッシュへダウンロードされる場合があります。
- 完全ローカル運用したい場合は、事前にモデルを配置し、GUIの「詳細設定 > モデルID」またはCLIの `--model` にローカルパスを指定してください。

## よくあるエラー

- `ffmpeg が見つかりません`
  - `brew install ffmpeg` を実行
- 依存ライブラリ不足
  - `./setup_mac.sh` を再実行
- `ref_text` 未入力
  - 参照音声の文字起こしを入力して再実行

## CLI互換（従来スクリプト）

従来の `voice_clone_batch.py` も引き続き利用できます。

```bash
python3 voice_clone_batch.py \
  --ref-audio myvoice.mp3 \
  --ref-text-file myvoice_ref.txt \
  --text-file input.txt \
  --out out.wav \
  --language Japanese
```
