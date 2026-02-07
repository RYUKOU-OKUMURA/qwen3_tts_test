# Voice Clone GUI (macOS)

Python や長いコマンドを意識せず使えるようにした、ローカル専用の音声クローンGUIです。

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

生成後、音声プレイヤーで再生できます。ファイル名は `voiceclone_YYYYmmdd_HHMMSS.wav` で自動保存されます。

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
