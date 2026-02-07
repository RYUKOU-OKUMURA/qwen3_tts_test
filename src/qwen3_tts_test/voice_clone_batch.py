# 使用例:
# python3 voice_clone_batch.py \
#   --ref-audio myvoice.mp3 \
#   --ref-text-file myvoice_ref.txt \
#   --text-file input.txt \
#   --out out.wav \
#   --language Japanese

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .voice_clone_core import VoiceCloneError, generate_voice_waveform, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio", required=True, help="参照音声 (mp3/mp4/wav等) パス")
    ap.add_argument(
        "--ref-text-file",
        default=None,
        help="参照音声の文字起こしテキスト(UTF-8) パス（推奨）",
    )
    ap.add_argument(
        "--x-vector-only", action="store_true", help="ref_text無し（品質低下の可能性）"
    )
    ap.add_argument(
        "--text-file", required=True, help="読み上げたいテキスト(UTF-8) パス"
    )
    ap.add_argument("--out", required=True, help="出力 wav パス")
    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Voice Clone用 Base モデル",
    )
    ap.add_argument(
        "--language", default="Japanese", help="Japanese / English / auto など"
    )
    ap.add_argument("--silence", type=float, default=0.25, help="行間無音秒")
    args = ap.parse_args()

    log_file = Path(__file__).resolve().parent.parent / "logs" / "batch.log"
    logger = setup_logging(log_file)
    logger.info("バッチ処理開始")

    ref_audio = Path(args.ref_audio).expanduser().resolve()
    text_file = Path(args.text_file).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not ref_audio.exists():
        logger.error("参照音声が見つかりません: %s", ref_audio)
        raise FileNotFoundError(f"ref audio not found: {ref_audio}")
    if not text_file.exists():
        logger.error("テキストファイルが見つかりません: %s", text_file)
        raise FileNotFoundError(f"text file not found: {text_file}")
    if not out_path.parent.exists():
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("出力ディレクトリ作成: %s", out_path.parent)
        except Exception as exc:
            logger.error("ディレクトリ作成失敗: %s", exc)
            raise RuntimeError(f"出力ディレクトリを作成できません: {exc}")

    lines = [
        ln.strip()
        for ln in text_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not lines:
        logger.error("テキストファイルが空")
        raise RuntimeError("text_file が空です。")
    logger.info("処理行数: %d", len(lines))

    ref_text: str | None = None
    if not args.x_vector_only:
        if not args.ref_text_file:
            logger.error("ref_text_file未指定")
            raise RuntimeError(
                "ref_text が必要です。--ref-text-file を指定するか、--x-vector-only を付けてください。"
            )
        ref_text = (
            Path(args.ref_text_file).expanduser().read_text(encoding="utf-8").strip()
        )
        if not ref_text:
            logger.error("ref_text_fileが空")
            raise RuntimeError("ref_text_file が空です。")
        logger.info("参照テキスト: %d文字", len(ref_text))

    wav_list: list[np.ndarray] = []
    out_sr = 0

    for i, line in enumerate(lines, 1):
        logger.info("行 %d/%d 処理中: %.50s...", i, len(lines), line)
        wav, sr = generate_voice_waveform(
            ref_audio_path=str(ref_audio),
            ref_text=ref_text,
            input_text=line,
            language=args.language,
            model_id=args.model,
            x_vector_only_mode=bool(args.x_vector_only),
        )
        wav_list.append(wav)
        out_sr = sr

    sil = np.zeros(int(out_sr * args.silence), dtype=np.float32)
    out_wav: list[np.ndarray] = []
    for i, w in enumerate(wav_list):
        out_wav.append(w.astype(np.float32, copy=False))
        if i != len(wav_list) - 1:
            out_wav.append(sil)
    merged = (
        np.concatenate(out_wav, axis=0) if out_wav else np.zeros((0,), dtype=np.float32)
    )

    sf.write(str(out_path), merged, out_sr)
    logger.info("保存完了: %s sr=%d lines=%d", out_path, out_sr, len(lines))


if __name__ == "__main__":
    try:
        main()
    except VoiceCloneError as err:
        logging.getLogger("qwen_tts").error("VoiceCloneError: %s", err)
        sys.exit(1)
    except Exception as err:
        logging.getLogger("qwen_tts").exception("予期しないエラー")
        sys.exit(1)
