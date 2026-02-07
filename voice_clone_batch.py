# python3 voice_clone_batch.py \
#   --ref-audio myvoice.mp3 \
#   --ref-text-file myvoice_ref.txt \
#   --text-file input.txt \
#   --out out.wav \
#   --language Japanese

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_clone_core import VoiceCloneError, generate_voice_waveform, preflight_check


def load_lines_utf8(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def concat_with_silence(wavs: list[np.ndarray], sr: int, silence_sec: float = 0.25) -> np.ndarray:
    sil = np.zeros(int(sr * silence_sec), dtype=np.float32)
    out = []
    for i, wav in enumerate(wavs):
        out.append(wav.astype(np.float32, copy=False))
        if i != len(wavs) - 1:
            out.append(sil)
    return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio", required=True, help="参照音声 (mp3/mp4/wav等) パス")
    ap.add_argument("--ref-text-file", default=None, help="参照音声の文字起こしテキスト(UTF-8) パス（推奨）")
    ap.add_argument("--x-vector-only", action="store_true", help="ref_text無し（品質低下の可能性）")
    ap.add_argument("--text-file", required=True, help="読み上げたいテキスト(UTF-8) パス")
    ap.add_argument("--out", required=True, help="出力 wav パス")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="Voice Clone用 Base モデル")
    ap.add_argument("--language", default="Japanese", help="Japanese / English / auto など")
    ap.add_argument("--device", default="auto", help="auto / mps / cpu / cuda:0")
    ap.add_argument("--silence", type=float, default=0.25, help="行間無音秒")
    args = ap.parse_args()

    issues = preflight_check()
    if issues:
        issue_msg = "\n".join(f"- {item}" for item in issues)
        raise RuntimeError(f"事前チェックに失敗しました。\n{issue_msg}")

    ref_audio = Path(args.ref_audio).expanduser().resolve()
    text_file = Path(args.text_file).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not ref_audio.exists():
        raise FileNotFoundError(f"ref audio not found: {ref_audio}")
    if not text_file.exists():
        raise FileNotFoundError(f"text file not found: {text_file}")

    ref_text: str | None = None
    if not args.x_vector_only:
        if not args.ref_text_file:
            raise RuntimeError("ref_text が必要です。--ref-text-file を指定するか、ref_text無しなら --x-vector-only を付けてください。")
        ref_text = Path(args.ref_text_file).expanduser().read_text(encoding="utf-8").strip()
        if not ref_text:
            raise RuntimeError("ref_text_file が空です。参照音声の文字起こしを入れてください。")

    lines = load_lines_utf8(text_file)
    if not lines:
        raise RuntimeError("text_file が空です。読み上げたいテキストを1行以上入れてください。")

    wav_list: list[np.ndarray] = []
    out_sr = 0
    runtime_info: dict[str, str] = {"device": args.device, "attn": "unknown", "dtype": "unknown"}

    for line in lines:
        wav, sr, runtime = generate_voice_waveform(
            ref_audio_path=str(ref_audio),
            ref_text=ref_text,
            input_text=line,
            language=args.language,
            model_id=args.model,
            device=args.device,
            x_vector_only_mode=bool(args.x_vector_only),
        )
        wav_list.append(wav)
        out_sr = sr
        runtime_info = runtime

    merged = concat_with_silence(wav_list, out_sr, silence_sec=args.silence)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), merged, out_sr)

    print(
        "saved: "
        f"{out_path} sr={out_sr} device={runtime_info['device']} "
        f"attn={runtime_info['attn']} dtype={runtime_info['dtype']} lines={len(lines)}"
    )


if __name__ == "__main__":
    try:
        main()
    except VoiceCloneError as err:
        print("[ERROR]", err, file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print("[ERROR]", err, file=sys.stderr)
        sys.exit(1)
