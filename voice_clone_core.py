from __future__ import annotations

import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_RUNTIME: dict[tuple[str, str], tuple[Any, str, str]] = {}


class VoiceCloneError(RuntimeError):
    """User-facing errors for voice clone operations."""


def _run_ffmpeg_to_wav(in_path: Path, out_path: Path, sr: int = 16000) -> None:
    if shutil.which("ffmpeg") is None:
        raise VoiceCloneError("ffmpeg が見つかりません。`brew install ffmpeg` を実行してください。")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        str(out_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        err = (p.stderr or "").strip().splitlines()
        detail = err[-1] if err else "不明なエラー"
        raise VoiceCloneError(f"ffmpeg 変換に失敗しました: {detail}")


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_runtime_params(device: str) -> tuple[Any, str]:
    try:
        import torch
    except Exception as exc:
        raise VoiceCloneError(f"PyTorch の読み込みに失敗しました: {exc}") from exc

    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return dtype, "flash_attention_2"
    if device == "mps":
        return torch.float16, "eager"
    return torch.float32, "eager"


def _load_model(model_id: str, device: str) -> tuple[Any, str, Any]:
    key = (model_id, device)
    if key in _MODEL_CACHE and key in _MODEL_RUNTIME:
        dtype, attn_impl, _ = _MODEL_RUNTIME[key]
        return _MODEL_CACHE[key], attn_impl, dtype

    try:
        from qwen_tts import Qwen3TTSModel
    except Exception as exc:
        raise VoiceCloneError(
            "qwen_tts が読み込めません。セットアップ後に再実行してください。"
        ) from exc

    dtype, attn_impl = resolve_runtime_params(device)

    try:
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
    except Exception as first_exc:
        if device.startswith("cuda") and attn_impl == "flash_attention_2":
            try:
                model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=device,
                    dtype=dtype,
                    attn_implementation="eager",
                )
                attn_impl = "eager"
            except Exception as second_exc:
                raise VoiceCloneError(f"モデル読み込みに失敗しました: {second_exc}") from second_exc
        else:
            raise VoiceCloneError(f"モデル読み込みに失敗しました: {first_exc}") from first_exc

    _MODEL_CACHE[key] = model
    _MODEL_RUNTIME[key] = (dtype, attn_impl, device)
    return model, attn_impl, dtype


def _build_output_path(output_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"voiceclone_{ts}.wav"


def preflight_check() -> list[str]:
    issues: list[str] = []
    if shutil.which("ffmpeg") is None:
        issues.append("ffmpeg が見つかりません（`brew install ffmpeg` を実行してください）。")

    try:
        import torch  # noqa: F401
    except Exception:
        issues.append("PyTorch が見つかりません（`./setup_mac.sh` を実行してください）。")

    try:
        import qwen_tts  # noqa: F401
    except Exception:
        issues.append("qwen_tts が見つかりません（`./setup_mac.sh` を実行してください）。")

    try:
        import soundfile  # noqa: F401
    except Exception:
        issues.append("soundfile が見つかりません（`./setup_mac.sh` を実行してください）。")

    return issues


def validate_required_inputs(
    ref_audio_path: str | None,
    ref_text: str | None,
    input_text: str | None,
    output_dir: str | None,
) -> list[str]:
    errors: list[str] = []

    if not ref_audio_path:
        errors.append("参照音声ファイルを選択してください。")
    elif not Path(ref_audio_path).expanduser().exists():
        errors.append("参照音声ファイルが見つかりません。")

    if not (ref_text or "").strip():
        errors.append("参照文字起こし（ref_text）を入力してください。")

    if not (input_text or "").strip():
        errors.append("読み上げテキストを入力してください。")

    if not output_dir:
        errors.append("保存先ディレクトリを指定してください。")

    return errors


def generate_voice_waveform(
    ref_audio_path: str,
    ref_text: str | None,
    input_text: str,
    language: str = "Japanese",
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device: str = "auto",
    x_vector_only_mode: bool = False,
) -> tuple[Any, int, dict[str, str]]:
    ref_audio = Path(ref_audio_path).expanduser().resolve()
    if not ref_audio.exists():
        raise VoiceCloneError("参照音声ファイルが見つかりません。")

    if not input_text.strip():
        raise VoiceCloneError("読み上げテキストが空です。")

    if not x_vector_only_mode and not (ref_text or "").strip():
        raise VoiceCloneError("ref_text が必要です。参照音声の文字起こしを入力してください。")

    try:
        import numpy as np
    except Exception as exc:
        raise VoiceCloneError(f"numpy の読み込みに失敗しました: {exc}") from exc

    actual_device = resolve_device(device)
    model, attn_impl, dtype = _load_model(model_id, actual_device)

    with tempfile.TemporaryDirectory() as td:
        ref_wav = Path(td) / "ref.wav"
        _run_ffmpeg_to_wav(ref_audio, ref_wav, sr=16000)

        wavs, sample_rate = model.generate_voice_clone(
            text=input_text,
            language=language,
            ref_audio=str(ref_wav),
            ref_text=(ref_text or None),
            x_vector_only_mode=bool(x_vector_only_mode),
        )

    if not wavs:
        raise VoiceCloneError("音声生成結果が空でした。入力テキストを見直してください。")

    wav = np.asarray(wavs[0], dtype=np.float32)
    runtime = {
        "device": actual_device,
        "attn": str(attn_impl),
        "dtype": str(dtype),
    }
    return wav, int(sample_rate), runtime


def synthesize_voice_clone(
    ref_audio_path: str,
    ref_text: str,
    input_text: str,
    output_dir: str,
    language: str = "Japanese",
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device: str = "auto",
) -> dict[str, object]:
    errors = validate_required_inputs(ref_audio_path, ref_text, input_text, output_dir)
    if errors:
        return {
            "ok": False,
            "output_path": None,
            "sample_rate": None,
            "message": " / ".join(errors),
        }

    try:
        import soundfile as sf
    except Exception as exc:
        return {
            "ok": False,
            "output_path": None,
            "sample_rate": None,
            "message": f"soundfile の読み込みに失敗しました: {exc}",
        }

    out_dir = Path(output_dir).expanduser().resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return {
            "ok": False,
            "output_path": None,
            "sample_rate": None,
            "message": f"保存先ディレクトリを作成できません: {exc}",
        }

    out_path = _build_output_path(out_dir)

    try:
        wav, sample_rate, runtime = generate_voice_waveform(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            input_text=input_text,
            language=language,
            model_id=model_id,
            device=device,
            x_vector_only_mode=False,
        )
    except VoiceCloneError as exc:
        return {
            "ok": False,
            "output_path": None,
            "sample_rate": None,
            "message": str(exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "output_path": None,
            "sample_rate": None,
            "message": f"予期しないエラーが発生しました: {exc}",
        }

    try:
        sf.write(str(out_path), wav, sample_rate)
    except PermissionError as exc:
        return {
            "ok": False,
            "output_path": None,
            "sample_rate": None,
            "message": f"保存に失敗しました。保存先の書き込み権限を確認してください: {exc}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "output_path": None,
            "sample_rate": None,
            "message": f"音声ファイルの保存に失敗しました: {exc}",
        }

    return {
        "ok": True,
        "output_path": str(out_path),
        "sample_rate": sample_rate,
        "message": (
            f"保存しました: {out_path} "
            f"(sr={sample_rate}, device={runtime['device']}, attn={runtime['attn']}, dtype={runtime['dtype']})"
        ),
    }
