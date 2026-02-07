from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel

# MPS必須チェック - 起動時に検証
if not torch.backends.mps.is_available():
    print("エラー: MPSが利用できません。Apple Silicon Macが必要です。", file=sys.stderr)
    sys.exit(1)

_MODEL_CACHE: dict[str, Any] = {}
_logger = logging.getLogger("qwen_tts")


class VoiceCloneError(RuntimeError):
    """音声クローン操作でユーザーに表示されるエラー"""


class GradioLogHandler(logging.Handler):
    """Gradio UI表示用にログをキャプチャするハンドラ"""

    def __init__(self) -> None:
        super().__init__()
        self.logs: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.logs.append(self.format(record))

    def get_logs(self) -> str:
        return "\n".join(self.logs)

    def clear(self) -> None:
        self.logs.clear()


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("qwen_tts")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


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
    x_vector_only_mode: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple[Any, int]:
    _logger.info("音声生成開始: model=%s, language=%s", model_id, language)

    ref_audio = Path(ref_audio_path).expanduser().resolve()
    if not ref_audio.exists():
        _logger.error("参照音声ファイルが見つかりません: %s", ref_audio)
        raise VoiceCloneError("参照音声ファイルが見つかりません。")
    if not input_text.strip():
        _logger.error("読み上げテキストが空です")
        raise VoiceCloneError("読み上げテキストが空です。")
    if not x_vector_only_mode and not (ref_text or "").strip():
        _logger.error("ref_text が未入力")
        raise VoiceCloneError(
            "ref_text が必要です。参照音声の文字起こしを入力してください。"
        )

    cache_key = f"{model_id}_{dtype}"
    if cache_key not in _MODEL_CACHE:
        _logger.info("モデル読み込み開始: %s (dtype=%s)", model_id, dtype)
        try:
            _MODEL_CACHE[cache_key] = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map="mps",
                dtype=dtype,
                attn_implementation="eager",
            )
            _logger.info("モデル読み込み完了: %s (dtype=%s)", model_id, dtype)
        except Exception as exc:
            _logger.error("モデル読み込み失敗: %s", exc)
            raise VoiceCloneError(f"モデル読み込みに失敗しました: {exc}") from exc
    model = _MODEL_CACHE[cache_key]

    with tempfile.TemporaryDirectory() as td:
        ref_wav = Path(td) / "ref.wav"
        try:
            AudioSegment.from_file(str(ref_audio)).set_channels(1).set_frame_rate(
                16000
            ).export(str(ref_wav), format="wav")
            _logger.info("音声変換完了: %s", ref_audio)
        except Exception as exc:
            _logger.error("音声変換失敗: %s", exc)
            raise VoiceCloneError(f"音声変換に失敗しました: {exc}") from exc

        try:
            _logger.info("音声生成実行中...")
            wavs, sample_rate = model.generate_voice_clone(
                text=input_text,
                language=language,
                ref_audio=str(ref_wav),
                ref_text=ref_text or None,
                x_vector_only_mode=bool(x_vector_only_mode),
            )
            _logger.info("音声生成完了: %d chunks, sr=%d", len(wavs), sample_rate)
        except RuntimeError as exc:
            _logger.error("音声生成失敗: %s", exc)
            raise VoiceCloneError(f"音声生成に失敗しました: {exc}") from exc

    if not wavs:
        _logger.error("音声生成結果が空")
        raise VoiceCloneError(
            "音声生成結果が空でした。入力テキストを見直してください。"
        )

    return np.asarray(wavs[0], dtype=np.float32), int(sample_rate)


def synthesize_voice_clone(
    ref_audio_path: str,
    ref_text: str,
    input_text: str,
    output_dir: str,
    language: str = "Japanese",
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    _logger.info("入力検証開始")

    if errors := validate_required_inputs(
        ref_audio_path, ref_text, input_text, output_dir
    ):
        _logger.warning("入力エラー: %s", errors)
        return {"ok": False, "message": " / ".join(errors)}

    out_dir = Path(output_dir).expanduser().resolve()
    if not out_dir.exists():
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            _logger.info("出力ディレクトリ作成: %s", out_dir)
        except Exception as exc:
            _logger.error("ディレクトリ作成失敗: %s", exc)
            return {
                "ok": False,
                "message": f"保存先ディレクトリを作成できません: {exc}",
            }

    if not out_dir.is_dir():
        return {"ok": False, "message": "保存先がディレクトリではありません。"}

    try:
        test_file = out_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        return {
            "ok": False,
            "message": "保存先ディレクトリへの書き込み権限がありません。",
        }

    out_path = out_dir / f"voiceclone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    _logger.info("出力ファイル: %s", out_path)

    try:
        wav, sample_rate = generate_voice_waveform(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            input_text=input_text,
            language=language,
            model_id=model_id,
            x_vector_only_mode=False,
            dtype=dtype,
        )
    except VoiceCloneError as exc:
        return {"ok": False, "message": str(exc)}
    except Exception as exc:
        _logger.exception("予期しないエラー")
        return {"ok": False, "message": f"予期しないエラー: {exc}"}

    try:
        sf.write(str(out_path), wav, sample_rate)
        _logger.info("ファイル保存完了: %s", out_path)
    except Exception as exc:
        _logger.error("ファイル保存失敗: %s", exc)
        return {"ok": False, "message": f"音声ファイル保存失敗: {exc}"}

    return {
        "ok": True,
        "output_path": str(out_path),
        "sample_rate": sample_rate,
        "message": f"保存しました: {out_path} (sr={sample_rate})",
    }
