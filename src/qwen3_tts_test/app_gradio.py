from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import torch

from .voice_clone_core import (
    GradioLogHandler,
    setup_logging,
    synthesize_voice_clone,
    validate_required_inputs,
)

# MPSチェックはvoice_clone_coreで実行済み（インポート時にエラー終了）

MODEL_QUALITY = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_SPEED = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

MODEL_PRESETS = {
    "品質重視 (1.7B-Base)": MODEL_QUALITY,
    "速度重視 (0.6B-Base)": MODEL_SPEED,
    "カスタム入力": "__custom__",
}

DEFAULT_MODEL = MODEL_QUALITY

# Use environment variable or current working directory as base
_BASE_DIR = Path(os.environ.get("APP_BASE_DIR", Path.cwd())).resolve()
DEFAULT_OUTPUT_DIR = str(_BASE_DIR / "outputs")
LOG_FILE = _BASE_DIR / "logs" / "app.log"

# Ensure log directory exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

STEPS = ["入力チェック", "モデル読み込み", "音声生成", "ファイル保存"]


def run_generation(
    ref_audio_path: str | None,
    ref_text: str,
    input_text: str,
    language: str,
    output_dir: str,
    model_id: str,
    use_float32: bool,
) -> Any:
    logger = setup_logging(LOG_FILE)
    gradio_handler = GradioLogHandler()
    gradio_handler.setLevel(logging.INFO)
    gradio_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(gradio_handler)
    gradio_handler.clear()

    current_step = 0
    status_text = "待機中"

    def flush(audio_path: str | None = None, out_path: str = "", enable_button: bool = True) -> Any:
        step_display = f"[{current_step}/{len(STEPS)}] {STEPS[current_step - 1] if current_step > 0 else ''}".strip()
        return (
            gradio_handler.get_logs(),
            audio_path,
            out_path,
            gr.update(interactive=enable_button),
            step_display,
            status_text,
        )

    current_step, status_text = 1, "入力チェック中..."
    logger.info("入力チェック開始")
    yield flush(enable_button=False)

    if errors := validate_required_inputs(ref_audio_path, ref_text, input_text, output_dir):
        current_step, status_text = 0, "入力エラー"
        logger.warning("入力エラー: %s", errors)
        yield flush(enable_button=True)
        logger.removeHandler(gradio_handler)
        return

    out_dir = Path(output_dir).expanduser().resolve()
    if not out_dir.exists():
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            current_step, status_text = 0, "ディレクトリ作成失敗"
            logger.error("ディレクトリ作成失敗: %s", exc)
            yield flush(enable_button=True)
            logger.removeHandler(gradio_handler)
            return

    try:
        test_file = out_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        current_step, status_text = 0, "書き込み権限エラー"
        logger.error("書き込み権限がありません")
        yield flush(enable_button=True)
        logger.removeHandler(gradio_handler)
        return

    current_step, status_text = 2, "モデル読み込み中..."
    logger.info("モデル読み込み開始")
    yield flush(enable_button=False)

    result_holder: dict[str, dict[str, Any]] = {}
    error_holder: dict[str, Exception] = {}
    done = threading.Event()

    dtype = torch.float32 if use_float32 else torch.float16

    def worker() -> None:
        try:
            result_holder["value"] = synthesize_voice_clone(
                ref_audio_path=ref_audio_path or "",
                ref_text=ref_text,
                input_text=input_text,
                output_dir=output_dir,
                language=language,
                model_id=model_id,
                dtype=dtype,
            )
        except Exception as exc:
            error_holder["value"] = exc
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    current_step, status_text = 3, "音声生成中..."
    yield flush(enable_button=False)

    done.wait()

    if "value" in error_holder:
        current_step, status_text = 0, "生成失敗"
        logger.error("エラー: %s", error_holder["value"])
        yield flush(enable_button=True)
        logger.removeHandler(gradio_handler)
        return

    result = result_holder["value"]
    logger.info("%s", result["message"])

    if result["ok"]:
        current_step, status_text = 4, "完了"
        logger.info("処理完了")
        yield flush(
            audio_path=str(result["output_path"]),
            out_path=str(result["output_path"]),
            enable_button=True,
        )
    else:
        current_step, status_text = 0, "生成失敗"
        logger.error("生成失敗")
        yield flush(enable_button=True)

    logger.removeHandler(gradio_handler)


def apply_model_preset(preset_label: str, current_model_id: str) -> Any:
    preset_model = MODEL_PRESETS.get(preset_label, "__custom__")
    if preset_model == "__custom__":
        return gr.update(value=current_model_id.strip() or DEFAULT_MODEL, interactive=True)
    return gr.update(value=preset_model, interactive=False)


def build_ui() -> gr.Blocks:
    demo: gr.Blocks
    with gr.Blocks(title="Voice Clone GUI (macOS)") as demo:
        gr.Markdown("## Voice Clone GUI (macOS)")
        gr.Markdown(
            "参照音声 + 参照テキスト + 読み上げテキストから、1つの音声ファイルを生成します。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                ref_audio = gr.Audio(
                    label="参照音声ファイル（必須）",
                    type="filepath",
                    sources=["upload"],
                )
                ref_text = gr.Textbox(
                    label="参照文字起こし ref_text（必須）",
                    lines=4,
                    placeholder="参照音声の内容を入力してください",
                )
                input_text = gr.Textbox(
                    label="読み上げテキスト（必須）",
                    lines=8,
                    placeholder="ここに読み上げたい文章を入力",
                )

            with gr.Column(scale=1):
                language = gr.Dropdown(
                    label="言語",
                    choices=["Japanese", "English", "auto"],
                    value="Japanese",
                )
                output_dir = gr.Textbox(label="保存先ディレクトリ", value=DEFAULT_OUTPUT_DIR)
                with gr.Accordion("詳細設定", open=False):
                    model_preset = gr.Dropdown(
                        label="モデルプリセット",
                        choices=list(MODEL_PRESETS.keys()),
                        value="品質重視 (1.7B-Base)",
                    )
                    model_id = gr.Textbox(label="モデルID", value=DEFAULT_MODEL, interactive=False)
                    use_float32 = gr.Checkbox(
                        label="float32を使用（推奨）",
                        value=True,
                        info="float16はMPSでエラーが出ることがあります。メモリ節約したい場合のみfloat16を試してください。",
                    )
                    gr.Markdown("モデルを自由入力する場合は `カスタム入力` を選択。")

                run_button = gr.Button("音声を生成", variant="primary")
                step_display = gr.Textbox(label="進捗", value="待機中", interactive=False)
                status_box = gr.Textbox(label="処理ステータス", value="待機中", interactive=False)
                log_box = gr.Textbox(label="実行ログ", lines=14, interactive=False)

        output_audio = gr.Audio(label="生成音声", type="filepath", interactive=False)
        output_path = gr.Textbox(label="出力ファイル", interactive=False)

        run_button.click(
            fn=run_generation,
            inputs=[
                ref_audio,
                ref_text,
                input_text,
                language,
                output_dir,
                model_id,
                use_float32,
            ],
            outputs=[
                log_box,
                output_audio,
                output_path,
                run_button,
                step_display,
                status_box,
            ],
        )
        model_preset.change(
            fn=apply_model_preset, inputs=[model_preset, model_id], outputs=[model_id]
        )

    return demo


def main() -> None:
    setup_logging(LOG_FILE)
    app = build_ui()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
