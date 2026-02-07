from __future__ import annotations

import threading
from pathlib import Path

import gradio as gr

from voice_clone_core import preflight_check, synthesize_voice_clone, validate_required_inputs

MODEL_QUALITY = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_SPEED = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_CUSTOM_VOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

MODEL_PRESETS = {
    "品質重視 (1.7B-Base)": MODEL_QUALITY,
    "速度重視 (0.6B-Base)": MODEL_SPEED,
    "CustomVoice (1.7B-CustomVoice)": MODEL_CUSTOM_VOICE,
    "カスタム入力": "__custom__",
}

DEFAULT_MODEL = MODEL_QUALITY
DEFAULT_OUTPUT_DIR = str((Path(__file__).resolve().parent / "outputs").resolve())


def run_generation(
    ref_audio_path: str | None,
    ref_text: str,
    input_text: str,
    language: str,
    output_dir: str,
    model_id: str,
    device: str,
):
    logs: list[str] = []
    progress_value = 0
    status_text = "待機中"

    def flush(audio_path: str | None = None, out_path: str = "", enable_button: bool = True):
        return (
            "\n".join(logs),
            audio_path,
            out_path,
            gr.update(interactive=enable_button),
            progress_value,
            status_text,
        )

    progress_value = 5
    status_text = "入力チェック中..."
    logs.append("入力チェック中...")
    yield flush(enable_button=False)

    issues = preflight_check()
    if issues:
        progress_value = 0
        status_text = "事前チェック失敗"
        logs.append("事前チェックで問題が見つかりました。")
        logs.extend([f"- {issue}" for issue in issues])
        yield flush(enable_button=True)
        return

    errors = validate_required_inputs(ref_audio_path, ref_text, input_text, output_dir)
    if errors:
        progress_value = 0
        status_text = "入力エラー"
        logs.append("入力エラーがあります。")
        logs.extend([f"- {err}" for err in errors])
        yield flush(enable_button=True)
        return

    progress_value = 12
    status_text = "生成準備中..."
    logs.append("音声生成を開始します。モデル初回読み込み時は時間がかかることがあります。")
    yield flush(enable_button=False)

    result_holder: dict[str, dict[str, object]] = {}
    error_holder: dict[str, Exception] = {}
    done = threading.Event()

    def worker() -> None:
        try:
            result_holder["value"] = synthesize_voice_clone(
                ref_audio_path=ref_audio_path or "",
                ref_text=ref_text,
                input_text=input_text,
                output_dir=output_dir,
                language=language,
                model_id=model_id,
                device=device,
            )
        except Exception as exc:
            error_holder["value"] = exc
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    progress_value = 20
    status_text = "生成中..."
    yield flush(enable_button=False)

    while not done.wait(timeout=0.8):
        progress_value = min(progress_value + 2, 92)
        status_text = f"生成中... {progress_value}%"
        yield flush(enable_button=False)

    if "value" in error_holder:
        progress_value = 0
        status_text = "生成失敗"
        logs.append(f"予期しないエラーが発生しました: {error_holder['value']}")
        yield flush(enable_button=True)
        return

    result = result_holder["value"]

    logs.append(result["message"])
    if result["ok"]:
        out_path = str(result["output_path"])
        progress_value = 100
        status_text = "完了"
        logs.append("完了しました。下のプレイヤーで確認できます。")
        yield flush(audio_path=out_path, out_path=out_path, enable_button=True)
    else:
        progress_value = 0
        status_text = "生成失敗"
        logs.append("失敗しました。上記メッセージを確認してください。")
        yield flush(enable_button=True)


def apply_model_preset(preset_label: str, current_model_id: str):
    preset_model = MODEL_PRESETS.get(preset_label, "__custom__")
    if preset_model == "__custom__":
        model_value = current_model_id.strip() or DEFAULT_MODEL
        return gr.update(value=model_value, interactive=True)
    return gr.update(value=preset_model, interactive=False)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Voice Clone GUI (macOS)") as demo:
        gr.Markdown("## Voice Clone GUI (macOS)")
        gr.Markdown("参照音声 + 参照テキスト + 読み上げテキストから、1つの音声ファイルを生成します。")

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
                output_dir = gr.Textbox(
                    label="保存先ディレクトリ",
                    value=DEFAULT_OUTPUT_DIR,
                )
                with gr.Accordion("詳細設定", open=False):
                    model_preset = gr.Dropdown(
                        label="モデルプリセット",
                        choices=list(MODEL_PRESETS.keys()),
                        value="品質重視 (1.7B-Base)",
                    )
                    model_id = gr.Textbox(label="モデルID", value=DEFAULT_MODEL, interactive=False)
                    device = gr.Dropdown(
                        label="デバイス",
                        choices=["auto", "mps", "cpu", "cuda:0"],
                        value="mps",
                    )
                    gr.Markdown("モデルを自由入力したい場合はプリセットを `カスタム入力` に切り替えてください。")
                    gr.Markdown("macOSでは `mps` 推奨です。`mps` 選択時はCPUへ自動フォールバックせず、利用不可ならエラー表示します。")

                run_button = gr.Button("音声を生成", variant="primary")
                progress_bar = gr.Slider(label="進捗 (%)", minimum=0, maximum=100, step=1, value=0, interactive=False)
                status_box = gr.Textbox(label="処理ステータス", value="待機中", interactive=False)
                log_box = gr.Textbox(label="実行ログ", lines=14, interactive=False)

        output_audio = gr.Audio(label="生成音声", type="filepath", interactive=False)
        output_path = gr.Textbox(label="出力ファイル", interactive=False)

        run_button.click(
            fn=run_generation,
            inputs=[ref_audio, ref_text, input_text, language, output_dir, model_id, device],
            outputs=[log_box, output_audio, output_path, run_button, progress_bar, status_box],
        )
        model_preset.change(
            fn=apply_model_preset,
            inputs=[model_preset, model_id],
            outputs=[model_id],
        )

    return demo


def main() -> None:
    app = build_ui()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
