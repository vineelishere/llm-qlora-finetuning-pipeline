import sys
import os
import time

import gradio as gr
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "inference"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))

from inference_engine import UnifiedInferenceEngine
from logger import get_logger

logger = get_logger(__name__)

MODEL_TYPE = os.environ.get("MODEL_TYPE", "llm")
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/merged_model")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "true").lower() == "true"

logger.info(f"Loading {MODEL_TYPE} model from {MODEL_PATH} for demo...")

try:
    engine = UnifiedInferenceEngine(
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    logger.info("Model loaded successfully for demo.")
    MODEL_LOAD_ERROR = None
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    engine = None
    MODEL_LOAD_ERROR = str(e)


def generate_llm_response(prompt: str, max_new_tokens: int, temperature: float) -> str:
    if engine is None:
        return f"Error: the model could not be loaded. Details: {MODEL_LOAD_ERROR}"

    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    start_time = time.time()
    response = engine.generate(
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
    )
    latency = time.time() - start_time

    return f"{response}\n\n---\n*Generated in {latency:.2f}s*"


def generate_vlm_response(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    if engine is None:
        return f"Error: the model could not be loaded. Details: {MODEL_LOAD_ERROR}"

    if image is None:
        return "Please upload an image."

    if not prompt or not prompt.strip():
        return "Please write a question about the image."

    start_time = time.time()
    response = engine.generate(
        image=image,
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
    )
    latency = time.time() - start_time

    return f"{response}\n\n---\n*Generated in {latency:.2f}s*"

# ==========================================================
# Gradio UI Layout
# ==========================================================

with gr.Blocks(title="QLoRA Fine-Tuning & Quantization Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # QLoRA Fine-Tuning & Quantization Demo
        Chat with either a QLoRA fine-tuned LLM or a vision-language model.
        Set `MODEL_TYPE` and `MODEL_PATH` before launching to choose which checkpoint is served.
        """
    )

    with gr.Tab("💬 Text Generation (LLM)"):
        with gr.Row():
            with gr.Column(scale=1):
                llm_prompt_input = gr.Textbox(
                    label="Your Prompt",
                    placeholder="e.g. Explain how neural networks work in simple terms.",
                    lines=4,
                )
                llm_max_tokens = gr.Slider(
                    minimum=16, maximum=512, value=200, step=16,
                    label="Max New Tokens",
                )
                llm_temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature (higher = more creative)",
                )
                llm_submit_btn = gr.Button("Generate Response", variant="primary")

            with gr.Column(scale=1):
                llm_output = gr.Textbox(label="Model Response", lines=12)

        llm_submit_btn.click(
            fn=generate_llm_response,
            inputs=[llm_prompt_input, llm_max_tokens, llm_temperature],
            outputs=llm_output,
        )

        gr.Examples(
            examples=[
                ["What is the difference between supervised and unsupervised learning?"],
                ["Write a short poem about the ocean."],
                ["Explain QLoRA fine-tuning in simple terms."],
            ],
            inputs=llm_prompt_input,
        )

    with gr.Tab("🖼️ Vision + Text (VLM)"):
        with gr.Row():
            with gr.Column(scale=1):
                vlm_image_input = gr.Image(label="Upload Image", type="pil")
                vlm_prompt_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. What is happening in this image?",
                    lines=2,
                )
                vlm_max_tokens = gr.Slider(
                    minimum=16, maximum=512, value=150, step=16,
                    label="Max New Tokens",
                )
                vlm_submit_btn = gr.Button("Generate Answer", variant="primary")

            with gr.Column(scale=1):
                vlm_output = gr.Textbox(label="Model Response", lines=12)

        vlm_submit_btn.click(
            fn=generate_vlm_response,
            inputs=[vlm_image_input, vlm_prompt_input, vlm_max_tokens],
            outputs=vlm_output,
        )

    gr.Markdown(
        """
        ---
        **Pipeline Features:** QLoRA Fine-Tuning • Multi-format Quantization (bitsandbytes/GGUF/AWQ) • FastAPI Serving • Docker Deployment
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

