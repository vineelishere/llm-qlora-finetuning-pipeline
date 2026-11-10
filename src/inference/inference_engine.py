import sys
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # transformers >= 5 renamed the vision-to-text auto class
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
from PIL import Image
from typing import Optional, Union

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from logger import get_logger

logger = get_logger(__name__)


class LLMInferenceEngine:

    def __init__(self, model_path: str, load_in_4bit: bool = True, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading LLM from: {model_path} (device: {self.device})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if load_in_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=bnb_config, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16
            ).to(self.device)

        self.model.eval()
        logger.info("LLM loaded successfully and ready for inference.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        return generated_text.strip()
    
class VLMInferenceEngine:

    def __init__(self, model_path: str, load_in_4bit: bool = True, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading VLM from: {model_path} (device: {self.device})")

        self.processor = AutoProcessor.from_pretrained(model_path)

        if load_in_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path, quantization_config=bnb_config, device_map="auto"
            )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path, dtype=torch.bfloat16
            ).to(self.device)

        self.model.eval()
        logger.info("VLM loaded successfully and ready for inference.")

    def generate(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text_prompt, images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        generated_text = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        return generated_text.strip()


class UnifiedInferenceEngine:
    def __init__(self, model_type: str, model_path: str, load_in_4bit: bool = True):
        self.model_type = model_type

        if model_type == "llm":
            self.engine = LLMInferenceEngine(model_path, load_in_4bit)
        elif model_type == "vlm":
            self.engine = VLMInferenceEngine(model_path, load_in_4bit)
        else:
            raise ValueError(f"Invalid model_type: '{model_type}'. Expected 'llm' or 'vlm'.")

    def generate(self, **kwargs) -> str:
        return self.engine.generate(**kwargs)


if __name__ == "__main__":
    
    engine = UnifiedInferenceEngine(
        model_type="llm",
        model_path="outputs/merged_model",
        load_in_4bit=True,
    )
    response = engine.generate(prompt="What is machine learning?")
    print(f"Response: {response}")
