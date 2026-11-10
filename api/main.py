import sys
import os
import time
import base64
import io 

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from contextlib import asynccontextmanager
 
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "inference"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from logger import get_logger
from api.schemas import (
    TextGenerationRequest,
    TextGenerationResponse,
    VLMGenerationRequest,
    VLMGenerationResponse,
    HealthCheckResponse,
)

logger = get_logger(__name__)

# The inference stack (torch/transformers) is heavy. Importing it lazily keeps
# the API bootable on a light machine: the server still starts, /health reports
# model_not_loaded, and generation endpoints answer 503 instead of crashing.
try:
    from inference_engine import UnifiedInferenceEngine
    INFERENCE_IMPORT_ERROR = None
except ImportError as e:
    UnifiedInferenceEngine = None
    INFERENCE_IMPORT_ERROR = str(e)
    logger.warning(
        f"Inference dependencies are not installed ({e}). "
        "The API will start but generation endpoints will return 503."
    )

model_engine = None
MODEL_TYPE = os.environ.get("MODEL_TYPE", "llm")
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/merged_model")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_engine
    logger.info(f"Starting up... Loading {MODEL_TYPE} model from {MODEL_PATH}")

    if UnifiedInferenceEngine is None:
        logger.warning(
            f"Skipping model load: {INFERENCE_IMPORT_ERROR}. "
            "Install requirements.txt to serve a model."
        )
    else:
        try:
            model_engine = UnifiedInferenceEngine(
                model_type=MODEL_TYPE,
                model_path=MODEL_PATH,
                load_in_4bit=LOAD_IN_4BIT,
            )
            logger.info("Model loaded successfully. API is ready to serve requests.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model_engine = None

    yield  # API requests are served here

    logger.info("Shutting down API server.")


app = FastAPI(
    title="QLoRA Fine-Tuning & Quantization Pipeline API",
    description="REST API for serving QLoRA fine-tuned and quantized LLMs/VLMs",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allows the Gradio demo (or any frontend) to call the API cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
def root():
    return {
        "message": "QLoRA Fine-Tuning & Quantization Pipeline API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
def health_check():
    return HealthCheckResponse(
        status="healthy" if model_engine is not None else "model_not_loaded",
        model_loaded=model_engine is not None,
        model_type=MODEL_TYPE if model_engine is not None else None,
    )


@app.post("/generate/text", response_model=TextGenerationResponse, tags=["LLM"])
def generate_text(request: TextGenerationRequest):
    if model_engine is None or MODEL_TYPE != "llm":
        raise HTTPException(
            status_code=503,
            detail="LLM model is not loaded. Check server configuration (MODEL_TYPE=llm).",
        )

    start_time = time.time()

    try:
        generated_text = model_engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    latency_ms = (time.time() - start_time) * 1000

    return TextGenerationResponse(
        generated_text=generated_text,
        prompt=request.prompt,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/generate/vision", response_model=VLMGenerationResponse, tags=["VLM"])
def generate_vision(request: VLMGenerationRequest):
    if model_engine is None or MODEL_TYPE != "vlm":
        raise HTTPException(
            status_code=503,
            detail="VLM model is not loaded. Check server configuration (MODEL_TYPE=vlm).",
        )

    start_time = time.time()

    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        generated_text = model_engine.generate(
            image=image,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        logger.error(f"VLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    latency_ms = (time.time() - start_time) * 1000

    return VLMGenerationResponse(
        generated_text=generated_text,
        prompt=request.prompt,
        latency_ms=round(latency_ms, 2),
    )

