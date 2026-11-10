from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt/instruction text")
    max_new_tokens: int = Field(256, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="Nucleus sampling threshold", ge=0.0, le=1.0)
    do_sample: bool = Field(True, description="Whether to sample or decode greedily")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Explain quantum computing in simple terms.",
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            }
        }
    )


class TextGenerationResponse(BaseModel):
    generated_text: str = Field(..., description="Text produced by the model")
    prompt: str = Field(..., description="Original input prompt")
    latency_ms: float = Field(..., description="Generation time in milliseconds")


class VLMGenerationRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image string")
    prompt: str = Field(..., description="Question/instruction about the image")
    max_new_tokens: int = Field(256, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)


class VLMGenerationResponse(BaseModel):
    generated_text: str = Field(..., description="Text produced by the model")
    prompt: str = Field(..., description="Original input prompt")
    latency_ms: float = Field(..., description="Generation time in milliseconds")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    model_type: Optional[str] = Field(None, description="Type of model loaded (llm/vlm)")
