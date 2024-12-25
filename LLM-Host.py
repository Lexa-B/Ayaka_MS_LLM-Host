# LLM-Host.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio
from dramatic_logger import DramaticLogger

# Import the shared model service class
from model_service import ModelService

# Initialize the FastAPI application
app = FastAPI(
    title="Ayaka LLM API",
    description="A local LLM microservice that serves the Ayaka LLMs. It roughly mimics the ChatNVIDIA parameters and API calls.",
    version="0.0.2",
)
favicon_path = 'favicon.ico'

# Initialize our model service object
model_service = ModelService()
DramaticLogger["Dramatic"]["info"](f"[LLM-Host] Initializing LLM-Host. Current model service status:", model_service.get_status())

# =======================================
# Pydantic Models
# =======================================
class InitializeRequest(BaseModel):
    # ToDo: 
    # Learn more about text generation strategies: https://huggingface.co/docs/transformers/v4.28.1/generation_strategies
    # Then do something like this, if I still think it's a good idea:
    # - Enable beam search
    #   - Make an option to enable/disable beam search
    #   - Move all model params specicic to beam search into a dictionary
    # - Move all model params specific to sampling into a sub-class
    #   - This will allow us to have a base class for all models that don't need sampling
    #   - And a dictionary key for all models that do need sampling
    model: str = Field(..., description="The model to use for chat.")
    ayaka_llm_api_key: Optional[str] = Field(None, description="The Ayaka LLM API key from the requestor.")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature in [0,1].")
    max_tokens: int = Field(256, description="Maximum number of tokens to generate.")
    top_k: int = Field(50, description="Top-k for next token selection.")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p for nucleus sampling.")
    seed: Optional[int] = Field(None, description="Seed for deterministic results.")
    stop: Optional[List[str]] = Field(None, description="List of cased stop words to halt generation.")
    quant_4bit: Optional[bool] = Field(True, description="Whether to use 4-bit quantization.")
    quant_type: Optional[str] = Field("nf4", description="Type of quantization.")
    quant_dtype: Optional[str] = Field("float16", description="Data type for quantization.")


class LLMRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(None, description="List of messages to send to the model.")


class LLMResponse(BaseModel):
    response: str = Field(..., description="The model-generated response.")

# =======================================
# Basic Routes
# =======================================

@app.get("/")
def home():
    return {"message": "Welcome! Please GET /docs for API information"}

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


# =======================================
# Model Routes
# =======================================

@app.post("/initialize", summary="Initialize the model and parameters")
async def initialize(request: InitializeRequest):
    """
    Initialize the language model with the provided parameters.
    """
    try:
        model_service.initialize_model(request)
        return {"message": f"Model {request.model} initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=LLMResponse, summary="Generate a response from the model")
async def LLM(request: LLMRequest):
    """
    Generate a response from the language model based on user input.
    """
    try:
        output_text = model_service.generate_response(request.messages)
        return LLMResponse(response=output_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream_LLM(request: LLMRequest):
    """Stream the LLM response token by token."""
    try:
        return model_service.stream_response(request.messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", summary="Check the status of the API")
async def status():
    """
    Check the current status of the API, including whether the model is initialized and its name.
    """
    return model_service.get_status()
