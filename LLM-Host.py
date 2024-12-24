# LLM-Host.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio

# Import the shared model service class
from model_service import ModelService

# Initialize the FastAPI application
app = FastAPI(
    title="ChatAyaka API",
    description="A local LLM microservice that serves the Ayaka LLMs. It roughly mimics the ChatNVIDIA parameters and API calls.",
    version="0.0.2",
)
favicon_path = 'favicon.ico'

# Initialize our model service object
model_service = ModelService()

# =======================================
# Pydantic Models
# =======================================
class InitializeRequest(BaseModel):
    model: str = Field(..., description="The model to use for chat.")
    chat_ayaka_api_key: Optional[str] = Field(None, description="The ChatAyaka API key for hosted NIM.")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature in [0,1].")
    max_tokens: int = Field(256, description="Maximum number of tokens to generate.")
    top_k: int = Field(50, description="Top-k for next token selection.")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p for nucleus sampling.")
    seed: Optional[int] = Field(None, description="Seed for deterministic results.")
    stop: Optional[List[str]] = Field(None, description="List of cased stop words to halt generation.")
    quant_4bit: Optional[bool] = Field(True, description="Whether to use 4-bit quantization.")
    quant_type: Optional[str] = Field("nf4", description="Type of quantization.")
    quant_dtype: Optional[str] = Field("float16", description="Data type for quantization.")


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(None, description="List of messages to send to the model.")


class ChatResponse(BaseModel):
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


@app.post("/generate", response_model=ChatResponse, summary="Generate a response from the model")
async def chat(request: ChatRequest):
    """
    Generate a response from the language model based on user input.
    """
    try:
        output_text = model_service.generate_response(request.messages)
        return ChatResponse(response=output_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream_chat(request: ChatRequest):
    """Stream the chat response token by token."""
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
