# LLM-Host.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
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

# Middleware to log all requests and responses
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        DramaticLogger["Normal"]["info"](f"[LLM-Host] Request: {request.method} {request.url}")
        if request.method == "POST":
            body = await request.body()
            try:
                body_str = body.decode('utf-8')
                DramaticLogger["Dramatic"]["debug"]("[LLM-Host] Request Body:", body_str)
            except UnicodeDecodeError:
                DramaticLogger["Dramatic"]["warning"]("[LLM-Host] Could not decode request body.")
            
            # Reassign the body so downstream can read it
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            request = Request(scope=request.scope, receive=receive)
        
        if request.method == "GET": # Development debugging only; avoid logging in production. This can reveal sensitive information, particularly API keys.
            headers = dict(request.headers)
            DramaticLogger["Dramatic"]["debug"]("[LLM-Host] GET Request Headers:", headers)  # Development debugging only; avoid logging in production
        
        response = await call_next(request)
        DramaticLogger["Normal"]["info"](f"[LLM-Host] Response: {response.status_code}")
        return response

app.add_middleware(LoggingMiddleware)

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

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    # Include other parameters as per OpenAI specs

class Choice(BaseModel):
    message: Message
    finish_reason: str
    index: int

class ChatCompletionResponse(BaseModel):
    choices: List[Choice]
    # Include other fields as per OpenAI specs

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    # Include other parameters as per OpenAI specs

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any]
    finish_reason: str

class CompletionResponse(BaseModel):
    choices: List[CompletionChoice]
    # Include other fields as per OpenAI specs

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[Any]]
    # Include other parameters as per OpenAI specs

class EmbeddingData(BaseModel):
    index: int
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    # Include other fields as per OpenAI specs

class Model(BaseModel):
    id: str
    object: str
    created: int
    # Include other fields as per OpenAI specs

class ModelsResponse(BaseModel):
    data: List[Model]


## =======================================----------------=======================================
## --------------------------------------- AUTHENTICATION ---------------------------------------
## =======================================----------------=======================================

# from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# from fastapi import Depends

# security = HTTPBearer()

# async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     if credentials.scheme != "Bearer":
#         raise HTTPException(status_code=401, detail="Invalid authentication scheme.")
#     return credentials.credentials

## =========================================-----------=========================================
## ----------------------------------------- STREAMING -----------------------------------------
## =========================================-----------=========================================

# from fastapi import WebSocket
# from fastapi.responses import StreamingResponse

# @app.post("/v1/chat/completions")
# async def chat_completions(request: ChatCompletionRequest, api_key: str = Depends(get_api_key)):
#     # Generate response stream
#     def generate():
#         # Yield responses in event format
#         pass
#     return StreamingResponse(generate(), media_type="text/event-stream")

## =======================================----------------=======================================
## --------------------------------------- ERROR HANDLING ---------------------------------------
## =======================================----------------=======================================

# class OpenAIError(BaseModel):
#     error: ErrorData

# class ErrorData(BaseModel):
#     message: str
#     type: str
#     param: Optional[str]
#     code: Optional[str]

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, exc):
#     return JSONResponse(
#         status_code=400,
#         content=OpenAIError(error=ErrorData(message=str(exc), type="invalid_request_error")).dict()
#     )

## ========================================--------------========================================
## ---------------------------------------- BASIC ROUTES ----------------------------------------
## ========================================--------------========================================

@app.get("/")
def home():
    return {"message": "Welcome! Please GET /docs for API information"}

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

## ========================================--------------========================================
## ---------------------------------------- MODEL ROUTES ----------------------------------------
## ========================================--------------========================================

# =======================================
# v1 Routes
# =======================================

@app.post("/v1/audio/") # TODO: Implement this
async def audio():
    """
    Audio endpoint - currently not implemented
    """
    raise HTTPException(status_code=501, detail="Audio endpoint not implemented")

@app.post("/v1/batch/") # TODO: Implement this
async def batch():
    """
    Batch endpoint - currently not implemented
    """
    raise HTTPException(status_code=501, detail="Batch endpoint not implemented")

@app.post("/v1/chat/") # TODO: Implement this
async def chat():
    """
    Chat endpoint - currently not implemented
    """
    raise HTTPException(status_code=501, detail="Chat endpoint not implemented")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest): #, api_key: str = Depends(get_api_key)):
    # Use api_key as needed
    pass

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    # Process the completion request
    # Generate response using model_service
    # Return CompletionResponse
    pass

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    # Process the embedding request
    # Generate embeddings using model_service
    # Return EmbeddingResponse
    pass

@app.post("/v1/files/") # TODO: Implement this
async def files():
    """
    Files endpoint - currently not implemented
    """
    raise HTTPException(status_code=501, detail="Files endpoint not implemented")

@app.post("/v1/fine-tuning/") # TODO: Implement this
async def fine_tuning():
    """
    Fine-tuning endpoint - currently not implemented
    """
    raise HTTPException(status_code=501, detail="Fine-tuning endpoint not implemented")

@app.post("/v1/images/") # TODO: Implement this
async def images():
    """
    Images endpoint - currently not implemented
    """
    raise HTTPException(status_code=501, detail="Images endpoint not implemented")

@app.get("/v1/models")
async def list_models():
    """
    Retrieve a list of available models following the OpenAI API specification.
    """
    try:
        models = model_service.get_available_models()
        return {"data": models}
    except Exception as e:
        DramaticLogger["Dramatic"]["error"]("[LLM-Host] Failed to list models:", str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve models.")

@app.post("/v1/models/") # TODO: Implement this
async def models():
    """
    Models endpoint - currently not implemented
    """
    raise HTTPException(status_code=501, detail="Models endpoint not implemented")

@app.post("/v1/moderations/") # TODO: Implement this
async def moderations():
    """
    Moderations endpoint - currently not implemented
    """
    raise HTTPException(status_code=501, detail="Moderations endpoint not implemented")

@app.post("/v1/uploads/") # TODO: Implement this
async def uploads():
    """
    Uploads endpoint - currently not implemented
    """
    raise HTTPException(status_code=501, detail="Uploads endpoint not implemented")


# =======================================
# Old Routes
# =======================================

@app.post("/old/initialize", summary="Initialize the model and parameters")
async def initialize(request: InitializeRequest):
    """
    Initialize the language model with the provided parameters.
    """
    try:
        model_service.initialize_model(request)
        return {"message": f"Model {request.model} initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/old/generate", response_model=LLMResponse, summary="Generate a response from the model")
async def LLM(request: LLMRequest):
    """
    Generate a response from the language model based on user input.
    """
    try:
        output_text = model_service.generate_response(request.messages)
        return LLMResponse(response=output_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/old/stream")
async def stream_LLM(request: LLMRequest):
    """Stream the LLM response token by token."""
    try:
        return model_service.stream_response(request.messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/old/status", summary="Check the status of the API")
async def status():
    """
    Check the current status of the API, including whether the model is initialized and its name.
    """
    return model_service.get_status()
