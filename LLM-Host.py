# LLM-Host.py

from fastapi import FastAPI, HTTPException, Request
from starlette import status
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import asyncio
from dramatic_logger import DramaticLogger
import json
import re

# Import the shared model service class
from model_service import ModelService

# Initialize the FastAPI application
app = FastAPI(
    title="Ayaka LLM API",
    description="A local LLM microservice that serves the Ayaka LLMs. It roughly mimics the ChatNVIDIA's expected OpenAI API parameters and API calls.",
    version="0.0.2",
)
favicon_path = 'favicon.ico'

# Initialize our model service object
model_service = ModelService()
DramaticLogger["Dramatic"]["info"](f"[LLM-Host] Initializing LLM-Host. Current model service status:", model_service.get_status())

## ========================================------------========================================
## ---------------------------------------- MIDDLEWARE ---------------------------------------
## ========================================------------========================================

# Middleware to log all requests and responses
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        DramaticLogger["Normal"]["info"](f"[LLM-Host] Request: {request.method} {request.url}")
        if request.method == "POST":
            body = await request.body()
            try:
                body_str = body.decode('utf-8')
                # Parse the JSON string back into a Python object
                body_obj = json.loads(body_str)
                # Convert back to JSON with ensure_ascii=False to preserve unicode
                pretty_body = json.dumps(body_obj, ensure_ascii=False, indent=2)
                DramaticLogger["Dramatic"]["debug"]("[LLM-Host] Request Body:", pretty_body)
            except UnicodeDecodeError:
                DramaticLogger["Dramatic"]["warning"]("[LLM-Host] Could not decode request body.")
            except json.JSONDecodeError:
                DramaticLogger["Dramatic"]["warning"]("[LLM-Host] Could not parse request body as JSON.")
            
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

## ========================================-----------------========================================
## ---------------------------------------- PYDANTIC MODELS ---------------------------------------
## ========================================-----------------========================================

class FunctionParameter(BaseModel):
    description: str
    type: str

class FunctionParameters(BaseModel):
    type: str = "object"
    properties: Dict[str, FunctionParameter]
    required: List[str]

class Function(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters

class Tool(BaseModel):
    type: str = "function"
    function: Function

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
    tools: Optional[List[Tool]] = Field(None, description="List of tools to use for the model.")


class LLMRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(None, description="List of messages to send to the model.")


class LLMResponse(BaseModel):
    response: str = Field(..., description="The model-generated response.")

class Message_Basic(BaseModel):
    role: str
    content: str
    ## This is a custom field that is not part of the OpenAI API spec.
    ## It is used to indicate that the message in the ChatAyaka API needs to modify the tokenizer template and strip the response marker.
    add_response_marker: Optional[bool] = None 
    
    # Allow additional fields
    class Config:
        extra = "allow"

class Message_ToolCall(BaseModel):
    role: str
    content: None
    tool_calls: Optional[List[Dict[str, Any]]] = None

# Create a Union type for messages
Message_Any = Union[Message_Basic, Message_ToolCall]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message_Any]  # Can accept either Message_Basic or Message_ToolCall
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    
    # Include other parameters as per OpenAI specs (e.g., frequency_penalty, presence_penalty)

class Choice(BaseModel):
    message: Message_Any
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
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Before checking tools, check if either is None and handle that case
        tools_changed = (request.tools is None) != (getattr(model_service, 'tools', None) is None) or \
                       (request.tools is not None and request.tools != getattr(model_service, 'tools', None))

        if (not model_service.model_initialized) or \
           (model_service.model_name != request.model) or \
           (request.temperature != model_service.temperature) or \
           (request.max_tokens != model_service.max_tokens) or \
           (request.top_p != model_service.top_p) or \
           tools_changed:
            # Convert tools to dict format if they exist
            tools_dict = [tool.dict() for tool in request.tools] if request.tools else None
            
            # Construct a minimal object matching InitializeRequest structure
            init_like_obj = InitializeRequest(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                tools=request.tools,  # Pass tools through to initialization

                # Provide defaults or additional fields as needed
            )
            try:
                model_service.initialize_model(init_like_obj)
            except ValueError as e:
                if "Model files not found" in str(e):
                    # Can't find the model's files. Return 503 Service Unavailable
                    DramaticLogger["Dramatic"]["warning"]("[LLM-Host] Model files not found:", str(e))
                    raise HTTPException(status_code=503, detail="Model files not found, downloading from HuggingFace Hub. This may take quite a while, please try again later.")
                else:
                    # Can't initialize the model. Return 500 Internal Server Error
                    DramaticLogger["Dramatic"]["error"]("[LLM-Host] Error in initialize:", str(e))
                    raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                # Unexpected error. Return 500 Internal Server Error
                DramaticLogger["Dramatic"]["error"]("[LLM-Host] Error in initialize:", str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        if request.stream:
            # Streaming response
            return model_service.stream_response(
                [m.dict() for m in request.messages],
                use_sse_format=True
            )
        else:
            # Non-streaming response
            output = model_service.generate_response([m.dict() for m in request.messages])
            text_output = output.content
            tool_calls = output.tool_calls
            # DramaticLogger["Dramatic"]["debug"]("[LLM-Host] Output type:", output.type)
            # DramaticLogger["Dramatic"]["debug"]("[LLM-Host] Output content:", output.content)
            # DramaticLogger["Dramatic"]["debug"]("[LLM-Host] Output tool_calls:", output.tool_calls)

            # Build the response
            if output.type == "basic":
                return ChatCompletionResponse(
                    choices=[
                        Choice(
                            index=0,
                            message=Message_Basic(role="assistant", content=text_output),
                            finish_reason="stop"
                        )
                    ]
                )
            elif output.type == "tool_call":
                # Build an OpenAI-style chat completion function call response:
                return ChatCompletionResponse(
                    choices=[
                        Choice(
                            index=0,
                            message=Message_ToolCall(role="assistant", content=None, tool_calls=tool_calls),
                            finish_reason="tool_calls"
                        )
                    ]
                )
            else:
                # Unexpected output type
                raise HTTPException(status_code=500, detail="Unexpected output type")

    except Exception as e:
        # Log any unexpected errors
        DramaticLogger["Dramatic"]["error"]("[LLM-Host] Error in chat_completions:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

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
    except ValueError as e:
        if "Model files not found" in str(e):
            # Can't find the model's files. Return 503 Service Unavailable
            DramaticLogger["Dramatic"]["warning"]("[LLM-Host] Model files not found:", str(e))
            raise HTTPException(status_code=503, detail="Model files not found, downloading from HuggingFace Hub. This may take quite a while, please try again later.")
        else:
            # Can't initialize the model. Return 500 Internal Server Error
            DramaticLogger["Dramatic"]["error"]("[LLM-Host] Error in initialize:", str(e))
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Unexpected error. Return 500 Internal Server Error
        DramaticLogger["Dramatic"]["error"]("[LLM-Host] Error in initialize:", str(e))
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
