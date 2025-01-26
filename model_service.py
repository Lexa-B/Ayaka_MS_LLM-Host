# model_service.py

import importlib
from typing import List, Dict, Any, Optional
from dramatic_logger import DramaticLogger
import torch
import os
import pkgutil
import re
import json
from typing import List, Dict
from pydantic import BaseModel

## =========================================-----------=========================================
## ----------------------------------------- FUNCTIONS -----------------------------------------
## =========================================-----------=========================================
def extract_function_calls(text: str) -> List[Dict]:
    """
    Extracts all function call requests from a given text chunk.

    Args:
        text (str): The text to search through.

    Returns:
        List[Dict]: A list of dictionaries containing function call details.
    """
    # Regular expression to match the function_call JSON structure
    pattern = re.compile(
        r'\{"function_call":\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}\}',
        re.DOTALL
    )
    
    matches = pattern.findall(text)
    function_calls = []

    for name, args_str in matches:
        try:
            # Parse the arguments string into a dictionary
            arguments = json.loads(args_str)
            function_calls.append({
                "name": name,
                "arguments": arguments
            })
        except json.JSONDecodeError as e:
            print(f"Failed to parse arguments for function '{name}': {e}")
            continue  # Skip invalid JSON structures

    return function_calls

def process_text_chunks(chunks: List[str]) -> List[Dict]:
    """
    Processes a list of text chunks to extract all function call requests.

    Args:
        chunks (List[str]): The list of text chunks to process.

    Returns:
        List[Dict]: A combined list of all function calls found.
    """
    all_function_calls = []
    for index, chunk in enumerate(chunks):
        print(f"Processing chunk {index + 1}/{len(chunks)}...")
        calls = extract_function_calls(chunk)
        if calls:
            print(f"  Found {len(calls)} function call(s).")
            all_function_calls.extend(calls)
        else:
            print("  No function calls found.")
    return all_function_calls

## =====================================-------------------=====================================
## ------------------------------------- CLASS DEFINITIONS -------------------------------------
## =====================================-------------------=====================================
# We'll define a class to store parameters:
class ModelParams:
    def __init__(self, request):
        self.model = request.model                            # Name of the model being used
        self.ayaka_llm_api_key = request.ayaka_llm_api_key    # Ayaka LLM API key (stored but not used)
        self.temperature = request.temperature                # Sampling temperature
        self.max_tokens = request.max_tokens                  # Maximum number of tokens to generate
        self.top_k = request.top_k                            # Top-k for next token selection
        self.top_p = request.top_p                            # Top-p for nucleus sampling
        self.seed = request.seed                              # Seed for deterministic results
        self.stop = request.stop if request.stop else []      # List of stop words to truncate responses
        self.quant_4bit = request.quant_4bit                  # Whether to use 4-bit quantization
        self.quant_type = request.quant_type                  # Type of quantization
        self.quant_dtype = request.quant_dtype                # Data type for quantization
        self.tools = request.tools if hasattr(request, 'tools') else None  # Add tools parameter

class Output(BaseModel):
    type: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Make tool_calls optional

    def __init__(self, **data):
        super().__init__(**data)

class ModelService:
    def __init__(self):
        self.current_handler = None  # Will hold an instance of a model-specific handler
        self.model_initialized = False
        self.model_name = None
        self.temperature = None
        self.max_tokens = None
        self.top_p = None
        self.tools = None
        self.top_k = None
        self.seed = None
        self.stop = None
        self.quant_4bit = None
        self.quant_type = None
        self.quant_dtype = None

    def initialize_model(self, request):
        """
        Dynamically load a model handler based on request.model.
        """
        # Convert the request to our parameter object
        params = ModelParams(request)
        self.model_name = params.model
        self.temperature = params.temperature
        self.max_tokens = params.max_tokens
        self.top_p = params.top_p
        self.tools = params.tools
        self.top_k = params.top_k
        self.seed = params.seed
        self.stop = params.stop
        self.quant_4bit = params.quant_4bit
        self.quant_type = params.quant_type
        self.quant_dtype = params.quant_dtype

        # We expect the file name to match something like "model_srv_{model_name}" 
        # but let's do a small transform: e.g. "mistralai_Mistral-7B-Instruct-v0.3" => "mistralai_mistral_7b_instruct_v0_3"
        # So let's just do a safe replace of hyphens/dots
        sanitized_name = params.model.lower().replace('-', '_').replace('.', '_')
        handler_module_name = f"LLM.model_srv_{sanitized_name}"

        try:
            handler_module = importlib.import_module(handler_module_name)
            handler_class = getattr(handler_module, "ModelHandler")
        except (ImportError, AttributeError) as e:
            DramaticLogger["Dramatic"]["error"](
                f"[ModelService] Could not find a handler for model '{params.model}':",
                f"Please ensure it is supported by the LLM-Host. Error: {e}"
            )
            raise ValueError(f"Model not available: {params.model}")

        try:
            self.current_handler = handler_class(params)
            self.model_initialized = True
            DramaticLogger["Normal"]["success"](f"[ModelService] Model '{params.model}' initialized successfully.")
        except Exception as e:
            if "Model files not found" in str(e):
                DramaticLogger["Dramatic"]["warning"](f"[ModelService] Model files not found:", str(e))
                raise ValueError(f"Model files not found for {params.model}")
            else:
                DramaticLogger["Dramatic"]["error"](f"[ModelService] Failed to initialize model:", str(e))
            raise ValueError(f"Failed to initialize model: {params.model}")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate text from the loaded model using the current handler.
        """
        if not self.model_initialized or not self.current_handler:
            DramaticLogger["Dramatic"]["error"](f"[ModelService] Model not initialized. Call /initialize first.")
            raise ValueError("Model not initialized. Call /initialize first.")

        input_ids = self.current_handler.prepare_input(messages)

        # If a GPU is available, move the input tokens to GPU
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        # Generate output
        outputs = self.current_handler.generate(input_ids)

        # Decode into a general output text object
        output_text = self.current_handler.decode_output(outputs, input_ids.shape[-1])
        output_text = self.current_handler.postprocess_output(output_text)

        # Evaluate if the model attempted to use a tool
        if "{\"function_call\"" in output_text:
            extracted_tool_calls = extract_function_calls(output_text)
            cleaned_tool_calls = []
            # Make sure every occurrence was parsed
            if len(extracted_tool_calls) == output_text.count("{\"function_call\""):
                for tool_call in extracted_tool_calls:
                    new_tool_call = {}
                    new_tool_call["id"] = "NOT_YET_IMPLEMENTED"
                    new_tool_call["type"] = "function"
                    new_tool_call["function"] = {}
                    new_tool_call["function"]["name"] = tool_call["name"]
                    # Convert the arguments to a JSON string, This is needed for LangChain to parse it
                    new_tool_call["function"]["arguments"] = json.dumps(tool_call["arguments"])
                    cleaned_tool_calls.append(new_tool_call)
                output = Output(type="tool_call", content=None, tool_calls=cleaned_tool_calls)
            else:
                DramaticLogger["Dramatic"]["warning"]("[ModelService] Inconsistent number of tool calls detected in output:", f"Parsed {len(extracted_tool_calls)} tool call(s), expected {output_text.count('{\"function_call\"')}")
                output = None
        else:
            output = Output(type="basic", content=output_text)

        return output
    


    def stream_response(self, messages: List[Dict[str, str]], use_sse_format: bool = False):
        """
        Stream the response token by token from the loaded model.
        If use_sse_format=True, wrap each token in SSE "data: ...\n\n" format.
        """
        if not self.model_initialized or not self.current_handler:
            DramaticLogger["Dramatic"]["error"]("[ModelService] Model not initialized. Call /initialize first.")
            raise ValueError("Model not initialized. Call /initialize first.")
        
        return self.current_handler.stream_output(messages, use_sse_format=use_sse_format)

    def get_status(self):
        """
        Returns status, including whether a model is initialized and which model is loaded.
        """
        return {
            "status": "running",
            "model_initialized": self.model_initialized,
            "model_name": self.model_name if self.model_initialized else "none"
        }

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Retrieves a list of available models with their metadata.
        """
        models = []
        llm_directory = os.path.join(os.path.dirname(__file__), 'LLM')

        for finder, name, ispkg in pkgutil.iter_modules([llm_directory]):
            if name.startswith("model_srv_") and not ispkg:
                model_id = name.replace("model_srv_", "").replace("_", "-")
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(os.path.getctime(llm_directory)),  # Example timestamp
                    "owned_by": "user",  # Adjust as needed
                    "permission": []  # Add permissions if applicable
                })

        DramaticLogger["Normal"]["info"]("[ModelService] Retrieved available models:", models)
        return models
