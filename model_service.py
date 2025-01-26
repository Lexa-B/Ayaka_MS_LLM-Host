# model_service.py

import importlib
from typing import List, Dict, Any
from dramatic_logger import DramaticLogger
import torch
import os
import pkgutil

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


class ModelService:
    def __init__(self):
        self.current_handler = None  # Will hold an instance of a model-specific handler
        self.model_initialized = False
        self.model_name = None

    def initialize_model(self, request):
        """
        Dynamically load a model handler based on request.model.
        """
        # Convert the request to our parameter object
        params = ModelParams(request)
        self.model_name = params.model

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

        # Decode
        output_text = self.current_handler.decode_output(outputs, input_ids.shape[-1])
        output_text = self.current_handler.postprocess_output(output_text)

        return output_text

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
