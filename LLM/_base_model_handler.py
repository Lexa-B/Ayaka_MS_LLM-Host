# LLM/base_model_handler.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from typing import List, Dict
import asyncio
import json
import time
import uuid
import os
from fastapi.responses import StreamingResponse
from dramatic_logger import DramaticLogger
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
HF_API_Token = os.getenv('HF_API_Token')

class BaseModelHandler:
    """
    The base class that implements shared logic: 
    generation, streaming, decode, etc.
    """

    def __init__(self, params):
        DramaticLogger["Normal"]["info"](f"[BaseModelHandler] Initialization called with model:", params.model)
        self.params = params
        self.model = None
        self.tokenizer = None

        try:
            # Let subclasses define or override build_model_path()
            self.model_path = self.build_model_path()

            # If a seed is provided, fix it for reproducibility
            if self.params.seed is not None:
                torch.manual_seed(self.params.seed)

            # Load the model and tokenizer
            self.load_model()  
            DramaticLogger["Normal"]["info"](f"[BaseModelHandler] init done. Model path:", self.model_path)

        except Exception as e:
            if "Model files not found" in str(e):
                DramaticLogger["Dramatic"]["warning"]("[BaseModelHandler] Model files not found:", str(e))
                raise ValueError(f"Model files not found for {self.params.model}")
            else:
                DramaticLogger["Dramatic"]["error"](f"[BaseModelHandler] Error in initialization:", str(e))
                raise

    def build_model_path(self) -> str:
        """
        By default, returns a fallback path (if a subclass does not override).
        Subclasses typically override this method or define a constant 
        to specify their custom path, e.g. ./LLM/Mistralai/Mistral-7B.
        """    
        try:
            DramaticLogger["Dramatic"]["warning"](
                "[BaseModelHandler] No explicit build_model_path() override found; using default."
            )
            return f"./LLM/{self.params.model}"
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] build_model_path() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e    

    def load_model(self):
        """
        A mostly generic approach:
          - Build BitsAndBytesConfig from self.params
          - Load tokenizer and model
          - Possibly set a default chat_template
        Subclasses can override if there's something truly unique.
        """
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=self.params.quant_4bit,
                bnb_4bit_quant_type=self.params.quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.params.quant_dtype)
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quant_config,
                local_files_only=True
            )

            # If no chat_template is defined, set a fallback
            if not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n
                {% endif %}
                {% endfor %}
                """

            if torch.cuda.is_available():
                DramaticLogger["Normal"]["debug"]("[BaseModelHandler] GPU is available. Moving model to GPU.")
                self.model.to("cuda")
            DramaticLogger["Normal"]["info"]("[BaseModelHandler] Default load_model() done.")
        except Exception as e:
            if "Incorrect path_or_model_id: './LLM/" in str(e):
                HubPath = str(e).split("'")[1].lstrip("./LLM/")
                DramaticLogger["Dramatic"]["warning"]("[BaseModelHandler] Model files not found:", str(e))
                print(f"Model path: {self.model_path}")
                self.download_model(HubPath) # Download the missing model files from Hugging Face Hub
                raise ValueError(f"Model files not found for {HubPath}")
            else:
                DramaticLogger["Dramatic"]["error"]("[BaseModelHandler] Error loading model:", f"Error: {str(e)}")
                raise Exception(f"Failed to load model: {str(e)}")
            
    def download_model(self, model_hub_path: str):
        if not HF_API_Token:
            DramaticLogger["Dramatic"]["error"]("[BaseModelHandler] Error getting API token:", "No HuggingFace API token found. Set the HF_API_Token environment variable.")
            raise ValueError("No HuggingFace API token found. Set the HF_API_Token environment variable.")

        try:
            DramaticLogger["Normal"]["info"]("[BaseModelHandler] Starting model download from Hugging Face Hub:", model_hub_path)
            # Start download in a separate thread
            import threading
            download_thread = threading.Thread(
                target=self._download_model_thread,
                args=(model_hub_path,)
            )
            download_thread.daemon = True
            download_thread.start()
            
            # Still raise the "Model not found" error
            raise ValueError(f"Model files not found for {model_hub_path}")
            
        except ValueError as ve:
            raise ve
        except Exception as e:
            DramaticLogger["Dramatic"]["error"]("[BaseModelHandler] Error downloading model files:", f"Error: {str(e)}")
            raise Exception(f"Failed to download model files: {str(e)}")

    def _download_model_thread(self, model_hub_path: str):
        """Helper method to handle the actual model download in a separate thread."""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_hub_path, 
                local_files_only=False, 
                token=HF_API_Token
            )
            tokenizer = AutoTokenizer.from_pretrained(  
                model_hub_path, 
                local_files_only=False, 
                token=HF_API_Token
            )
            model.save_pretrained(f"./LLM/{model_hub_path}")
            tokenizer.save_pretrained(f"./LLM/{model_hub_path}")
            DramaticLogger["Normal"]["info"]("[BaseModelHandler] Model files downloaded successfully.")
        except Exception as e:
            DramaticLogger["Dramatic"]["error"]("[BaseModelHandler] Error in download thread:", f"Error: {str(e)}")

    # ----------------------------------------------------------------
    #  PREPROCESS
    # ----------------------------------------------------------------
    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        try:
            DramaticLogger["Dramatic"]["info"](
                "[BaseModelHandler] Preprocessing messages:", messages
            )
            return messages  # By default, no special transformation
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] preprocess_messages() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e
        
    # ----------------------------------------------------------------
    #  APPLY CHAT TEMPLATE
    # ----------------------------------------------------------------
    def format_tools(self, tools) -> str:
        """Format tools into a string that can be included in the prompt."""
        if not tools:
            return ""
        
        tools_str = "\nAvailable tools:\n"
        for tool in tools:
            # Handle both Tool objects and dictionaries
            if isinstance(tool, dict):
                func = tool['function']
                func_name = func['name']
                func_desc = func['description']
                func_params = func['parameters']
            else:
                func = tool.function
                func_name = func.name
                func_desc = func.description
                func_params = func.parameters.dict()
            
            tools_str += f"\n{func_name}: {func_desc}\n"
            tools_str += "Parameters:\n"
            for param_name, param in func_params['properties'].items():
                required = "required" if param_name in func_params['required'] else "optional"
                tools_str += f"  - {param_name} ({param['type']}, {required}): {param['description']}\n"
        return tools_str

    def apply_chat_template(self, messages: List[Dict[str, str]]):
        DramaticLogger["Dramatic"]["trace"]("[BaseModelHandler] Recieved message:", messages)
        try:
            # If the model has tools available, include them in the system message
            if hasattr(self.params, 'tools') and self.params.tools:
                tools_str = self.format_tools(self.params.tools)
                
                # Add or update system message with tools
                has_system = any(msg.get('role') == 'system' for msg in messages)
                if has_system:
                    # Append tools to existing system message
                    for msg in messages:
                        if msg['role'] == 'system':
                            msg['content'] = msg['content'] + "\n" + tools_str
                else:
                    # Create new system message with tools
                    messages.insert(0, {
                        'role': 'system',
                        'content': f"You have access to the following tools. When you need to use a tool, format your response as a JSON function call like: {{\"function_call\": {{\"name\": \"tool_name\", \"arguments\": {{\"arg1\": \"value1\"}}}}}}\n{tools_str}"
                    })

            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] apply_chat_template() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e
        
    # ----------------------------------------------------------------
    #  PREPARE INPUT
    # ----------------------------------------------------------------
    def prepare_input(self, messages: List[Dict[str, str]]):
        try:
            processed_msgs = self.preprocess_messages(messages)
            input_ids = self.apply_chat_template(processed_msgs)
            DramaticLogger["Dramatic"]["debug"](
                "[BaseModelHandler] Input IDs:", self.tokenizer.decode(input_ids[0])
            )
            try:
                tensor_info = f"Shape: {input_ids.shape}, Device: {input_ids.device}, Type: {input_ids.dtype}"
                DramaticLogger["Dramatic"]["trace"]("[BaseModelHandler] Input IDs:", tensor_info)
            except Exception as e:
                DramaticLogger["Dramatic"]["warning"]("[BaseModelHandler] Could not log tensor details:", str(e))            
            return input_ids
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] prepare_input() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e
        
    # ----------------------------------------------------------------
    #  GENERATE
    # ----------------------------------------------------------------
    def generate(self, input_ids):
        try:
            if self.params.temperature > 0: # Only sample if not beam and temperature is greater than 0
                gen_kwargs = {
                    "max_length": input_ids.shape[1] + self.params.max_tokens, # Max length is the input length plus the max tokens
                    "temperature": self.params.temperature,                    # Temperature is the temperature
                    "top_k": self.params.top_k,                                # Top-k is the top-k
                    "top_p": self.params.top_p,                                # Top-p is the top-p
                    "do_sample": True,                                         # Do sample
                    "eos_token_id": self.get_terminators(),                    # Use terminators
                    "pad_token_id": self.tokenizer.eos_token_id,               # Pad token is the eos token
                }
            else:
                gen_kwargs = { # Not beam search and temperature is 0, do not sample, just generate
                    "max_length": input_ids.shape[1] + self.params.max_tokens, # Max length is the input length plus the max tokens
                    "do_sample": False,                                        # Do not sample, no temperature, no top_k, no top_p
                    "eos_token_id": self.get_terminators(),                    # Use terminators
                    "pad_token_id": self.tokenizer.eos_token_id,               # Pad token is the eos token
                }   
            
            outputs = self.model.generate(input_ids, **gen_kwargs)
            # Safely log the tensor information
            try:
                tensor_info = f"Shape: {outputs.shape}, Device: {outputs.device}, Type: {outputs.dtype}"
                DramaticLogger["Dramatic"]["trace"]("[BaseModelHandler] Generated outputs:", tensor_info)
            except Exception as e:
                DramaticLogger["Dramatic"]["warning"]("[BaseModelHandler] Could not log tensor details:", str(e))
            return outputs
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] generate() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e
        
    # ----------------------------------------------------------------
    #  DECODE
    # ----------------------------------------------------------------
    def decode_output(self, outputs, prompt_len: int):
        try:
            DecodedOutput = self.tokenizer.decode(
                outputs[0][prompt_len:],
                skip_special_tokens=True
            )
            DramaticLogger["Dramatic"]["trace"](
                "[BaseModelHandler] Decoded output:", DecodedOutput
            )
            return DecodedOutput
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] decode_output() encountered an error:",
                  str(e),
                exc_info=True
            )
            raise e
        
    # ----------------------------------------------------------------
    #  POSTPROCESS
    # ----------------------------------------------------------------
    def postprocess_output(self, text: str) -> str:
        try:
            if text.endswith("</s>"):
                text = text[:-4]
            DramaticLogger["Dramatic"]["debug"](
                "[BaseModelHandler] Postprocessed output:", text
            )
            return text
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] postprocess_output() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e
        
    # ----------------------------------------------------------------
    #  TERMINATORS
    # ----------------------------------------------------------------
    def get_terminators(self) -> List[int]:
        try:
            terminators = [self.tokenizer.eos_token_id]
            for possible in ["<|eot_id|>", "<|im_end|>", "</s>"]:
                tok_id = self.safe_token_id(possible)
                if tok_id is not None:
                    terminators.append(tok_id)
            return list(set(terminators))
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] get_terminators() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e

    def safe_token_id(self, token_str: str):
        try:
            if token_str in self.tokenizer.vocab:
                return self.tokenizer.vocab[token_str]
            return self.tokenizer.convert_tokens_to_ids(token_str)
        except Exception as e:
            # If an exception occurs (token not found or something else),
            # just log it and return None rather than failing everything.
            DramaticLogger["Normal"]["warning"](
                f"[BaseModelHandler] safe_token_id() could not retrieve ID for '{token_str}': {str(e)}"
            )
            return None
        
    # ----------------------------------------------------------------
    #  STREAMING
    # ----------------------------------------------------------------
    def stream_output(self, messages: List[Dict[str, str]], use_sse_format: bool = False):
        try:
            input_ids = self.prepare_input(messages)
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
            
            streamer = self.get_streamer()
            
            if self.params.temperature > 0: # Only sample if not beam and temperature is greater than 0
                gen_kwargs = {
                    "max_length": input_ids.shape[1] + self.params.max_tokens, # Max length is the input length plus the max tokens
                    "temperature": self.params.temperature,                    # Temperature is the temperature
                    "top_k": self.params.top_k,                                # Top-k is the top-k
                    "top_p": self.params.top_p,                                # Top-p is the top-p
                    "do_sample": True,                                         # Do sample
                    "eos_token_id": self.get_terminators(),                    # Use terminators
                    "pad_token_id": self.tokenizer.eos_token_id,               # Pad token is the eos token
                    "streamer": streamer                                       # Use the streamer
                }
            else:
                gen_kwargs = { # Not beam search and temperature is 0, do not sample, just generate
                    "max_length": input_ids.shape[1] + self.params.max_tokens, # Max length is the input length plus the max tokens
                    "do_sample": False,                                        # Do not sample, no temperature, no top_k, no top_p
                    "eos_token_id": self.get_terminators(),                    # Use terminators
                    "pad_token_id": self.tokenizer.eos_token_id,               # Pad token is the eos token
                    "streamer": streamer                                       # Use the streamer
                }   
            
            import threading
            thread = threading.Thread(
                target=self.model.generate,
                kwargs=dict(input_ids=input_ids, **gen_kwargs)
            )
            thread.daemon = True
            thread.start()
        
            async def token_generator():
                completion_id = f"cmpl-{str(uuid.uuid4())}"
                first_chunk = True
                try:
                    for text in streamer:
                        # skip the first chunk if it's the prompt
                        if first_chunk:
                            first_chunk = False
                            continue
    
                        # remove trailing </s> if any
                        text = text.rstrip("</s>")
                        DramaticLogger["Normal"]["debug"]("[BaseModelHandler] Streaming output:", text)
                        
                        # SSE format (or raw text) depending on use_sse_format
                        if use_sse_format:
                            # Build respond in OpenAI “chat.completion.chunk” SSE style
                            json_payload = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": self.params.model,
                                "choices": [
                                    {
                                        "delta": {"content": text},
                                        "index": 0,
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(json_payload)}\n\n"
                        else:
                            # raw text
                            yield text
                        await asyncio.sleep(0)
                except Exception as e:
                    DramaticLogger["Dramatic"]["error"](
                        "[BaseModelHandler] Exception in token_generator:",
                        str(e),
                        exc_info=True
                    )
                    raise e
                finally:
                    streamer.end()
                    if thread.is_alive():
                        thread.join(timeout=1.0)
                    # Once everything is done, return the final [DONE] line for SSE client
                    if use_sse_format:
                        yield "data: [DONE]\n\n"
                    
            return StreamingResponse(
                token_generator(),
                media_type="text/event-stream" if use_sse_format else "text/plain"
            )
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] stream_output() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e
        
    def get_streamer(self):
        try:
            return TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] get_streamer() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e