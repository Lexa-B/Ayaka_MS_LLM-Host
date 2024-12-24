# LLM/base_model_handler.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from typing import List, Dict
import asyncio
from fastapi.responses import StreamingResponse
from dramatic_logger import DramaticLogger

class BaseModelHandler:
    """
    The base class that implements shared logic: 
    generation, streaming, decode, etc.
    """

    def __init__(self, params):
        DramaticLogger["Dramatic"]["debug"](f"[BaseModelHandler] Initialization called, initializing with params:", f"Model: {params.model}\nAPI Key?: {(type(params.chat_ayaka_api_key) == str and ((len(params.chat_ayaka_api_key) > 0) and (params.chat_ayaka_api_key != "string")))}\nTemperature: {params.temperature}\nMax_tokens: {params.max_tokens}\nTop_k: {params.top_k}\nTop_p: {params.top_p}\nSeed: {params.seed}\nStop: {params.stop}\nQuant_4bit: {params.quant_4bit}\nQuant_type: {params.quant_type}\nQuant_dtype: {params.quant_dtype}")
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
            DramaticLogger["Normal"]["info"](
                f"[BaseModelHandler] init done. Model path:", self.model_path
            )
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] __init__ encountered an error:", 
                str(e), 
                exc_info=True
            )
            raise e

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
                DramaticLogger["Normal"]["debug"](
                    "[BaseModelHandler] GPU is available. Moving model to GPU."
                )
                self.model.to("cuda")

            DramaticLogger["Normal"]["info"](
                "[BaseModelHandler] Default load_model() done."
            )
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] load_model() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e        

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
    def apply_chat_template(self, messages: List[Dict[str, str]]):
        DramaticLogger["Dramatic"]["trace"]("[BaseModelHandler] Recieved message:", messages)
        try:
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
        processed_msgs = self.preprocess_messages(messages)
        input_ids = self.apply_chat_template(processed_msgs)
        DramaticLogger["Dramatic"]["debug"]("[BaseModelHandler] Input IDs:", self.tokenizer.decode(input_ids[0]))
        return input_ids

    # ----------------------------------------------------------------
    #  GENERATE
    # ----------------------------------------------------------------
    def generate(self, input_ids):
        gen_kwargs = {
            "max_length": input_ids.shape[1] + self.params.max_tokens,
            "temperature": self.params.temperature,
            "top_k": self.params.top_k,
            "top_p": self.params.top_p,
            "do_sample": True,
            "eos_token_id": self.get_terminators(),
            "pad_token_id": self.tokenizer.eos_token_id
        }
        outputs = self.model.generate(input_ids, **gen_kwargs)
        return outputs

    # ----------------------------------------------------------------
    #  DECODE
    # ----------------------------------------------------------------
    def decode_output(self, outputs, prompt_len: int):
        DecodedOutput = self.tokenizer.decode(
            outputs[0][prompt_len:],
            skip_special_tokens=True
        )
        DramaticLogger["Dramatic"]["trace"]("[BaseModelHandler] Decoded output:", DecodedOutput)
        return DecodedOutput

    # ----------------------------------------------------------------
    #  POSTPROCESS
    # ----------------------------------------------------------------
    def postprocess_output(self, text: str) -> str:
        if text.endswith("</s>"):
            text = text[:-4]
        DramaticLogger["Dramatic"]["debug"]("[BaseModelHandler] Postprocessed output:", text)
        return text

    # ----------------------------------------------------------------
    #  TERMINATORS
    # ----------------------------------------------------------------
    def get_terminators(self) -> List[int]:
        terminators = [self.tokenizer.eos_token_id]
        for possible in ["<|eot_id|>", "<|im_end|>", "</s>"]:
            tok_id = self.safe_token_id(possible)
            if tok_id is not None:
                terminators.append(tok_id)
        return list(set(terminators))

    def safe_token_id(self, token_str: str):
        if token_str in self.tokenizer.vocab:
            return self.tokenizer.vocab[token_str]
        try:
            return self.tokenizer.convert_tokens_to_ids(token_str)
        except:
            return None

    # ----------------------------------------------------------------
    #  STREAMING
    # ----------------------------------------------------------------
    def stream_output(self, messages: List[Dict[str, str]]):
        input_ids = self.prepare_input(messages)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        streamer = self.get_streamer()
        gen_kwargs = {
            "max_length": input_ids.shape[1] + self.params.max_tokens,
            "temperature": self.params.temperature,
            "top_k": self.params.top_k,
            "top_p": self.params.top_p,
            "do_sample": True,
            "eos_token_id": self.get_terminators(),
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer
        }

        import threading
        thread = threading.Thread(target=self.model.generate, kwargs=dict(input_ids=input_ids, **gen_kwargs))
        thread.daemon = True
        thread.start()

        async def token_generator():
            first_chunk = True
            try:
                for text in streamer:
                    # skip the first chunk if it's the prompt
                    if first_chunk:
                        first_chunk = False
                        continue

                    DramaticLogger["Normal"]["debug"]("[BaseModelHandler] Streaming output:", text)

                    # remove trailing </s> if any
                    txt = text.rstrip("</s>")
                    if txt:
                        yield txt
                        await asyncio.sleep(0)
            finally:
                streamer.end()
                if thread.is_alive():
                    thread.join(timeout=1.0)

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    def get_streamer(self):
        return TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
