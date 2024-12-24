# LLM/base_model_handler.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from typing import List, Dict
import asyncio
from fastapi.responses import StreamingResponse
from loguru import logger

class BaseModelHandler:
    """
    The base class that implements shared logic: 
    generation, streaming, decode, etc.
    """

    def __init__(self, params):
        self.params = params
        self.model = None
        self.tokenizer = None
        
        # Let subclasses define or override build_model_path()
        self.model_path = self.build_model_path()

        # If a seed is provided, fix it for reproducibility
        if self.params.seed is not None:
            torch.manual_seed(self.params.seed)

        # Load the model and tokenizer
        self.load_model()  
        logger.info(f"[BaseModelHandler] init done. Model path: {self.model_path}")

    def build_model_path(self) -> str:
        """
        By default, returns a fallback path (if a subclass does not override).
        Subclasses typically override this method or define a constant 
        to specify their custom path, e.g. ./LLM/Mistralai/Mistral-7B.
        """
        logger.warning("No explicit build_model_path() override found; using default.")
        return f"./LLM/{self.params.model}"

    def load_model(self):
        """
        A mostly generic approach:
          - Build BitsAndBytesConfig from self.params
          - Load tokenizer and model
          - Possibly set a default chat_template
        Subclasses can override if there's something truly unique.
        """
        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.params.quant_4bit,
            bnb_4bit_quant_type=self.params.quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.params.quant_dtype)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
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
            logger.info("GPU is available. Moving model to GPU.")
            self.model.to("cuda")

        logger.info("[BaseModelHandler] Default load_model() done.")

    # ----------------------------------------------------------------
    #  PREPROCESS
    # ----------------------------------------------------------------
    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return messages  # By default, no special transformation

    # ----------------------------------------------------------------
    #  APPLY CHAT TEMPLATE
    # ----------------------------------------------------------------
    def apply_chat_template(self, messages: List[Dict[str, str]]):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

    # ----------------------------------------------------------------
    #  PREPARE INPUT
    # ----------------------------------------------------------------
    def prepare_input(self, messages: List[Dict[str, str]]):
        processed_msgs = self.preprocess_messages(messages)
        input_ids = self.apply_chat_template(processed_msgs)
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
        return self.tokenizer.decode(
            outputs[0][prompt_len:],
            skip_special_tokens=True
        )

    # ----------------------------------------------------------------
    #  POSTPROCESS
    # ----------------------------------------------------------------
    def postprocess_output(self, text: str) -> str:
        if text.endswith("</s>"):
            text = text[:-4]
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
