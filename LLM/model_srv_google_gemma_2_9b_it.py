from LLM._base_model_handler import BaseModelHandler
from typing import List, Dict
from dramatic_logger import DramaticLogger
import torch


# The explicit relative path on storage for the Gemma-2-9b-it model
MODEL_PATH = "./LLM/google/gemma-2-9b-it"


class ModelHandler(BaseModelHandler):
    """
    Google Gemma 2-9B IT-specific handler.
    """

    def build_model_path(self) -> str:
        """
        Return the explicit path for Gemma-2-9b-it.
        """
        return MODEL_PATH

    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Implement any preprocessing required by Gemma-2-9b-it.
        For Gemma, ensure that system messages are not present as they are not supported.
        If a system role is detected, raise an exception.
        Additionally, ensure that conversation roles alternate between user and assistant.
        """
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "system":
                raise ValueError("System role is not supported for Gemma-2-9b-it.")
            if idx % 2 == 0 and role != "user":
                raise ValueError("Conversation roles must alternate between user and assistant.")
            if idx % 2 != 0 and role != "assistant":
                raise ValueError("Conversation roles must alternate between user and assistant.")
        return messages

    def load_model(self):
        """
        Load the Gemma-2-9b-it model using the base method and set model-specific configurations.
        """
        # From config.json, we know it uses bfloat16
        self.params.quant_dtype = "bfloat16"
        
        super().load_model()

        # Configure tokenizer settings based on tokenizer_config.json
        self.tokenizer.add_special_tokens({
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]
        })

        # Set padding settings
        self.tokenizer.padding_side = "left"  # Gemma typically uses left padding
        self.tokenizer.pad_token = "<pad>"
        
        # Ensure clean tokenization settings from config
        self.tokenizer.clean_up_tokenization_spaces = False
        self.tokenizer.spaces_between_special_tokens = False

        # Apply the custom chat template from the provided config
        self.tokenizer.chat_template = (
            """{{ bos_token }}"""    # Start with BOS token
            """{% if messages[0]['role'] == 'system' %}"""
            """{{ raise_exception('System role not supported') }}"""
            """{% endif %}"""
            """{% for message in messages %}"""
            """{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"""
            """{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"""
            """{% endif %}"""
            """{% if (message['role'] == 'assistant') %}{% set role = 'model' %}"""
            """{% else %}{% set role = message['role'] %}{% endif %}"""
            """{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"""
            """{% endfor %}"""
            """{% if add_generation_prompt %}"""
            """{{ '<start_of_turn>model\n' }}"""
            """{% endif %}"""
        )

        DramaticLogger["Normal"]["debug"](f"[Gemma-2-9b-it Handler] Loaded Gemma-2-9b-it model with custom chat template.")

    def get_terminators(self) -> List[int]:
        """
        Override get_terminators to use Gemma's specific EOS token.
        """
        # From special_tokens_map.json, we know <eos> is the primary terminator
        terminators = [self.tokenizer.eos_token_id]
        return terminators 