# LLM/model_srv_llama_7b_instruct.py

from LLM._base_model_handler import BaseModelHandler
from loguru import logger

MODEL_PATH = "./LLM/meta-llama/Llama-3.2-3B-Instruct"

class ModelHandler(BaseModelHandler):
    """
    Llama-specific handler
    """

    def build_model_path(self) -> str:
        return MODEL_PATH

    # If no special role transform is required, we skip `preprocess_messages()`.
    # If we want a special chat template, we override load_model() partially:
    def load_model(self):
        super().load_model()
        # Example: set a llama-ish template
        self.tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
<|USER|> {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
<|ASSISTANT|> {{ message['content'] }}
{% endif %}
{% endfor %}
"""
        logger.info("[Llama Handler] Llama model loaded with custom template.")
