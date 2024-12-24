# LLM/model_srv_mistralai_mistral_7b_instruct_v0_3.py

from LLM._base_model_handler import BaseModelHandler
from typing import List, Dict
from dramatic_logger import DramaticLogger

## ===========================================-----------============================================
## ------------------------------------------- CONSTANTS -------------------------------------------
## ===========================================-----------============================================
# The explicit relative path on storage for this Mistral model
MODEL_PATH = "./LLM/mistralai/Mistral-7B-Instruct-v0.3"


## ===========================================-------============================================
## ------------------------------------------- CLASS -------------------------------------------
## ===========================================-------============================================
class ModelHandler(BaseModelHandler):
    """
    Mistral-specific handler.
    """

    def build_model_path(self) -> str:
        """
        Return the explicit path for Mistral-7B v0.3
        """
        return MODEL_PATH

    def preprocess_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Mistral requires system messages to be turned into user messages
        to avoid role ordering issues.
        """
        for msg in messages:
            if msg.get("role") == "system":
                msg["role"] = "user"
        return messages

    def load_model(self):
        """
        Load the model using the base method and set Mistral-specific chat template.
        """
        super().load_model()  # Loads model + sets a default fallback template

        # Set Mistral-specific chat template
        self.tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}
{% set system_message = messages[0]['content'] %}
{% set messages = messages[1:] %}
{% endif %}
<s>
{% if system_message %}
[INST] {{ system_message }}
{% endif %}
{% for message in messages %}
{% if message['role'] == 'user' %}[INST] {{ message['content'] }}[/INST]
{% elif message['role'] == 'assistant' %}{{ message['content'] }}
{% endif %}
{% endfor %}
</s>
"""
        DramaticLogger["Normal"]["debug"](f"[Mistral Handler] Loaded Mistral model with custom template.")
