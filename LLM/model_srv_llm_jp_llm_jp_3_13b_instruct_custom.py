# LLM/model_srv_llm_jp_llm_jp_3_13b_instruct_custom.py

from LLM._base_model_handler import BaseModelHandler
from dramatic_logger import DramaticLogger

MODEL_PATH = "./LLM/llm-jp/llm-jp-3-13b-instruct"

class ModelHandler(BaseModelHandler):
    """
    LLM-jp-instruct specific handler
    """

    def build_model_path(self) -> str:
        return MODEL_PATH

    def load_model(self):
        super().load_model()
        DramaticLogger["Normal"]["warning"](
            "[LLM-jp-Instruct Handler] Using custom chat template."
        )
        # Using custom chat template
        self.tokenizer.chat_template = """{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '\n\n### 指示:\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\n\n### 応答:\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n\n### 応答:\n' }}{% endif %}{% endfor %}""" 
