# LLM/model_srv_llm_jp_llm_jp_3_13b_instruct.py

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
        # The model already has a chat template defined in tokenizer_config.json
        # We can verify it's loaded correctly
        if self.tokenizer.chat_template:
            DramaticLogger["Normal"]["info"](
                "[LLM-jp-Instruct Handler] Using model's built-in chat template."
            )
        else:
            DramaticLogger["Normal"]["warning"](
                "[LLM-jp-Instruct Handler] Model's chat template not found, using fallback."
            )
            # Fallback that matches the model's built-in template format
            self.tokenizer.chat_template = """{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '\n\n### 指示:\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。' }}{% elif message['role'] == 'assistant' %}{{ '\n\n### 応答:\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n\n### 応答:\n' }}{% endif %}{% endfor %}""" 