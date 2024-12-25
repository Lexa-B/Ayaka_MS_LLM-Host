# LLM/model_srv_mistralai_ministral_3b_instruct.py

from LLM._base_model_handler import BaseModelHandler
from dramatic_logger import DramaticLogger

MODEL_PATH = "./LLM/mistralai/Ministral-3b-instruct"

class ModelHandler(BaseModelHandler):
    """
    Ministral instruct model handler
    """

    def build_model_path(self) -> str:
        return MODEL_PATH

    def load_model(self):
        super().load_model()
        # The model has a chat template defined in tokenizer_config.json
        # We can verify it's loaded correctly
        if self.tokenizer.chat_template:
            DramaticLogger["Normal"]["info"](
                "[Ministral Handler] Using model's built-in chat template."
            )
        else:
            DramaticLogger["Normal"]["warning"](
                "[Ministral Handler] Model's chat template not found, using fallback."
            )
            # Fallback that matches the model's built-in template
            self.tokenizer.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\n' }}{% endif %}""" 