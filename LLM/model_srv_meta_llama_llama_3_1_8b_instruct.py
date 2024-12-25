# LLM/model_srv_meta_llama_llama_3_1_8b_instruct.py

from LLM._base_model_handler import BaseModelHandler
from dramatic_logger import DramaticLogger

MODEL_PATH = "./LLM/meta-llama/Llama-3.1-8B-Instruct"

class ModelHandler(BaseModelHandler):
    """
    Llama 3.1 Instruct handler
    """

    def build_model_path(self) -> str:
        return MODEL_PATH

    def load_model(self):
        super().load_model()
        # If the model has a chat template defined in tokenizer_config.json
        # We can verify it's loaded correctly
        if self.tokenizer.chat_template:
            DramaticLogger["Normal"]["info"](
                "[Llama 3.1 Handler] Using model's built-in chat template."
            )
        else:
            DramaticLogger["Normal"]["warning"](
                "[Llama 3.1 Handler] Model's chat template not found, using fallback."
            )
            # Fallback that matches Llama's format
            self.tokenizer.chat_template = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

<|start_header_id|>system<|end_header_id|>

{{ system_message }}<|eot_id|>

{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}""" 