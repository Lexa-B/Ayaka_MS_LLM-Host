# LLM/model_srv_llama_7b_instruct.py

from LLM._base_model_handler import BaseModelHandler
from dramatic_logger import DramaticLogger

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
        # The model already has a chat template defined in tokenizer_config.json
        # We can verify it's loaded correctly
        if self.tokenizer.chat_template:
            DramaticLogger["Normal"]["info"](
                "[Llama Handler] Using model's built-in chat template."
            )
        else:
            DramaticLogger["Normal"]["warning"](
                "[Llama Handler] Model's chat template not found, using fallback."
            )
            # Simplified version of the model's built-in template
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