# LLM/model_srv_deepseek_ai_deepseek_llm_7b_base.py

from LLM._base_model_handler import BaseModelHandler
from dramatic_logger import DramaticLogger

MODEL_PATH = "./LLM/deepseek-ai/deepseek-llm-7b-base"

class ModelHandler(BaseModelHandler):
    """
    DeepSeek base model handler
    """

    def build_model_path(self) -> str:
        return MODEL_PATH

    def load_model(self):
        super().load_model()
        # The model might have a chat template defined in tokenizer_config.json
        # We can verify it's loaded correctly
        if self.tokenizer.chat_template:
            DramaticLogger["Normal"]["info"](
                "[DeepSeek Handler] Using model's built-in chat template."
            )
        else:
            DramaticLogger["Normal"]["warning"](
                "[DeepSeek Handler] Model's chat template not found, using fallback."
            )
            # Define a chat template using the model's special tokens
            self.tokenizer.chat_template = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

System: {{ system_message }}{{ eos_token }}

{%- for message in messages %}
    {%- if message['role'] == 'user' %}
User: {{ message['content'] | trim }}{{ eos_token }}
    {%- elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] | trim }}{{ eos_token }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
Assistant:""" 