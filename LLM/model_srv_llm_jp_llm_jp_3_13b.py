# LLM/model_srv_llm_jp_llm_jp_3_13b.py

from LLM._base_model_handler import BaseModelHandler
from dramatic_logger import DramaticLogger

MODEL_PATH = "./LLM/llm-jp/llm-jp-3-13b"

class ModelHandler(BaseModelHandler):
    """
    LLM-jp specific handler
    """

    def build_model_path(self) -> str:
        return MODEL_PATH

    def load_model(self):
        super().load_model()
        # If the model has a chat template defined in tokenizer_config.json
        # We can verify it's loaded correctly
        if (False): # self.tokenizer.chat_template: ## Currently the chat template causes issues, so we're not using it
            DramaticLogger["Normal"]["info"](
                "[LLM-jp Handler] Using model's built-in chat template."
            )
        else:
            DramaticLogger["Normal"]["warning"](
                "[LLM-jp Handler] Model's chat template not found, using fallback."
            )
            # Define a chat template using the model's special tokens
            self.tokenizer.chat_template = """{% if messages and messages[0]['role'] == 'system' -%}
### システム: {{ messages[0]['content'] | trim }}

{% set messages = messages[1:] -%}
{% endif -%}

{% for message in messages -%}
{% if message['role'] == 'user' -%}
### ユーザー: {{ message['content'] | trim }}

{% elif message['role'] == 'assistant' -%}
### アシスタント:
{{ message['content'] | trim }}

{% endif -%}
{% endfor -%}
{% if add_generation_prompt -%}
### アシスタント: {% endif -%}"""
          
#             self.tokenizer.chat_template = """{{ bos_token }}
# {%- if messages[0]['role'] == 'system' %}
#     {%- set system_message = messages[0]['content']|trim %}
#     {%- set messages = messages[1:] %}
# {%- else %}
#     {%- set system_message = "" %}
# {%- endif %}

# システム: {{ system_message }}{{ sep_token }}

# {%- for message in messages %}
#     {%- if message['role'] == 'user' %}
# ユーザー: {{ message['content'] | trim }}{{ sep_token }}
#     {%- elif message['role'] == 'assistant' %}
# アシスタント: {{ message['content'] | trim }}{{ sep_token }}
#     {%- endif %}
# {%- endfor %}
# {%- if add_generation_prompt %}
# アシスタント:"""

        DramaticLogger["Normal"]["info"](
            "[LLM-jp Handler] Model loaded with Japanese chat template."
        ) 