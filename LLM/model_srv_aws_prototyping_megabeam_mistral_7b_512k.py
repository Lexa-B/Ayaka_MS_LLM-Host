# LLM/model_srv_aws_prototyping_megabeam_mistral_7b_512k.py

from LLM._base_model_handler import BaseModelHandler
from dramatic_logger import DramaticLogger

MODEL_PATH = "./LLM/aws-prototyping/MegaBeam-Mistral-7B-512k"

class ModelHandler(BaseModelHandler):
    """
    MegaBeam Mistral handler
    """

    def build_model_path(self) -> str:
        return MODEL_PATH

    def load_model(self):
        super().load_model()
        # The model has a chat template defined in tokenizer_config.json
        # We can verify it's loaded correctly
        if self.tokenizer.chat_template:
            DramaticLogger["Normal"]["info"](
                "[MegaBeam Mistral Handler] Using model's built-in chat template."
            )
        else:
            DramaticLogger["Normal"]["warning"](
                "[MegaBeam Mistral Handler] Model's chat template not found, using fallback."
            )
            # Fallback that matches Mistral's format
            self.tokenizer.chat_template = """{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}

{{- bos_token }}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif %}
    {%- if message['role'] == 'user' %}
        {%- if loop.first and system_message is defined %}
            {{- ' [INST] ' + system_message + '\n\n' + message['content'] + ' [/INST]' }}
        {%- else %}
            {{- ' [INST] ' + message['content'] + ' [/INST]' }}
        {%- endif %}
    {%- elif message['role'] == 'assistant' %}
        {{- ' ' + message['content'] + eos_token}}
    {%- else %}
        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}
    {%- endif %}
{%- endfor %}
""" 