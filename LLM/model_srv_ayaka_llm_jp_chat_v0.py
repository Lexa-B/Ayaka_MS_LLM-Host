# LLM/model_srv_llm_jp_llm_jp_3_13b_instruct.py

from LLM._base_model_handler import BaseModelHandler

MODEL_PATH = "./LLM/llm-jp/llm-jp-3-13b-instruct3"

class ModelHandler(BaseModelHandler):
    """
    LLM-jp-instruct specific handler
    """

    def build_model_path(self) -> str:
        return MODEL_PATH

    def preprocess_messages(self, messages):
        """
        Construct the exact prompt format the model expects.
        Checks for add_response_marker in either the message or its additional_kwargs.
        """
        if len(messages) != 1 or messages[0]['role'] not in ['user', 'human']:
            raise ValueError("This model only accepts a single message with role 'user' or 'human'")

        content = messages[0]['content']
        
        # Debug: Print the message structure
        print("Message:", messages[0])
        print("Has add_response_marker?", 'add_response_marker' in messages[0])
        print("Has additional_kwargs?", 'additional_kwargs' in messages[0])
        if 'additional_kwargs' in messages[0]:
            print("additional_kwargs:", messages[0]['additional_kwargs'])
        
        # Try top level first
        add_response_marker = messages[0].get('add_response_marker')
        if add_response_marker is None:
            # If not at top level, try additional_kwargs
            add_response_marker = messages[0].get('additional_kwargs', {}).get('add_response_marker', True)
        
        print("Final add_response_marker value:", add_response_marker)

        # Create exact format as a raw string
        raw_prompt = f"以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{content}"
        
        if add_response_marker:
            raw_prompt += "\n\n### 応答:\n"

        return [{
            'role': 'user',
            'content': raw_prompt
        }]

    def load_model(self):
        super().load_model()
        # Override any chat template with a simple passthrough
        self.tokenizer.chat_template = "{{bos_token}}{% for message in messages %}{{message['content']}}{% endfor %}" 