# To Run: 
* Make sure the CUDA tools and driver are installed 

## To run locally:
1. Build the venv & install the python requirements:
> ./Scripts/RebuildVenvLx.sh
2. Run the app:
> uvicorn LLM-Host:app --host 0.0.0.0 --port 41443 --reload

## To run in Docker:
1. Build the Docker image:
> docker build -t [[IMAGENAME]] .
2. Run the Docker container:
> docker run -it --rm --gpus all -p 41443:41443 [[IMAGENAME]]

# To add a new model:
1. Add a new handler file in the LLM/ directory

# Layout
* LLM/ - All model-specific code
* LLM-Host.py - The FastAPI app
* model_service.py - The model service that loads the model handler
* model_srv_{model_name}.py - The model-specific handler
* _base_model_handler.py - The base class for all model handlers

## Consists of four key pieces:

    main.py:
        Contains the FastAPI application, route definitions, and minimal “controller” logic.
        Delegates all model handling to a new “service” layer.

    model_service.py:
        Coordinates the loading of different model handlers based on the requested model name.
        Maintains references to whichever model is currently loaded.
        Provides shared logic and bridging between the FastAPI endpoints and the specific model implementations.

    models/base_model_handler.py (or something similar):
        Defines a base handler class (BaseModelHandler) that sets out the interface for loading a model, preprocessing messages, applying templates, generating text, streaming, etc.
        Common logic can live here.

    models/model_srv_mistralai_Mistral-7B-Instruct-v0.3.py:
        A model-specific file with a subclass of BaseModelHandler.
        Contains only the Mistral-specific code, templates, message preprocessing, tokenization quirks, etc.
        In the future, you can create parallel files (e.g., model_srv_llama_Llama-3B-Instruct.py, model_srv_falcon_Falcon-7B.py, etc.) for each model type.



                   ┌────────────────────────────┐
                   │         main.py            │
                   │----------------------------│
(1)  FastAPI(...) -> + app = FastAPI(...)       │
                   │                            │
     + /initialize ---- calls --> model_service.initialize_model(params)
                   │                            │
     + /generate   ---- calls --> model_service.generate_response(messages)
                   │                            │
     + /stream     ---- calls --> model_service.stream_response(messages)
                   │                            │
     + /status     ---- calls --> model_service.get_status()
                   └────────────────────────────┘
                              ▲
                              │
                              │
             ┌───────────────────────────────────────────────────────────┐
             │                   model_service.py                        │
             │-----------------------------------------------------------│
             │ Global or class-based structure (e.g. ModelService class) │
             │ --------------------------------------------------------- │
             │   + model, tokenizer, model_params, etc.                  │
             │   + def initialize_model(params):                          │
             │       - loads the correct model, checks quant config, etc.│
             │       - sets chat_template or special transforms          │
             │                                                           │
             │   + def generate_response(messages):                       │
             │       - handle Mistral-specific role transformations      │
             │       - apply chat template                               │
             │       - call model.generate(...)                          │
             │       - return text                                       │
             │                                                           │
             │   + def stream_response(messages):                         │
             │       - streaming version of above logic                  │
             │       - uses streamer & yields tokens                     │
             │                                                           │
             │   + def get_status():                                     │
             │       - returns whether model is loaded, which model, etc.│
             └────────────────────────────────────────────────────────────┘

------------------------------------------------------------------

┌────────────────────────────┐         ┌─────────────────────────────┐
│         main.py            │         │        model_service.py     │
│----------------------------│         │-----------------------------│
│ - FastAPI app definition   │         │ - Shared utilities          │
│ - Endpoint definitions     │         │ - Model registry            │
│ - Delegates requests to    │◄───────►│ - Common processing logic   │
│   appropriate model handler│         └─────────────────────────────┘
└────────────────────────────┘                      ▲
                                                    │
                                                    │
                       ┌────────────────────────────┴─────────────────────────────┐
                       │                         LLM/                             │
                       │----------------------------------------------------------│
                       │                                                          │
                       │   ┌──────────────────────────────────────────────────┐   │
                       │   │ model_srv_mistralai_Mistral-7B-Instruct-v0.3.xxx │   │ 
					   │   │--------------------------------------------------│   │
                       │   │ - Model-specific configurations                  │   │
                       │   │ - Override methods for preprocessing             │   │
                       │   │ - Override methods for postprocessing            │   │
                       │   │ - Define chat templates                          │   │
                       │   └──────────────────────────────────────────────────┘   │
                       │                                                          │
                       │   ┌───────────────────────────────────────────────┐      │
                       │   │ model_srv_llama_Llama-3.2-3B-Instruct.xxx     │      │
                       │   │-----------------------------------------------│      │
                       │   │ - Model-specific configurations               │      │
                       │   │ - Override methods for preprocessing          │      │
                       │   │ - Override methods for postprocessing         │      │
                       │   │ - Define chat templates                       │      │
                       │   └───────────────────────────────────────────────┘      │
                       │                                                          │
                       │   ... (additional model files) ...                       │
                       └──────────────────────────────────────────────────────────┘


project_root/
│
├── LLM-Host.py
├── model_service.py
├── requirements.txt
│
├── LLM/
│   ├── __init__.py
│   ├── _base_model_handler.py
│   ├── model_srv_mistralai_mistral_7b_instruct_v0_3.py
│   ├── model_srv_llama_llama_3_2_3b_instruct.py
│   ├── ... (additional model handler files)
│   │
│   ├── mistralai/
│   │   └── Mistral-7B-Instruct-v0.3/
│   │       ├── tokenizer/
│   │       ├── config.json
│   │       └── ... (model files)
│   ├── llama/
│   │   └── Llama-3.2-3B-Instruct/
│   │       ├── tokenizer/
│   │       ├── config.json
│   │       └── ... (model files)
│   └── ... (additional model directories)
│
└── Dev/
    └── Development-only-files
