
from llama_index.core import Settings
from llama_index.core.llms.utils import LLMType

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import dotenv
import os

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def get_llm_model() -> LLMType:
    model = Gemini(model="models/gemini-1.5-pro-latest", max_tokens=1024)
    # model = Ollama(model=LLM_MODEL, request_timeout=360.0)

    return model


def config():
    # Carrega as configurações do arquivo .env,
    # define os modelos de embedding e llm,
    # e define as configurações do sistema

    dotenv.load_dotenv()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    Settings.chunk_size = 2048
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL)
    Settings.llm = get_llm_model()
