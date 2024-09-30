from llama_index.core import SummaryIndex, SimpleDirectoryReader, get_response_synthesizer

import re

from utils import config

config()

# Diretório onde fica armazenado o arquivo de entrada
INPUT_DIR = "input"

# Retorna a query de busca para o arquivo de entrada
def get_resume_query(resume_path: str) -> str:
    reader = SimpleDirectoryReader(input_files=[resume_path])
    documents = reader.load_data(show_progress=True)

    index = SummaryIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    search_query = query_engine.query(
        """Create a search query for this resume general area of expertise only including letters and with a 5 words limit, prefer less words and try to be generic
        example: mid-level javascript software engineer
        """)

    clean = re.sub(' +', '+', str(search_query))

    return clean

# Retorna o caminho do arquivo de entrada
def get_resume_path(resume_filename: str) -> str:
    return f"{INPUT_DIR}/{resume_filename}"

# Retorna o conteúdo do arquivo de entrada
def get_resume_str(resume_filename: str) -> str:
    resume_path = get_resume_path(resume_filename)

    with open(resume_path, "r", encoding="utf-8") as f:
        return f.read()
