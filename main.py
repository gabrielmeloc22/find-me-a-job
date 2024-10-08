import os

from llama_index.core import (StorageContext, VectorStoreIndex,
                              get_response_synthesizer,
                              load_index_from_storage)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

from crawler import crawl_indeed
from job import JobList, print_job_list
from output import create_output_file
from resume import get_resume_path, get_resume_query, get_resume_str
from utils import config

# Inicializa configurações do programa
config()

# Diretório onde os índices serão armazenados
INDEX_DIR = "data"


def get_index(resume_filename: str, pages: int = 1) -> VectorStoreIndex:
    # Função para obter o índice de vetores (VectorStoreIndex) com base no currículo do usuário
    # Se o índice não existir, ele será criado a partir do crawling de páginas de empregos
    filename = resume_filename.split(".")[0]
    resume_path = get_resume_path(resume_filename)

    resume_index = f"{INDEX_DIR}/{filename}-{pages}"

    if not os.path.exists(resume_index):
        index = VectorStoreIndex([])
        search_query = get_resume_query(resume_path)

        crawl_indeed(index=index, pages=pages, search_query=search_query)

        index.storage_context.persist(persist_dir=resume_index)

        return index

    storage_context = StorageContext.from_defaults(persist_dir=resume_index)

    return load_index_from_storage(storage_context)


def get_resume_from_input() -> str:
    # Função para obter o currículo do usuário via input
    resume_filename = input(
        "Enter the filename of your resume: ")
    try:
        resume_str = get_resume_str(resume_filename)
        return (resume_str, resume_filename)

    except FileNotFoundError:
        print("File not found, please check the filename and try again.")


def get_pages_from_input() -> int:
    # Função para obter o número de páginas a serem buscadas do usuário via input
    try:
        return int(input("What is the maximum number of web pages you want me to search for jobs? Enter a number: "))
    except ValueError:
        print("Please enter a valid number.")


def main():
    print("Hello, im here to help you find an awesome job in no time! To continue, please copy your resume to the input folder and enter its filename\n")

    # Solicita o currículo do usuário
    resume, resume_filename = get_resume_from_input()

    print(f"Great job, your resume was loaded successfully. Now, let's get started!\n")

    # Solicita o número de páginas de busca
    page_num = get_pages_from_input()

    # Obtém (ou cria) o índice de vetores com base no currículo e nas páginas buscadas
    index = get_index(resume_filename=resume_filename, pages=page_num)

    print(f"Index created, searching for jobs that fit your resume...\n")

    # Configura o recuperador de índices de vetores, definindo o número máximo de resultados similares
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=50,
    )

    # Define o template da resposta para formatar a lista de empregos de acordo com o currículo
    summary_template = PromptTemplate(
        """Given the context, the candidate resume and no other information:
        Context: {context_str}

        You should return a JobList.
        You should use the url on the document metadata, avoid using any url you find on the text content or to generate a url.
        You should create a brief description of the job
        You should use the similarity that you can find on the document to score and rank the jobs.

        Resume: {query_str}
        """)

    # Configura o sintetizador de resposta com base no template e o tipo de saída esperado
    response_synthesizer = get_response_synthesizer(
        summary_template=summary_template,
        response_mode='tree_summarize',
        output_cls=JobList,
        use_async=True,  # use async calls for better performance
    )

    # Cria o mecanismo de consulta com o recuperador e sintetizador configurados
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=50
            )
        ]
    )

    # Executa a consulta com base no currículo do usuário
    result = query_engine.query(resume)

    # Cria um arquivo de saída com a lista de empregos gerada
    file_path = create_output_file(result, resume_filename)

    print(f"Output file created at {file_path}\n")
    print("Here are some jobs that fit your resume:\n")

    # Exibe a lista de empregos gerada
    print_job_list(result)


if __name__ == '__main__':
    main()
