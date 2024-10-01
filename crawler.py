import sys
import os

from llama_index.core import VectorStoreIndex
from llama_index.readers.web import WholeSiteReader


def crawl_indeed(index: VectorStoreIndex, pages: int = 10, max_depth: int = 10, search_query: str = ""):
    # Busca vagas de emprego no site Indeed e insere os dados no índice de vetores
    # Parâmetros:
    # - index: O índice onde os documentos serão armazenados
    # - pages: Número de páginas a serem raspadas (cada página contém 10 resultados)
    # - max_depth: Profundidade máxima do crawler (número de links que ele segue)
    # - search_query: A consulta de pesquisa usada para buscar vagas

    stdout = sys.stdout

    def get_base_url(start: int = 0):
        return f"https://br.indeed.com/jobs?q={search_query}&start={start * 10}"

    def crawl(page: int, prefix: str):
        print(f"Starting crawl: page={page}")

        sys.stdout = open(os.devnull, 'w')

        # Cria um scraper com base no prefixo e profundidade máxima
        scraper = WholeSiteReader(
            prefix=prefix,
            max_depth=max_depth
        )

        documents = scraper.load_data(
            base_url=get_base_url(page))

        # Insere cada documento no índice de vetores de maneira thread-safe
        for document in list(documents):
            index.insert(document)

        sys.stdout = stdout
        print(f"Finished crawl: page={page}")

    urls = ["https://br.indeed.com/pagead", "https://br.indeed.com/rc"]

    for page in range(0, pages + 1):
        for url in urls:
            crawl(page=page, prefix=url)
