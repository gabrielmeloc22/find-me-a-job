import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from llama_index.core import VectorStoreIndex
from llama_index.readers.web import WholeSiteReader

# Criação de um lock para sincronizar o acesso ao índice (evitar condições de corrida)
lock = threading.Lock()

# Busca vagas de emprego no site Indeed e insere os dados no índice de vetores
# Parâmetros:
# - index: O índice onde os documentos serão armazenados
# - pages: Número de páginas a serem raspadas (cada página contém 10 resultados)
# - max_depth: Profundidade máxima do crawler (número de links que ele segue)
# - search_query: A consulta de pesquisa usada para buscar vagas
async def crawl_indeed(index: VectorStoreIndex, pages: int = 10, max_depth: int = 10, search_query: str = ""):
    def get_base_url(start: int = 0):
        return f"https://br.indeed.com/jobs?q={search_query}&start={start}"

    def crawl(page: int, prefix: str):
        logging.info(f"Starting crawl: page={page}, prefix={prefix}")

        # Cria um scraper com base no prefixo e profundidade máxima
        scraper = WholeSiteReader(
            prefix=prefix,
            max_depth=max_depth
        )
        
        documents = scraper.load_data(
            base_url=get_base_url(page))
            
        # Insere cada documento no índice de vetores de maneira thread-safe
        for document in list(documents):
            with lock:
                index.insert(document)

        logging.info(f"Finished crawl: page={page}, prefix={prefix}")

    urls = ["https://br.indeed.com/pagead", "https://br.indeed.com/rc"]

    # Define o número máximo de threads simultâneas
    max_workers = 4
    tasks = []

    # Obtém o loop de eventos atual para executar tarefas assíncronas
    loop = asyncio.get_event_loop()

    # Cria um pool de threads para executar o crawler de forma paralela
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for page in range(0, pages * 10, 10):
            for url in urls:
                task = loop.run_in_executor(
                    executor,
                    crawl,
                    page,
                    url,
                )
                tasks.append(task)

    await asyncio.gather(*tasks)
