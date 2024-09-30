import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from llama_index.core import VectorStoreIndex
from llama_index.readers.web import WholeSiteReader

lock = threading.Lock()


async def crawl_indeed(index: VectorStoreIndex, pages: int = 10, max_depth: int = 10, search_query: str = ""):
    def get_base_url(start: int = 0):
        return f"https://br.indeed.com/jobs?q={search_query}&start={start}"

    def crawl(page: int, prefix: str):
        logging.info(f"Starting crawl: page={page}, prefix={prefix}")

        scraper = WholeSiteReader(
            prefix=prefix,
            max_depth=max_depth
        )
        documents = scraper.load_data(
            base_url=get_base_url(page))

        for document in list(documents):
            with lock:
                index.insert(document)

        logging.info(f"Finished crawl: page={page}, prefix={prefix}")

    urls = ["https://br.indeed.com/pagead", "https://br.indeed.com/rc"]

    max_workers = 4
    tasks = []
    loop = asyncio.get_event_loop()

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
