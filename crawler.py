from llama_index.core import VectorStoreIndex
from llama_index.readers.web import WholeSiteReader


def crawl_indeed(index: VectorStoreIndex, pages: int = 10, max_depth: int = 10, search_query: str = ""):
    def get_base_url(start: int = 0):
        return f"https://br.indeed.com/jobs?q={search_query}&start={start}"

    def crawl(page: int, prefix: str):
        scraper = WholeSiteReader(
            prefix=prefix,
            max_depth=max_depth
        )
        documents = scraper.load_data(
            base_url=get_base_url(page))

        for document in list(documents):
            index.insert(document)

    urls = ["https://br.indeed.com/pagead", "https://br.indeed.com/rc"]

    for page in range(0, pages * 10, 10):
        for url in urls:
            crawl(page, prefix=url)
