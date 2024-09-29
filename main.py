import os

from llama_index.readers.web import WholeSiteReader
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext, Settings
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


from pydantic import BaseModel, Field
from typing import List

from utils import config
from resume import resume, get_resume_query

config()

INDEX_DIR = "data"
INPUT_DIR = "input"
OUTPUT_DIR = "output"


def get_index(resume_filename: str, pages: int = 1) -> VectorStoreIndex:
    filename = resume_filename.split(".")[0]
    resume_path = f"{INPUT_DIR}/{resume_filename}"

    resume_index = f"{INDEX_DIR}/{filename}-{pages}"

    if not os.path.exists(resume_index):
        index = VectorStoreIndex([])
        search_query = get_resume_query(resume_path)

        crawl_indeed(index=index, pages=pages, search_query=search_query)

        index.storage_context.persist(persist_dir=resume_index)

        return index

    storage_context = StorageContext.from_defaults(persist_dir=resume_index)

    return load_index_from_storage(storage_context)


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


class JobListing(BaseModel):
    title: str = Field(..., description="Job title")
    description: str = Field(...,
                             description="A brief description of the job, max of 20 words")
    fit: float = Field(...,
                       description="The fit score based on the applicant's resume")
    url: str = Field(..., description="Url to the job listing view page, included in the document metadata")
    fit_reason: str = Field(
        ..., description="Reason why the job is a great fit based on applicant resume")


class JobList(BaseModel):
    items: List[JobListing]


def print_job_list(job_list: JobList, file=None):
    for job in job_list.items:
        print(
            f"\n{job.title} ({job.fit})\nDescription: {job.description}\n{job.fit_reason}\nLink:{job.url}\n\n", file=file)


def check_file_exists(filename: str) -> bool:
    return os.path.exists(f"{INPUT_DIR}/{filename}")


def get_iso_time():
    import datetime

    return datetime.datetime.now().isoformat(timespec='seconds')


def create_output_file(result, resume_filename: str):
    filename = resume_filename.split(".")[0]
    timestamp = get_iso_time().replace(":", "-")

    file_path = f"{OUTPUT_DIR}/{filename}-output-{timestamp}.txt"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        print_job_list(result, f)

    return file_path


def main():
    print("Hello, im here to help you find an awesome job in no time! To continue, please copy your resume to the input folder and enter its filename\n")
    resume_filename = input(
        "Enter the filename of your resume: ")

    if not check_file_exists(resume_filename):
        print("File not found, please check the filename and try again.\n")
        return

    print(f"Great job, your resume was loaded successfully. Now, let's get started!\n")

    page_num = int(
        input("What is the maximum number of web pages you want me to search for jobs? Enter a number: "))

    index = get_index(resume_filename=resume_filename, pages=page_num)

    print(f"Index created, searching for jobs that fit your resume...\n")


    query_engine = index.as_query_engine(
        output_cls=JobList,
        similarity_top_k=10,
        response_mode="tree_summarize",
        node_postprocessors=[
            SentenceEmbeddingOptimizer(
                percentile_cutoff=0.6, embed_model=Settings.embed_model),
            # FlagEmbeddingReranker(
            #     top_n=10,
            #     model="BAAI/bge-reranker-large",
            # )
        ]
    )

    result = query_engine.query(
        f"""List the top jobs that fit this resume: 
        {resume}""")

    file_path = create_output_file(result, resume_filename)

    print(f"Output file created at {file_path}\n")
    print("Here are some jobs that fit your resume:\n")

    print_job_list(result)


if __name__ == '__main__':
    main()
