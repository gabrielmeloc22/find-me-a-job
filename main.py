import os
import dotenv

from llama_index.readers.web import WholeSiteReader
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext, Settings
from llama_index.core.llms.utils import LLMType

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pydantic import BaseModel, Field

from multiprocess.pool import Pool
from multiprocessing import Lock


dotenv.load_dotenv()

INDEX_PATH = "data"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def get_llm_model() -> LLMType:

    model = Gemini(model="models/gemini-1.5-pro")
    # model = Ollama(model=LLM_MODEL, request_timeout=360.0)

    return model


def config():
    Settings.chunk_size = 2048
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL)
    Settings.llm = get_llm_model()


def get_index() -> VectorStoreIndex:
    if not os.path.exists(INDEX_PATH):
        index = VectorStoreIndex([])
        crawl_indeed(index=index, pages=1)

        index.storage_context.persist(persist_dir=INDEX_PATH)

        return index

    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)

    return load_index_from_storage(storage_context)


def crawl_indeed(index: VectorStoreIndex, pages: int = 10, max_depth: int = 10):
    def get_base_url(start: int = 0):
        return f"https://br.indeed.com/jobs?q=software+engineer&start={start}"

    def crawl(page: int, prefix: str):
        scraper = WholeSiteReader(
            prefix=prefix,
            max_depth=max_depth
        )
        documents = scraper.load_data(
            base_url=get_base_url(page))

        for document in list(documents):
            index.insert(document)

    for page in range(0, pages * 10, 10):
        crawl(page, prefix="https://br.indeed.com/pagead")
        crawl(page, prefix="https://br.indeed.com/rc")


class JobListing(BaseModel):
    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Job description")
    fit: float = Field(...,
                       description="The fit score based on the applicant resume")


def main():
    config()

    TOP_N = 10

    # reranker = FlagEmbeddingReranker(
    #     top_n=TOP_N,
    #     model="BAAI/bge-reranker-large",
    # )

    index = get_index()

    sllm = Settings.llm.as_structured_llm(output_cls=JobListing)
    query_engine = index.as_query_engine(
        # llm=sllm,
        similarity_top_k=10,
        response_mode="tree_summarize",
        # node_postprocessors=reranker
    )

    """ fake resume for testing """
    result = query_engine.query(
        """List the top 10 jobs that fit this resume alongside a brief description of the job, include the job listing link:

            Johnathan M. Carter
            1234 Maple Street, Apt 6B, Springfield, IL 62701
            Phone: (555) 123-4567 | Email: john.carter@email.com
            LinkedIn: linkedin.com/in/johnathanmcarter | GitHub: github.com/john-carter-dev

            Professional Summary
            Results-driven software engineer with 6+ years of experience in designing and developing scalable web applications and services. Adept in full-stack development, with a focus on optimizing application performance and user experiences. Proven track record of delivering complex projects on time while collaborating effectively across teams.

            Work Experience
            Senior Software Engineer
            Techify Solutions – Chicago, IL | June 2021 – Present

            Lead a team of 5 engineers to build a cloud-based platform for managing large-scale financial data, reducing data processing time by 30%.
            Developed key microservices using Node.js, React, and AWS Lambda, improving system scalability.
            Integrated third-party APIs, such as Stripe and Twilio, to enhance payment and communication features.
            Collaborated with product managers to prioritize features, resulting in a 20% increase in user retention.
            Software Engineer
            ByteWorks – St. Louis, MO | July 2018 – May 2021

            Engineered web applications using the MERN stack (MongoDB, Express, React, Node.js) for e-commerce platforms, reducing load times by 40%.
            Implemented RESTful APIs and GraphQL endpoints to streamline client-server communication.
            Automated testing using Jest and Cypress, improving code reliability and reducing bug reports by 15%.
            Mentored junior developers, providing guidance on best practices in version control, testing, and code reviews.
            Junior Software Developer
            CodeCraft Innovations – Indianapolis, IN | August 2016 – June 2018

            Contributed to the development of internal tools using Python and Django to automate business workflows.
            Collaborated with designers to improve UI/UX, resulting in a 25% boost in user engagement.
            Conducted code reviews and refactoring sessions to maintain high code quality standards.
            Education
            Master of Science in Computer Science
            University of Illinois at Urbana-Champaign | 2016

            GPA: 3.8/4.0
            Focus: Distributed Systems, Data Structures, Cloud Computing
            Bachelor of Science in Computer Engineering
            Purdue University, West Lafayette | 2014

            GPA: 3.7/4.0
            Activities: ACM Programming Team, Robotics Club
            Technical Skills
            Languages: JavaScript, TypeScript, Python, Java, SQL
            Web Technologies: React, Redux, Node.js, Express, GraphQL
            Cloud Platforms: AWS (Lambda, S3, EC2), Google Cloud
            Databases: MongoDB, PostgreSQL, MySQL
            Tools: Docker, Kubernetes, Jenkins, Git, Terraform
            Testing Frameworks: Jest, Mocha, Cypress, Selenium
            Certifications
            AWS Certified Solutions Architect – Associate | 2020
            Google Cloud Professional Cloud Architect | 2021
            Projects
            Expense Tracker App – Full-Stack Application

            Developed a personal finance tracker using React, Node.js, and MongoDB, allowing users to log expenses and generate reports.
            Deployed the application on AWS using Docker and Kubernetes.
            Real-Time Chat Application – WebSocket-based Messaging App

            Built a real-time chat application using React and WebSockets, featuring group chats, message notifications, and file sharing.
            Integrated Redis for session management and message queueing.
            Interests & Volunteer Work
            Passionate about open-source contributions, with regular participation in Hacktoberfest and maintaining a small collection of npm packages.
            Volunteer mentor at Code for Good, helping underrepresented students learn programming and computer science basics.
            """)

    print(result)


if __name__ == '__main__':
    main()
