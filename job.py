from pydantic import BaseModel, Field
from typing import List


class JobListing(BaseModel):
    title: str = Field(..., description="Job title")
    description: str = Field(...,
                             description="A brief description of the job")
    fit: float = Field(...,
                       description="The fit score based on the applicant's resume")
    job_view_url: str = Field(
        ..., description="Url to the job listing view page")
    fit_reason: str = Field(
        ..., description="Reason why the job is a great fit based on applicant resume")


class JobList(BaseModel):
    items: List[JobListing]

# Printa a lista de jobs que se encaixam no resume
def print_job_list(job_list: JobList, file=None):
    for job in job_list.items:
        print(
            f"""{job.title} ({job.fit})\nDescription: {job.description}\nReason: {job.fit_reason}\nLink: {job.job_view_url}\n\n""", file=file)
