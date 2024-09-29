import os

from job import print_job_list

OUTPUT_DIR = "output"


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
