## Setup project

- install `uv` package manager from doc `https://github.com/astral-sh/uv?tab=readme-ov-file#installation`
- copy `.env.example` to `.env` and fill in your Google API keytkj
- alternatively, setup a local ollama model (you'll need to change the model name in the `get_llm_model` function)
- run `uv pip install -r pyproject.toml` to install dependencies

## Run

- create a resume in `input` dir
- run `uv run main.py` to run the main.py file
