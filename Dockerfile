# Use a slim Python base image
FROM python:3.12-slim-bookworm

# Build tools
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Workdir
WORKDIR /code

# Dependencies first (cache-friendly)
COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen

# Install spaCy model INTO THE UV ENV
RUN uv pip install \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl

# App code
COPY main.py /code/
COPY app /code/app

RUN mkdir -p /code/dat
EXPOSE 8000

# Start
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
