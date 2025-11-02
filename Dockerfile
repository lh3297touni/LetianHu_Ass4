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

# ---- deps (cache-friendly) ----
COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen

# spaCy 模型装进 uv 的虚拟环境
RUN uv pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl

# ---- app source ----
COPY main.py /code/
COPY app /code/app
# ✅ 新增：把 helper_lib 也复制进去（GAN/CNN 都在这里）
COPY helper_lib /code/helper_lib

# ✅ 修正：创建 data 目录（而不是 dat）；顺便把子目录建好
RUN mkdir -p /code/data/gan /code/data/cifar

EXPOSE 8000

# Start
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
