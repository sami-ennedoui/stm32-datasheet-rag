# STM32 datasheet RAG service.
# Python 3.12 because torch and sentence-transformers do not yet ship wheels
# for Python 3.14.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Keep the model cache inside the image layer at build, writable at runtime.
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install the CPU-only torch first to keep the image smaller, then the rest.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# The PDF and vector store are not baked in. At deploy time run the ingest step
# once (see README) so the vector store is built on a persistent disk, or bake
# it by adding a build step that runs: python -m app.ingest
EXPOSE 8000

# Bind to the platform provided port if present, default 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
