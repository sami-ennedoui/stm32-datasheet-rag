"""Central configuration.

All paths and model names live here so the rest of the code stays small.
Values can be overridden with environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load a local .env if present. Also load the shared smolagent .env so the
# HF_TOKEN that already exists on this machine is reused without copying it.
load_dotenv()
_SMOLAGENT_ENV = Path.home() / "smolagent" / ".env"
if _SMOLAGENT_ENV.exists():
    load_dotenv(_SMOLAGENT_ENV, override=False)

# Project layout.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VECTOR_DIR = Path(os.getenv("STM32_VECTOR_DIR", str(ROOT / "vectorstore")))
PDF_PATH = Path(os.getenv("STM32_PDF_PATH", str(DATA_DIR / "rm0433.pdf")))

# Chroma collection name.
COLLECTION = os.getenv("STM32_COLLECTION", "stm32h7_docs")

# Chunking. Token counts are approximated by characters to stay dependency free.
CHUNK_CHARS = int(os.getenv("STM32_CHUNK_CHARS", "1400"))
CHUNK_OVERLAP = int(os.getenv("STM32_CHUNK_OVERLAP", "200"))

# Embedding model. A small local sentence-transformers model so the demo runs
# without any API key.
EMBED_MODEL = os.getenv("STM32_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval.
TOP_K = int(os.getenv("STM32_TOP_K", "5"))

# LLM for answer generation. Hugging Face inference API, same family the
# smolagent project already uses.
HF_TOKEN = os.getenv("HF_TOKEN")
ANSWER_MODEL = os.getenv("STM32_ANSWER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
CODE_MODEL = os.getenv("STM32_CODE_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
