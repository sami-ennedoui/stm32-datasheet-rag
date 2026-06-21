"""Read the STM32 PDF, split it into chunks, embed them, store in Chroma.

Run this once before starting the API:

    python -m app.ingest

It is safe to run again. The collection is reset each time.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import fitz  # PyMuPDF

from . import config
from .store import get_collection, reset_collection
from .embed import Embedder


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int  # 1-based page number in the PDF


def extract_pages(pdf_path) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text). Page numbers are 1-based."""
    doc = fitz.open(pdf_path)
    pages = []
    for index in range(doc.page_count):
        page = doc.load_page(index)
        text = page.get_text("text")
        pages.append((index + 1, text))
    doc.close()
    return pages


def _clean(text: str) -> str:
    # Collapse runs of whitespace but keep paragraph breaks readable.
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_chars: int = config.CHUNK_CHARS,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split each page into overlapping character windows.

    Chunks never cross page boundaries, so every chunk keeps one exact page
    number for citation.
    """
    chunks: list[Chunk] = []
    for page_number, raw in pages:
        text = _clean(raw)
        if len(text) < 40:
            continue
        start = 0
        part = 0
        while start < len(text):
            window = text[start : start + chunk_chars]
            chunk_id = f"p{page_number}-{part}"
            chunks.append(Chunk(chunk_id=chunk_id, text=window, page=page_number))
            part += 1
            if start + chunk_chars >= len(text):
                break
            start += chunk_chars - overlap
    return chunks


def run() -> int:
    pdf_path = config.PDF_PATH
    if not pdf_path.exists():
        print(f"error: PDF not found at {pdf_path}", file=sys.stderr)
        print("Download it first, see README.", file=sys.stderr)
        return 1

    print(f"Reading {pdf_path} ...")
    pages = extract_pages(pdf_path)
    print(f"  {len(pages)} pages")

    chunks = chunk_pages(pages)
    print(f"  {len(chunks)} chunks")

    print(f"Loading embedding model {config.EMBED_MODEL} ...")
    embedder = Embedder()

    print("Embedding chunks (this can take a few minutes the first time) ...")
    vectors = embedder.encode([c.text for c in chunks])

    print("Writing to vector store ...")
    reset_collection()
    collection = get_collection()
    batch = 500
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        collection.add(
            ids=[c.chunk_id for c in part],
            documents=[c.text for c in part],
            embeddings=[v for v in vectors[i : i + batch]],
            metadatas=[{"page": c.page, "chunk_id": c.chunk_id} for c in part],
        )
        print(f"  stored {min(i + batch, len(chunks))}/{len(chunks)}")

    print(f"Done. Collection '{config.COLLECTION}' has {collection.count()} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
