"""Retrieval augmented generation pipeline.

Given a question:
  1. embed it with the same local model used at ingest time,
  2. retrieve the top matching chunks from Chroma,
  3. ask the LLM to answer using only those chunks,
  4. return the answer plus the cited sources (page and chunk id).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from . import config, llm
from .store import get_collection
from .embed import Embedder

# The embedder is heavy, so load it once and reuse it.
_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


@dataclass
class Citation:
    page: int
    chunk_id: str
    score: float
    preview: str


@dataclass
class Retrieved:
    text: str
    page: int
    chunk_id: str
    score: float


def retrieve(question: str, top_k: int | None = None) -> list[Retrieved]:
    top_k = top_k or config.TOP_K
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError(
            "Vector store is empty. Run 'python -m app.ingest' first."
        )
    query_vec = _get_embedder().encode_one(question)
    result = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            Retrieved(
                text=doc,
                page=int(meta["page"]),
                chunk_id=str(meta["chunk_id"]),
                # cosine distance -> similarity, rounded for display
                score=round(1.0 - float(dist), 4),
            )
        )
    return out


_SYSTEM = (
    "You are a precise assistant for STM32 microcontrollers. "
    "Answer ONLY using the provided context from the STM32H7 reference manual. "
    "If the context does not contain the answer, say you cannot find it in the "
    "provided pages. Cite the page numbers you used, like [page 1944]. "
    "Be concise and technical."
)


def _build_prompt(question: str, chunks: list[Retrieved]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[page {c.page}, chunk {c.chunk_id}]\n{c.text}")
    context = "\n\n---\n\n".join(blocks)
    return (
        f"Context from the STM32H7 reference manual:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and cite page numbers."
    )


def answer(question: str, top_k: int | None = None) -> dict:
    chunks = retrieve(question, top_k=top_k)
    prompt = _build_prompt(question, chunks)
    text = llm.chat(_SYSTEM, prompt, model=config.ANSWER_MODEL)
    citations = [
        Citation(
            page=c.page,
            chunk_id=c.chunk_id,
            score=c.score,
            preview=c.text[:160].replace("\n", " ").strip(),
        )
        for c in chunks
    ]
    return {
        "answer": text,
        "citations": [asdict(c) for c in citations],
    }


_CODE_SYSTEM = (
    "You are an embedded C engineer working on STM32H7 microcontrollers. "
    "Using ONLY the register details in the provided context, write a short, "
    "well commented C snippet that configures the requested peripheral. "
    "Use direct register access with the standard STM32H7 register names "
    "(for example USART1->BRR). Do not invent register names that are not in "
    "the context. After the code, list the page numbers you relied on."
)


def draft_code(question: str, top_k: int | None = None) -> dict:
    """Bonus endpoint: draft a small C register configuration snippet.

    Reuses the retrieved datasheet context and the Qwen coder model, the same
    approach as the ~/smolagent project.
    """
    chunks = retrieve(question, top_k=top_k)
    prompt = _build_prompt(question, chunks)
    text = llm.chat(
        _CODE_SYSTEM, prompt, model=config.CODE_MODEL, max_tokens=900
    )
    citations = [
        Citation(
            page=c.page,
            chunk_id=c.chunk_id,
            score=c.score,
            preview=c.text[:160].replace("\n", " ").strip(),
        )
        for c in chunks
    ]
    return {
        "code": text,
        "citations": [asdict(c) for c in citations],
    }
