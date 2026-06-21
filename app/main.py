"""FastAPI service for the STM32 datasheet RAG assistant.

Endpoints:
  GET  /health        liveness and chunk count
  POST /ask           {question} -> {answer, citations}
  POST /draft-code    {question} -> {code, citations}   (bonus)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, rag
from .llm import LLMError
from .store import get_collection

app = FastAPI(
    title="STM32 datasheet RAG",
    description="Ask questions about the STM32H7 reference manual and get cited answers.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["How is the USART baud rate configured on STM32H7?"])
    top_k: int | None = Field(default=None, ge=1, le=15)


class Citation(BaseModel):
    page: int
    chunk_id: str
    score: float
    preview: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


class CodeResponse(BaseModel):
    code: str
    citations: list[Citation]


@app.get("/health")
def health():
    try:
        count = get_collection().count()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"vector store error: {exc}")
    return {
        "status": "ok",
        "collection": config.COLLECTION,
        "chunks": count,
        "embed_model": config.EMBED_MODEL,
        "answer_model": config.ANSWER_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return rag.answer(req.question, top_k=req.top_k)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/draft-code", response_model=CodeResponse)
def draft_code(req: AskRequest):
    try:
        return rag.draft_code(req.question, top_k=req.top_k)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
