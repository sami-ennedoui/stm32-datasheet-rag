"""Chroma vector store helpers.

A local on-disk Chroma database holds the chunk vectors. No server needed.
"""
from __future__ import annotations

import chromadb

from . import config

_client = None


def get_client():
    global _client
    if _client is None:
        config.VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.VECTOR_DIR))
    return _client


def get_collection():
    # Cosine space, embeddings are supplied by us (no Chroma embedding fn).
    return get_client().get_or_create_collection(
        name=config.COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    client = get_client()
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass
    return get_collection()
