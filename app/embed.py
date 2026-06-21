"""Local embedding model wrapper.

We use sentence-transformers so the demo runs offline without any API key.
The model is small (all-MiniLM-L6-v2, 384 dims) and downloads once.
"""
from __future__ import annotations

from . import config


class Embedder:
    def __init__(self, model_name: str | None = None):
        # Imported here so the rest of the app can be imported without torch.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.EMBED_MODEL
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]
