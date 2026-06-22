"""Hugging Face inference API client.

Same token and same Qwen model family as the existing ~/smolagent project.
The token is read from the environment (HF_TOKEN), never hardcoded.
"""
from __future__ import annotations

from . import config


class LLMError(RuntimeError):
    pass


def _resolve_token(token: str | None) -> str:
    """Pick the token to use: a caller supplied one, else the configured one.

    A visitor can pass their own token through the demo so a request uses their
    own inference quota. A blank value falls back to the configured token.
    """
    cleaned = (token or "").strip()
    tok = cleaned or config.HF_TOKEN
    if not tok:
        raise LLMError(
            "HF_TOKEN is not set. Put it in a .env file, or pass your own token. "
            "See README."
        )
    return tok


def _client(token: str | None = None):
    from huggingface_hub import InferenceClient

    return InferenceClient(token=_resolve_token(token))


def chat(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 700,
    temperature: float = 0.1,
    token: str | None = None,
) -> str:
    """Call the HF chat completion endpoint and return the text answer."""
    model = model or config.ANSWER_MODEL
    client = _client(token)
    try:
        response = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # network or model errors
        raise LLMError(f"Hugging Face inference failed: {exc}") from exc
    return response.choices[0].message.content.strip()
