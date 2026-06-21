"""Hugging Face inference API client.

Same token and same Qwen model family as the existing ~/smolagent project.
The token is read from the environment (HF_TOKEN), never hardcoded.
"""
from __future__ import annotations

from . import config


class LLMError(RuntimeError):
    pass


def _client():
    from huggingface_hub import InferenceClient

    if not config.HF_TOKEN:
        raise LLMError(
            "HF_TOKEN is not set. Put it in a .env file or the environment. "
            "See README."
        )
    return InferenceClient(token=config.HF_TOKEN)


def chat(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 700,
    temperature: float = 0.1,
) -> str:
    """Call the HF chat completion endpoint and return the text answer."""
    model = model or config.ANSWER_MODEL
    client = _client()
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
