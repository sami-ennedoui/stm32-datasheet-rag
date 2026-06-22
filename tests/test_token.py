"""The demo lets a visitor supply their own HF token for a single request.

These cover the token resolution rule without any network call.
"""
import pytest

from app import llm, config


def test_resolve_token_prefers_explicit():
    assert llm._resolve_token("hf_visitor") == "hf_visitor"


def test_resolve_token_falls_back_to_config(monkeypatch):
    monkeypatch.setattr(config, "HF_TOKEN", "hf_env")
    assert llm._resolve_token(None) == "hf_env"


def test_resolve_token_ignores_blank_explicit(monkeypatch):
    monkeypatch.setattr(config, "HF_TOKEN", "hf_env")
    assert llm._resolve_token("   ") == "hf_env"


def test_resolve_token_raises_when_missing(monkeypatch):
    monkeypatch.setattr(config, "HF_TOKEN", None)
    with pytest.raises(llm.LLMError):
        llm._resolve_token(None)
