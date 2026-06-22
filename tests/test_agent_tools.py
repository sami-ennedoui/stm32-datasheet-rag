"""Tests for the agent tools that do not need the network.

search_datasheet runs against the local Chroma vector store (local embeddings,
no API key), so it is deterministic enough to test. The LLM loop itself is
verified by a live run, not mocked.
"""
from app.agent import SearchDatasheetTool, BuildRegisterHeaderTool, AgentSession


def test_search_datasheet_returns_pages_and_records_citations():
    session = AgentSession()
    tool = SearchDatasheetTool(session, top_k=3)
    out = tool.forward("USART baud rate BRR register configuration")
    assert "page" in out.lower()
    assert session.citations, "expected at least one retrieved page recorded"
    sample = next(iter(session.citations.values()))
    assert isinstance(sample["page"], int)
    assert "score" in sample


def test_build_register_header_tool_renders_and_stores_header():
    session = AgentSession()
    tool = BuildRegisterHeaderTool(session)
    header = tool.forward(
        "USART1",
        "0x40011000",
        [{"name": "USART_BRR", "offset": "0x0C"}],
    )
    assert "0x4001100Cu" in header
    assert session.header == header
