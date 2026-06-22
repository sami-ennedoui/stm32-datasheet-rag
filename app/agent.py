"""Agentic register-header drafter.

A smolagents CodeAgent that, given a peripheral name, searches the STM32H7
reference manual (RM0433) through the project's RAG retrieval, reads the relevant
pages to collect the base address and register offsets, then calls a
deterministic tool to validate the data and render the C header.

The model orchestrates and reads. It never computes an absolute address: the
build_register_header tool does the base + offset arithmetic and the validation
in plain Python. This is the merge of the RAG project and the older
datasheet-to-header tool, and it makes the agent genuinely tool using.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

from . import config, rag
from .regtools import build_header


@dataclass
class AgentSession:
    """Per-run state shared with the tools.

    A fresh session per run keeps the FastAPI endpoint safe under concurrency.
    """

    citations: dict[int, dict] = field(default_factory=dict)
    header: str | None = None

    def record(self, page: int, chunk_id: str, score: float, preview: str) -> None:
        prev = self.citations.get(page)
        if prev is None or score > prev["score"]:
            self.citations[page] = {
                "page": page,
                "chunk_id": chunk_id,
                "score": score,
                "preview": preview,
            }

    def sorted_citations(self) -> list[dict]:
        return sorted(self.citations.values(), key=lambda c: -c["score"])


def _import_tool_base():
    from smolagents import Tool

    return Tool


_Tool = _import_tool_base()


class SearchDatasheetTool(_Tool):
    name = "search_datasheet"
    description = (
        "Search the STM32H7 reference manual (RM0433) and return the most relevant "
        "passages, each tagged with its page number. Use it to find a peripheral's "
        "base address and the names and offsets of its registers. You may call it "
        "several times with different queries to gather the full register map."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "What to look up, for example "
                "'USART register map base address and register offsets'."
            ),
        }
    }
    output_type = "string"

    def __init__(self, session: AgentSession, top_k: int = 8):
        super().__init__()
        self.session = session
        self.top_k = top_k

    def forward(self, query: str) -> str:
        chunks = rag.retrieve(query, top_k=self.top_k)
        if not chunks:
            return "No passages found. Try a different query."
        blocks = []
        for c in chunks:
            preview = c.text[:160].replace("\n", " ").strip()
            self.session.record(c.page, c.chunk_id, c.score, preview)
            blocks.append(f"[page {c.page}]\n{c.text}")
        return "\n\n---\n\n".join(blocks)


class BuildRegisterHeaderTool(_Tool):
    name = "build_register_header"
    description = (
        "Validate register data and render the final C header. Pass the peripheral "
        "name, its base address as a hex string, and a list of registers, each a "
        "dict {'name': str, 'offset': hex string relative to the base}. This tool "
        "computes every absolute address itself as base + offset, so never compute "
        "an address yourself. It raises an error if names or offsets conflict; if "
        "so, fix the data and call it again. Returns the C header text."
    )
    inputs = {
        "peripheral": {
            "type": "string",
            "description": "Peripheral name, for example USART1.",
        },
        "base_address": {
            "type": "string",
            "description": "Peripheral base address as hex, for example 0x40011000.",
        },
        "registers": {
            "type": "array",
            "description": (
                "List of registers, each {'name': str, 'offset': hex string} where "
                "offset is relative to the base address."
            ),
        },
    }
    output_type = "string"

    def __init__(self, session: AgentSession):
        super().__init__()
        self.session = session

    def forward(self, peripheral: str, base_address: str, registers: list) -> str:
        header = build_header(peripheral, base_address, registers)
        self.session.header = header
        return header


_TASK = (
    "Generate a C register header for the {peripheral} peripheral of the STM32H7 "
    "microcontroller, using only the RM0433 reference manual.\n\n"
    "Follow these steps:\n"
    "1. Call search_datasheet to find the {peripheral} register boundary or register "
    "map table: its base address and each register's name and address offset. Search "
    "again with different queries if the first passages are not enough.\n"
    "2. From the passages, collect the base address and a list of registers, each "
    "with its offset relative to the base. Keep register names as written in the "
    "manual, for example {peripheral}_ISR or USART_BRR.\n"
    "3. Call build_register_header(peripheral, base_address, registers). It computes "
    "the absolute addresses and validates the data. Do not compute any absolute "
    "address yourself.\n"
    "4. If build_register_header reports a conflict or error, fix the offsets or "
    "names and call it again.\n"
    "5. Return the exact C header text that build_register_header produced."
)


def draft_header(
    peripheral: str,
    model_id: str | None = None,
    max_steps: int = 8,
    top_k: int = 8,
    token: str | None = None,
) -> dict:
    """Run the agent and return the header plus the datasheet pages it used."""
    tok = (token or "").strip() or config.HF_TOKEN
    if not tok:
        raise RuntimeError(
            "HF_TOKEN is not set. Put it in a .env file, or pass your own token. See README."
        )

    from smolagents import CodeAgent, InferenceClientModel

    session = AgentSession()
    model = InferenceClientModel(model_id=model_id or config.CODE_MODEL, token=tok)
    agent = CodeAgent(
        tools=[SearchDatasheetTool(session, top_k=top_k), BuildRegisterHeaderTool(session)],
        model=model,
        add_base_tools=False,
        max_steps=max_steps,
        additional_authorized_imports=["re"],
    )
    answer = agent.run(_TASK.format(peripheral=peripheral))
    header = session.header or str(answer)
    return {
        "peripheral": peripheral,
        "header": header,
        "citations": session.sorted_citations(),
        "agent_answer": str(answer),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stm32-rag-agent",
        description="Draft a cited C register header for an STM32H7 peripheral.",
    )
    parser.add_argument("peripheral", help="peripheral name, e.g. USART1")
    parser.add_argument("--model", help="Hugging Face model id (default: coder model)")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args(argv)

    result = draft_header(args.peripheral, model_id=args.model, max_steps=args.max_steps)
    print(result["header"])
    pages = ", ".join(str(c["page"]) for c in result["citations"])
    print(f"\n/* Sources RM0433, pages: {pages} */", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
