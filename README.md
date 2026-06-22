# Assistant RAG sur datasheets STM32

A retrieval augmented generation assistant over STM32H7 documentation. You ask a
technical question about an STM32H7 peripheral. The service retrieves the most
relevant passages from the real STM32H7 reference manual (RM0433) and asks an
LLM to answer using only those passages. It returns the answer with citations
(page number and chunk id), so every claim points back to the manual.

This is built for the N7 Racing Team, who run STM32H7 boards
(repo N7-Racing-Team/Pole_data). It extends an earlier tool that generates C
register headers from datasheet memory maps.

Live demo: https://sami-ennedoui-stm32-datasheet-rag.hf.space
The home page is a small interface and the interactive API is on /docs.

The demo shares a free Hugging Face inference quota that can run out, especially
the agent, which makes several model calls per request. The page has an optional
token field: paste your own Hugging Face token to use your own quota. It is used
only for that request and never stored. The `/ask` and `/agent/header` endpoints
also accept an optional `hf_token` field for the same reason.

## What it does

- Ingests a real public PDF: STM32H7 reference manual RM0433 (about 3300 pages).
- Splits it into overlapping chunks, one page per chunk so citations are exact.
- Embeds the chunks with a local sentence-transformers model (no API key needed
  for the embedding step).
- Stores the vectors in a local Chroma database on disk.
- On a question: embeds the question, retrieves the top matching chunks, and
  asks a Hugging Face hosted LLM to answer using only that context.
- Returns the answer plus the cited sources.

## Pipeline

```
PDF (RM0433) -> page text -> chunks -> local embeddings -> Chroma (on disk)

question -> embed -> retrieve top k chunks -> LLM (HF inference) -> answer + citations
```

## Stack

- PyMuPDF for PDF text extraction
- sentence-transformers (all-MiniLM-L6-v2) for local embeddings
- Chroma for the on-disk vector store
- Hugging Face inference API (Qwen2.5 family) for answer generation
- FastAPI plus uvicorn for the service

## Why Python 3.12

The system Python here is 3.14. torch and sentence-transformers do not ship
wheels for 3.14 yet. The project runs on Python 3.12, both in the local venv and
in the Docker image (python:3.12-slim).

## Run it locally

You need a Hugging Face token for the answer step. The embedding step is local
and needs no token. Get a free token at
https://huggingface.co/settings/tokens.

```bash
# 1. Create the environment (uv shown; plain venv works too)
uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Set your token
cp .env.example .env
# edit .env and put your token in HF_TOKEN

# 3. Download the real datasheet (RM0433)
bash scripts/download_pdf.sh

# 4. Build the vector store (downloads the embedding model on first run)
python -m app.ingest

# 5. Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then ask a question:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How is the USART baud rate configured on STM32H7?"}' | python3 -m json.tool
```

Or use the helper script:

```bash
scripts/ask.sh "How is the USART baud rate configured on STM32H7?"
```

## Endpoints

- `GET /health` returns status and how many chunks are stored.
- `POST /ask` with `{"question": "..."}` returns `{"answer", "citations"}`.
- `POST /draft-code` with `{"question": "..."}` returns `{"code", "citations"}`.
  This bonus endpoint reuses the retrieved context and a Qwen coder model to
  draft a small C register configuration snippet.
- `POST /agent/header` with `{"peripheral": "USART1"}` returns
  `{"header", "citations", "agent_answer"}`. This runs the smolagents agent
  described below.

A sample real answer and its citations are saved under `examples/`.

## Agent: draft a cited C register header

This project absorbs an earlier tool that generated C register headers from a
datasheet memory map. Here it becomes a real tool using agent that grounds its
work in the actual reference manual.

Given a peripheral name, a smolagents `CodeAgent` runs a short loop with two
tools:

- `search_datasheet(query)` reuses the RAG retrieval to pull the most relevant
  RM0433 passages, each tagged with its page number.
- `build_register_header(peripheral, base_address, registers)` is deterministic.
  It validates every name as a C identifier, rejects conflicting duplicates, and
  computes each absolute address as base plus offset.

The model reads the pages and proposes register names and offsets relative to
the base. It never computes an absolute address: that arithmetic stays in
`app/regtools.py`, in plain Python, because a hallucinated address can brick
hardware. The header it returns cites the datasheet pages the agent consulted.

```bash
python -m app.agent USART1
```

The result is a draft. Always confirm the offsets against the official ST
document before flashing hardware. The model can misread a value in a dense
table, but the addresses are always computed and validated, never invented.

## Run with Docker

```bash
docker build -t stm32-rag .
# Build the vector store on a mounted volume, then run.
docker run --rm -e HF_TOKEN=your_token \
  -v "$PWD/data:/app/data" -v "$PWD/vectorstore:/app/vectorstore" \
  stm32-rag sh -c "bash scripts/download_pdf.sh && python -m app.ingest"

docker run --rm -p 8000:8000 -e HF_TOKEN=your_token \
  -v "$PWD/data:/app/data" -v "$PWD/vectorstore:/app/vectorstore" \
  stm32-rag
```

## Deploy to the cloud (Render)

The repo ships a `render.yaml`. To deploy:

1. Create a free account at https://render.com and connect a GitHub account.
2. New, then Blueprint, then pick this repo. Render reads `render.yaml`.
3. In the service settings add the secret `HF_TOKEN`.
4. Deploy. The first boot downloads the PDF and builds the vector store on the
   attached disk, so it is slow the first time. Later boots are fast.
5. Test: `curl https://YOUR-SERVICE.onrender.com/health`.

Notes:
- torch needs more memory than the free tier. The config uses the standard plan.
- The vector store lives on a 5 GB persistent disk mounted at `/var/data`.

## Notes on accuracy

The model answers only from the retrieved pages and is told to say so when the
pages do not contain the answer. Citations let you check every claim against the
real manual. This is a demo, not a certified reference. Always confirm register
details against the official ST document before flashing hardware.
