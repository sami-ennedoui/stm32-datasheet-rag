"""FastAPI service for the STM32 datasheet RAG assistant.

Endpoints:
  GET  /health         liveness and chunk count
  POST /ask            {question} -> {answer, citations}
  POST /draft-code     {question} -> {code, citations}   (bonus)
  POST /agent/header   {peripheral} -> {header, citations}  (smolagents agent)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config, rag
from .llm import LLMError
from .store import get_collection

app = FastAPI(
    title="STM32 datasheet RAG",
    description="Ask questions about the STM32H7 reference manual and get cited answers.",
    version="1.0.0",
)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STM32 datasheet RAG</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 820px; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.5; }
  h1 { margin-bottom: .2rem; }
  p.lead { color: #555; margin-top: .2rem; }
  textarea { width: 100%; box-sizing: border-box; padding: .6rem; font-size: 1rem; }
  .row { display: flex; gap: 1rem; align-items: center; margin: .6rem 0; flex-wrap: wrap; }
  button { padding: .55rem 1.1rem; font-size: 1rem; cursor: pointer; }
  pre { white-space: pre-wrap; background: #00000010; padding: .8rem; border-radius: 6px; }
  table { border-collapse: collapse; width: 100%; font-size: .9rem; margin-top: .6rem; }
  th, td { border: 1px solid #8884; padding: .35rem .5rem; text-align: left; vertical-align: top; }
  .muted { color: #777; font-size: .85rem; }
  a { color: #1A5276; }
</style>
</head>
<body>
  <h1>Assistant RAG sur les datasheets STM32</h1>
  <p class="lead">Pose une question sur le manuel de reference STM32H7 (RM0433).
  Le service retrouve les passages les plus proches dans le manuel et demande a
  un modele de langage de repondre uniquement a partir de ces passages. Chaque
  reponse cite les pages d'origine.</p>
  <textarea id="q" rows="3" placeholder="How is the USART baud rate configured on STM32H7?"></textarea>
  <div class="row">
    <label><input type="radio" name="mode" value="ask" checked> Repondre</label>
    <label><input type="radio" name="mode" value="draft-code"> Generer du C</label>
    <button id="go">Demander</button>
    <span id="status" class="muted"></span>
  </div>
  <div id="answer"></div>
  <div id="cites"></div>

  <h2>Brouillon d'en-tete C de registres</h2>
  <p class="lead">Donne le nom d'un peripherique STM32H7. Un agent smolagents
  cherche sa table de registres dans le manuel, lit les offsets, puis un outil
  deterministe calcule les adresses absolues et rend l'en-tete C. Le modele ne
  calcule jamais une adresse, il lit et orchestre seulement.</p>
  <div class="row">
    <input id="periph" type="text" placeholder="USART1"
           style="flex:1; min-width:12rem; padding:.5rem; font-size:1rem;">
    <button id="goAgent">Generer l'en-tete</button>
    <span id="agentStatus" class="muted"></span>
  </div>
  <div id="agentOut"></div>
  <div id="agentCites"></div>

  <p class="muted">API interactive sur <a href="/docs">/docs</a>.
  Code source: <a href="https://github.com/sami-ennedoui/stm32-datasheet-rag">GitHub</a>.</p>
<script>
const $ = (id) => document.getElementById(id);
async function run() {
  const q = $("q").value.trim();
  if (q.length < 3) { $("status").textContent = "Pose une question plus longue."; return; }
  const mode = document.querySelector('input[name=mode]:checked').value;
  $("go").disabled = true; $("status").textContent = "Recherche et generation...";
  $("answer").innerHTML = ""; $("cites").innerHTML = "";
  try {
    const r = await fetch("/" + mode, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: q})
    });
    const data = await r.json();
    if (!r.ok) { $("status").textContent = "Erreur: " + (data.detail || r.status); return; }
    const body = mode === "draft-code" ? data.code : data.answer;
    const pre = document.createElement("pre"); pre.textContent = body;
    $("answer").appendChild(pre);
    if (data.citations && data.citations.length) {
      let html = "<table><thead><tr><th>Page</th><th>Score</th><th>Extrait</th></tr></thead><tbody>";
      for (const c of data.citations) {
        const prev = (c.preview || "").replace(/</g, "&lt;");
        html += `<tr><td>${c.page}</td><td>${c.score}</td><td>${prev}</td></tr>`;
      }
      html += "</tbody></table>";
      $("cites").innerHTML = html;
    }
    $("status").textContent = "";
  } catch (e) {
    $("status").textContent = "Erreur reseau: " + e;
  } finally {
    $("go").disabled = false;
  }
}
$("go").addEventListener("click", run);

async function runAgent() {
  const p = $("periph").value.trim();
  if (p.length < 2) { $("agentStatus").textContent = "Donne un nom de peripherique."; return; }
  $("goAgent").disabled = true;
  $("agentStatus").textContent = "L'agent cherche dans le manuel et itere, cela peut prendre une minute...";
  $("agentOut").innerHTML = ""; $("agentCites").innerHTML = "";
  try {
    const r = await fetch("/agent/header", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({peripheral: p})
    });
    const data = await r.json();
    if (!r.ok) { $("agentStatus").textContent = "Erreur: " + (data.detail || r.status); return; }
    const pre = document.createElement("pre"); pre.textContent = data.header;
    $("agentOut").appendChild(pre);
    if (data.citations && data.citations.length) {
      let html = "<table><thead><tr><th>Page</th><th>Score</th><th>Extrait</th></tr></thead><tbody>";
      for (const c of data.citations) {
        const prev = (c.preview || "").replace(/</g, "&lt;");
        html += `<tr><td>${c.page}</td><td>${c.score}</td><td>${prev}</td></tr>`;
      }
      html += "</tbody></table>";
      $("agentCites").innerHTML = html;
    }
    $("agentStatus").textContent = "Verifie toujours les offsets contre le manuel officiel avant usage.";
  } catch (e) {
    $("agentStatus").textContent = "Erreur reseau: " + e;
  } finally {
    $("goAgent").disabled = false;
  }
}
$("goAgent").addEventListener("click", runAgent);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return INDEX_HTML


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


class AgentHeaderRequest(BaseModel):
    peripheral: str = Field(..., min_length=2, max_length=40, examples=["USART1"])
    max_steps: int | None = Field(default=8, ge=1, le=12)


class AgentHeaderResponse(BaseModel):
    peripheral: str
    header: str
    citations: list[Citation]
    agent_answer: str


@app.post("/agent/header", response_model=AgentHeaderResponse)
def agent_header(req: AgentHeaderRequest):
    """Run the smolagents agent: search the datasheet, then build a cited C header.

    The agent reads RM0433 passages and proposes register names and relative
    offsets. The deterministic build step computes every absolute address and
    validates the data, so the model never emits an address itself.
    """
    from .agent import draft_header

    try:
        return draft_header(req.peripheral, max_steps=req.max_steps or 8)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # the agent loop can fail for model side reasons
        raise HTTPException(status_code=500, detail=f"agent failed: {exc}")
