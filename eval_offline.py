"""
Highland Greenz Offline Eval — runs without a running backend.

Tests every golden row via DIRECT OpenAI calls with the brochure as context,
measuring:
  - rag_latency_ms  : time to do vector similarity search (numpy cosine sim)
  - llm_latency_ms  : time for OpenAI to generate the answer
  - total_ms        : rag + llm

This gives real latency numbers for the LLM/RAG path, independent of
FastAPI/Redis/Postgres overhead.  Compare with the live backend run to
quantify exactly how much the backend middleware adds.

Produces the same Excel schema as eval_runner.py so the two reports can be
merged for a side-by-side view.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import openai
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GOLDEN_XLSX   = r"C:\Users\ADVORA-ALWYN\OneDrive\Desktop\highland_greenz_golden_dataset.xlsx"
BROCHURE_TXT  = Path(__file__).parent / "knowledge_base.txt"  # enriched: brochure + pricing
RESULTS_DIR   = Path(__file__).parent / "results"

OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL   = "text-embedding-3-small"
LLM_MODEL     = "gpt-4o-mini"
CHUNK_SIZE    = 800      # chars per chunk
TOP_K         = 4        # chunks to retrieve
MAX_ROWS      = 20       # None = all 450

# ---------------------------------------------------------------------------
# Chunker (mirrors the backend's chunking logic)
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[dict]:
    chunks = []
    page_re = re.compile(r"\[PAGE (\d+)\]")
    current_page = 0
    lines = text.split("\n")
    buf = []
    for line in lines:
        m = page_re.match(line.strip())
        if m:
            current_page = int(m.group(1))
            continue
        buf.append(line)
        if sum(len(l) for l in buf) >= size:
            chunks.append({"page": current_page, "content": "\n".join(buf).strip()})
            buf = []
    if buf:
        chunks.append({"page": current_page, "content": "\n".join(buf).strip()})
    return [c for c in chunks if c["content"]]


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
class EmbedCache:
    """Cache embeddings so we don't re-embed the same chunks every run."""
    CACHE_FILE = Path(__file__).parent / "embed_cache.json"

    def __init__(self):
        self._cache: dict[str, list[float]] = {}
        if self.CACHE_FILE.exists():
            try:
                self._cache = json.loads(self.CACHE_FILE.read_text())
                print(f"[embed] loaded {len(self._cache)} cached embeddings")
            except Exception:
                pass

    def save(self):
        self.CACHE_FILE.write_text(json.dumps(self._cache))

    def get(self, text: str) -> list[float] | None:
        return self._cache.get(text)

    def set(self, text: str, vec: list[float]):
        self._cache[text] = vec


_cache = EmbedCache()
_oai   = openai.OpenAI(api_key=OPENAI_KEY)


def embed_batch(texts: list[str]) -> list[list[float]]:
    missing = [t for t in texts if _cache.get(t) is None]
    if missing:
        resp = _oai.embeddings.create(model=EMBED_MODEL, input=missing)
        for text, item in zip(missing, resp.data):
            _cache.set(text, item.embedding)
    return [_cache.get(t) for t in texts]


# ---------------------------------------------------------------------------
# RAG retriever
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    page: int
    content: str
    embedding: list[float] = field(default_factory=list)


def cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class LocalRAG:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def retrieve(self, query: str, k: int = TOP_K) -> tuple[list[Chunk], float]:
        """Returns (top_k_chunks, retrieval_ms)."""
        t0 = time.perf_counter()
        q_vec = embed_batch([query])[0]
        scores = [(cosine(q_vec, c.embedding), c) for c in self.chunks]
        scores.sort(key=lambda x: -x[0])
        t1 = time.perf_counter()
        return [c for _, c in scores[:k]], round((t1 - t0) * 1000, 1)


def build_rag() -> LocalRAG:
    """Parse brochure, embed chunks, build index."""
    print("[rag] building local RAG index…")
    text = BROCHURE_TXT.read_text(encoding="utf-8")
    raw  = chunk_text(text)
    contents = [c["content"] for c in raw]
    print(f"[rag] embedding {len(contents)} chunks…")
    vecs = embed_batch(contents)
    _cache.save()
    chunks = [Chunk(page=r["page"], content=r["content"], embedding=v)
              for r, v in zip(raw, vecs)]
    print(f"[rag] index ready ({len(chunks)} chunks)")
    return LocalRAG(chunks)


# ---------------------------------------------------------------------------
# Accuracy judge
# ---------------------------------------------------------------------------
def judge_sync(utterance: str, response: str, required: str,
               forbidden: str, constraints: str) -> dict:
    system = (
        "You are a strict evaluator for a real-estate voice-agent. "
        "Score the agent response 0-100 based on REQUIRED_FACTS coverage and FORBIDDEN_FACTS avoidance. "
        "Return JSON: {score:int, reason:str, facts_hit:[str], forbidden_hit:[str]}"
    )
    user = (
        f"USER UTTERANCE: {utterance}\n\n"
        f"AGENT RESPONSE: {response}\n\n"
        f"REQUIRED_FACTS: {required or 'none'}\n"
        f"FORBIDDEN_FACTS: {forbidden or 'none'}\n"
        f"ANSWER_CONSTRAINTS: {constraints or 'none'}\n\nEvaluate and return JSON."
    )
    try:
        r = _oai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=30,
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {"score": -1, "reason": str(e), "facts_hit": [], "forbidden_hit": []}


# ---------------------------------------------------------------------------
# Main answering loop (simulates backend process_turn)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful real-estate sales assistant for DSR Highland Greenz, "
    "a luxury apartment project in Bengaluru. "
    "Answer only from the provided knowledge chunks. "
    "Be concise (≤40 words). Speak in crore, not raw digits. "
    "If the answer is not in the chunks, say you'll check and escalate."
)

@dataclass
class OfflineResult:
    row_id: str
    lang: str
    difficulty: str
    utterance: str
    response_text: str = ""
    total_latency_ms: float = 0
    backend_wall_ms: float = 0   # same as total (no middleware)
    rag_ms: float = 0
    llm_ms: float = 0
    tool_ms: float = 0
    accuracy_score: float = 0
    facts_hit: str = ""
    forbidden_hit: str = ""
    constraints_ok: bool = True
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def run_one(rag: LocalRAG, row: dict) -> OfflineResult:
    res = OfflineResult(
        row_id=str(row["id"]),
        lang=str(row.get("lang", "")),
        difficulty=str(row.get("difficulty", "")),
        utterance=str(row["user_utterance"]),
    )

    # 1. RAG retrieval
    chunks, rag_ms = rag.retrieve(res.utterance)
    res.rag_ms = rag_ms

    context = "\n\n".join(
        f"[Chunk {i+1} | Page {c.page}]\n{c.content}"
        for i, c in enumerate(chunks)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Knowledge chunks:\n{context}\n\n"
            f"Customer question: {res.utterance}"
        )},
    ]

    # 2. LLM generation
    t0 = time.perf_counter()
    try:
        r = _oai.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=150,
            timeout=60,
        )
        res.response_text = r.choices[0].message.content.strip()
    except Exception as e:
        res.error = str(e)
        return res
    t1 = time.perf_counter()

    res.llm_ms  = round((t1 - t0) * 1000, 1)
    res.total_latency_ms  = round(res.rag_ms + res.llm_ms, 1)
    res.backend_wall_ms   = res.total_latency_ms  # no middleware overhead

    # 3. Accuracy judge
    verdict = judge_sync(
        res.utterance, res.response_text,
        str(row.get("required_facts") or ""),
        str(row.get("forbidden_facts") or ""),
        str(row.get("answer_constraints") or ""),
    )
    res.accuracy_score = verdict.get("score", 0) or 0
    res.facts_hit      = "; ".join(verdict.get("facts_hit") or [])
    res.forbidden_hit  = "; ".join(verdict.get("forbidden_hit") or [])

    print(
        f"  {res.row_id:6s}  rag={res.rag_ms:5.0f}ms  "
        f"llm={res.llm_ms:5.0f}ms  "
        f"acc={res.accuracy_score:3.0f}  "
        f"{res.response_text[:60]!r}"
    )
    return res


# ---------------------------------------------------------------------------
# Excel writer (same schema as eval_runner.py)
# ---------------------------------------------------------------------------
COLS = [
    "row_id", "lang", "difficulty", "utterance",
    "response_text",
    "total_latency_ms", "backend_wall_ms", "rag_ms", "llm_ms", "tool_ms",
    "accuracy_score", "facts_hit", "forbidden_hit", "constraints_ok",
    "error", "timestamp",
]

def _style(ws, df: pd.DataFrame, hdr_color="1F4E79"):
    thin = Side(style="thin", color="D0D0D0")
    brd  = Border(left=thin, right=thin, top=thin, bottom=thin)
    for ci, col in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=ci, value=col)
        cell.font = Font(bold=True, color="FFFFFF", size=10)
        cell.fill = PatternFill("solid", fgColor=hdr_color)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = brd

    for ri, row_data in enumerate(df.itertuples(index=False), 2):
        for ci, val in enumerate(row_data, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.border = brd
            cell.font   = Font(size=9)
            col = df.columns[ci - 1]
            if col == "accuracy_score" and isinstance(val, (int, float)):
                cell.fill = PatternFill("solid", fgColor=(
                    "C6EFCE" if val >= 80 else "FFEB9C" if val >= 50 else "FFC7CE"))
            if col == "total_latency_ms" and isinstance(val, (int, float)):
                cell.fill = PatternFill("solid", fgColor=(
                    "FFC7CE" if val > 5000 else "FFEB9C" if val > 2000 else "C6EFCE"))
            if col == "error" and val:
                cell.fill = PatternFill("solid", fgColor="FFC7CE")

    widths = {"row_id":8,"lang":6,"difficulty":9,"utterance":40,"response_text":50,
              "total_latency_ms":14,"backend_wall_ms":15,"rag_ms":10,"llm_ms":10,
              "tool_ms":10,"accuracy_score":12,"facts_hit":35,"forbidden_hit":30,
              "error":30,"timestamp":20}
    for ci, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(ci)].width = widths.get(col, 14)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def write_excel(results: list[OfflineResult], out: Path):
    rows = [
        {c: getattr(r, c, "") for c in COLS}
        for r in results
    ]
    df = pd.DataFrame(rows, columns=COLS)

    # Summary
    lat  = df["total_latency_ms"].dropna()
    rag  = df["rag_ms"].dropna()
    llm  = df["llm_ms"].dropna()
    acc  = df["accuracy_score"].dropna()

    sum_rows = []
    for lang, grp in [("ALL", df)] + list(df.groupby("lang")):
        g_lat = grp["total_latency_ms"].dropna()
        if g_lat.empty:
            continue
        sum_rows.append({
            "lang": lang,
            "n":          len(grp),
            "mean_ms":    round(g_lat.mean(), 1),
            "median_ms":  round(g_lat.median(), 1),
            "p75_ms":     round(g_lat.quantile(0.75), 1),
            "p95_ms":     round(g_lat.quantile(0.95), 1),
            "mean_rag_ms":round(grp["rag_ms"].mean(), 1),
            "mean_llm_ms":round(grp["llm_ms"].mean(), 1),
            "mean_acc":   round(grp["accuracy_score"].mean(), 1),
            "errors":     int((grp["error"] != "").sum()),
        })
    sum_df = pd.DataFrame(sum_rows)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    ws_sum = wb.create_sheet("Summary")
    for ri, row_d in enumerate([sum_df.columns.tolist()] + sum_df.values.tolist(), 1):
        for ci, v in enumerate(row_d, 1):
            ws_sum.cell(row=ri, column=ci, value=v)
    _style(ws_sum, sum_df, "375623")

    ws_main = wb.create_sheet("Offline_Results")
    for ri, row_d in enumerate([df.columns.tolist()] + df.values.tolist(), 1):
        for ci, v in enumerate(row_d, 1):
            ws_main.cell(row=ri, column=ci, value=v)
    _style(ws_main, df, "1F4E79")

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"\n[report] → {out}")
    print(f"\n  mean total={lat.mean():.0f}ms  (rag={rag.mean():.0f}ms  llm={llm.mean():.0f}ms)")
    print(f"  mean accuracy = {acc.mean():.1f}/100")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if not OPENAI_KEY:
        print("Set OPENAI_API_KEY env var before running.")
        return

    golden = pd.read_excel(GOLDEN_XLSX, sheet_name="Golden_SingleTurn")
    for col in ["required_facts", "forbidden_facts", "answer_constraints",
                "difficulty", "source_of_truth"]:
        golden[col] = golden[col].fillna("").astype(str)
    if MAX_ROWS:
        golden = golden.head(MAX_ROWS)

    print(f"[golden] {len(golden)} rows loaded")

    rag = build_rag()

    print(f"\n{'='*60}")
    print(f"OFFLINE EVAL  ({len(golden)} rows, model={LLM_MODEL})")
    print(f"{'='*60}")

    results = []
    for i, (_, row) in enumerate(golden.iterrows(), 1):
        print(f"\n[{i}/{len(golden)}] {row['id']}  {str(row['user_utterance'])[:70]!r}")
        res = run_one(rag, row)
        results.append(res)

    out = RESULTS_DIR / "offline_latency_accuracy_report.xlsx"
    write_excel(results, out)


if __name__ == "__main__":
    main()
