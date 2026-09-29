"""
Highland Greenz Voice Agent — Latency & Accuracy Evaluation Harness

Tests every row in Golden_SingleTurn through two paths:
  1. BACKEND  — direct httpx calls to http://localhost:8000
  2. UI       — Playwright browser automation against http://localhost:3000

For each row it measures:
  - total_latency_ms   : wall time of the complete HTTP round-trip (our timer)
  - backend_wall_ms    : wall_ms reported inside debug.timing by the engine itself
  - rag_ms, llm_ms     : sub-timings extracted from debug.timing
  - accuracy_score     : 0-100 from an LLM judge comparing response vs required_facts
  - facts_hit          : which required facts were found in the response
  - forbidden_hit      : which forbidden facts leaked into the response

Outputs:
  eval_harness/results/latency_accuracy_report.xlsx
    Sheet "Backend_Results"  — one row per golden row
    Sheet "UI_Results"       — same columns, measured through the browser
    Sheet "Summary"          — mean/p50/p95/p99 latencies + accuracy by lang/difficulty
"""

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_BASE   = "http://localhost:8000"
UI_BASE        = "http://localhost:3000"
GOLDEN_XLSX    = r"C:\Users\ADVORA-ALWYN\OneDrive\Desktop\highland_greenz_golden_dataset.xlsx"
BROCHURE_TXT   = Path(__file__).parent / "knowledge_base.txt"   # enriched: brochure + pricing
RESULTS_DIR    = Path(__file__).parent / "results"

# Credentials — adjust if seeded differently
EMAIL    = "admin@advora.ai"
PASSWORD = "Admin@123"

# How many golden rows to run (None = all 450).
# Start with 20 for a quick sanity check; bump to None for the full dataset.
MAX_ROWS = 20

# OpenAI key for LLM judge (reads from env or the backend .env)
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

JUDGE_MODEL   = "gpt-4o-mini"
JUDGE_TIMEOUT = 30   # seconds per judgement call

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalRow:
    row_id: str
    lang: str
    channel: str
    difficulty: str
    utterance: str
    expected_intent: str
    required_facts: str
    forbidden_facts: str
    answer_constraints: str
    source_of_truth: str
    memory_state: str

@dataclass
class TurnResult:
    row_id: str
    path: str                    # "backend" | "ui"
    utterance: str
    response_text: str = ""
    total_latency_ms: float = 0
    backend_wall_ms: float = 0
    rag_ms: float = 0
    llm_ms: float = 0
    tool_ms: float = 0
    accuracy_score: float = 0    # 0-100
    facts_hit: str = ""
    forbidden_hit: str = ""
    constraints_ok: bool = True
    intent_matched: bool = False
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

async def get_token(client: httpx.AsyncClient) -> str:
    r = await client.post(f"{BACKEND_BASE}/auth/login",
                          json={"email": EMAIL, "password": PASSWORD})
    r.raise_for_status()
    return r.json()["access_token"]

# ---------------------------------------------------------------------------
# Knowledge base: upload brochure once per run
# ---------------------------------------------------------------------------

async def ensure_knowledge(client: httpx.AsyncClient, token: str) -> str | None:
    """Upload brochure text if not already present; return document id."""
    hdrs = {"Authorization": f"Bearer {token}"}

    # Check if already uploaded
    docs = (await client.get(f"{BACKEND_BASE}/knowledge/documents", headers=hdrs)).json()
    for d in docs:
        if "highland" in d["title"].lower() or "brochure" in d["title"].lower():
            print(f"[knowledge] using existing doc '{d['title']}' ({d['id']})")
            return d["id"]

    if not BROCHURE_TXT.exists():
        print("[knowledge] brochure_text.txt not found — run extract_pdf.py first")
        return None

    text = BROCHURE_TXT.read_text(encoding="utf-8")
    print(f"[knowledge] uploading brochure ({len(text):,} chars)…")
    r = await client.post(
        f"{BACKEND_BASE}/knowledge/documents",
        headers=hdrs,
        json={
            "title": "DSR Highland Greenz Brochure",
            "doc_type": "brochure",
            "content": text,
            "ingest_now": True,
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    print(f"[knowledge] uploaded → {data.get('id')} ({data.get('chunks_created', '?')} chunks, status={data.get('status')})")
    return data.get("id")

# ---------------------------------------------------------------------------
# Accuracy judge (LLM)
# ---------------------------------------------------------------------------

def _judge_sync(utterance: str, response: str, required_facts: str,
                forbidden_facts: str, answer_constraints: str) -> dict:
    """Call OpenAI synchronously (run in thread)."""
    import openai

    if not OPENAI_KEY:
        return {"score": -1, "reason": "no openai key", "facts_hit": [], "forbidden_hit": []}

    client = openai.OpenAI(api_key=OPENAI_KEY)

    system_prompt = (
        "You are a strict evaluator for a real-estate voice-agent. "
        "Score the agent's response on a 0-100 scale based on how well it satisfies "
        "the REQUIRED_FACTS and avoids FORBIDDEN_FACTS. "
        "Return JSON: {score: int, reason: str, facts_hit: [str], forbidden_hit: [str]}\n"
        "facts_hit: list required facts that APPEARED in the response (even paraphrased).\n"
        "forbidden_hit: list forbidden facts that leaked into the response."
    )

    user_prompt = (
        f"USER UTTERANCE: {utterance}\n\n"
        f"AGENT RESPONSE: {response}\n\n"
        f"REQUIRED_FACTS: {required_facts or 'none'}\n"
        f"FORBIDDEN_FACTS: {forbidden_facts or 'none'}\n"
        f"ANSWER_CONSTRAINTS: {answer_constraints or 'none'}\n\n"
        "Evaluate and return JSON."
    )

    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=JUDGE_TIMEOUT,
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {"score": -1, "reason": str(e), "facts_hit": [], "forbidden_hit": []}


async def judge_accuracy(utterance: str, response: str, required_facts: str,
                         forbidden_facts: str, answer_constraints: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _judge_sync, utterance, response, required_facts,
        forbidden_facts, answer_constraints
    )

# ---------------------------------------------------------------------------
# Backend tester
# ---------------------------------------------------------------------------

async def run_backend_turn(
    client: httpx.AsyncClient,
    token: str,
    row: EvalRow,
) -> TurnResult:
    hdrs = {"Authorization": f"Bearer {token}"}
    result = TurnResult(row_id=row.row_id, path="backend", utterance=row.utterance)

    try:
        # 1. Start session
        t0 = time.perf_counter()
        sr = await client.post(
            f"{BACKEND_BASE}/sessions",
            headers=hdrs,
            json={"channel": "chat", "direction": "inbound"},
            timeout=30,
        )
        sr.raise_for_status()
        session_id = sr.json()["session_id"]

        # 2. Send turn
        t1 = time.perf_counter()
        tr = await client.post(
            f"{BACKEND_BASE}/sessions/{session_id}/turns",
            headers=hdrs,
            json={"text": row.utterance},
            timeout=90,
        )
        t2 = time.perf_counter()
        tr.raise_for_status()
        data = tr.json()

        result.total_latency_ms = round((t2 - t1) * 1000, 1)
        result.response_text    = data.get("response_text") or ""

        # Extract debug timings
        debug  = data.get("debug") or {}
        timing = debug.get("timing") or {}
        result.backend_wall_ms = timing.get("wall_ms", 0)
        result.rag_ms          = timing.get("retrieval_ms") or timing.get("rag_ms") or 0
        result.llm_ms          = timing.get("llm_ms") or timing.get("generation_ms") or 0
        result.tool_ms         = timing.get("tool_ms") or 0

        # 3. End session (fire and forget)
        await client.post(
            f"{BACKEND_BASE}/sessions/{session_id}/end",
            headers=hdrs,
            json={"analyse": "skip"},
            timeout=10,
        )

    except Exception as e:
        result.error = str(e)
        return result

    # 4. Accuracy
    verdict = await judge_accuracy(
        row.utterance, result.response_text,
        row.required_facts, row.forbidden_facts, row.answer_constraints,
    )
    result.accuracy_score = verdict.get("score", 0) or 0
    result.facts_hit      = "; ".join(verdict.get("facts_hit") or [])
    result.forbidden_hit  = "; ".join(verdict.get("forbidden_hit") or [])

    print(
        f"  [backend] {row.row_id:6s}  {result.total_latency_ms:6.0f} ms  "
        f"wall={result.backend_wall_ms:.0f}ms  acc={result.accuracy_score}  "
        f"{'ERROR:'+result.error[:40] if result.error else ''}"
    )
    return result

# ---------------------------------------------------------------------------
# UI tester (Playwright)
# ---------------------------------------------------------------------------

async def run_ui_turn(row: EvalRow, token: str) -> TurnResult:
    """
    Drive the React console page in a headless browser and measure:
      - total_latency_ms  : from pressing Enter to the response bubble appearing
    The response text is read from the DOM after it appears.
    """
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout

    result = TurnResult(row_id=row.row_id, path="ui", utterance=row.utterance)

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ctx = await browser.new_context(
                base_url=UI_BASE,
                extra_http_headers={"Authorization": f"Bearer {token}"},
            )

            # Inject the auth token so the UI's localStorage is pre-populated
            page = await ctx.new_page()
            await page.goto(UI_BASE)
            await page.evaluate(
                """([access, refresh]) => {
                    localStorage.setItem('voice_agent_token', access);
                    localStorage.setItem('voice_agent_refresh', refresh);
                }""",
                [token, token],
            )

            # Navigate to the Console page
            await page.goto(f"{UI_BASE}/console", wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle", timeout=15000)

            # Select "Unknown caller" / anonymous session start:
            # Type a fake phone number in the customer search box to get an anonymous start
            search = page.locator('input[placeholder*="Search by name"]')
            await search.fill("+91 9999999999")
            await page.wait_for_timeout(800)

            # Click "Call from ..." button if it appears
            call_btn = page.locator('button:has-text("Call from")')
            if await call_btn.count() > 0:
                await call_btn.first.click()
            await page.wait_for_timeout(300)

            # Click "Start conversation"
            start = page.locator('button:has-text("Start conversation")')
            await start.click()
            await page.wait_for_selector("text=thinking", timeout=15000)
            # Wait for opening line to appear
            await page.wait_for_selector(".agent-bubble, text=Agent", timeout=10000)
            await page.wait_for_timeout(500)

            # Find the chat input
            chat_input = page.locator('input[placeholder*="Type what the caller"]')
            await chat_input.fill(row.utterance)

            # Measure from Enter key to response appearing
            t0 = time.perf_counter()
            await chat_input.press("Enter")

            # Wait for "thinking…" to disappear (agent is processing)
            await page.wait_for_selector("text=thinking…", timeout=5000)
            # Wait for it to disappear (response arrived)
            await page.wait_for_selector("text=thinking…", state="hidden", timeout=90000)
            t1 = time.perf_counter()

            result.total_latency_ms = round((t1 - t0) * 1000, 1)

            # Read the last agent bubble text
            bubbles = page.locator('.rounded-2xl.bg-slate-100')
            count   = await bubbles.count()
            if count:
                result.response_text = (await bubbles.nth(count - 1).inner_text()).strip()

            await browser.close()

    except Exception as e:
        result.error = str(e)
        return result

    # Accuracy (same judge)
    verdict = await judge_accuracy(
        row.utterance, result.response_text,
        row.required_facts, row.forbidden_facts, row.answer_constraints,
    )
    result.accuracy_score = verdict.get("score", 0) or 0
    result.facts_hit      = "; ".join(verdict.get("facts_hit") or [])
    result.forbidden_hit  = "; ".join(verdict.get("forbidden_hit") or [])

    print(
        f"  [ui]      {row.row_id:6s}  {result.total_latency_ms:6.0f} ms  "
        f"acc={result.accuracy_score}  "
        f"{'ERROR:'+result.error[:40] if result.error else ''}"
    )
    return result

# ---------------------------------------------------------------------------
# Excel report writer
# ---------------------------------------------------------------------------

COLS = [
    "row_id", "lang", "difficulty", "utterance",
    "response_text",
    "total_latency_ms", "backend_wall_ms", "rag_ms", "llm_ms", "tool_ms",
    "accuracy_score", "facts_hit", "forbidden_hit", "constraints_ok",
    "error", "timestamp",
]

def _results_to_df(results: list[TurnResult], golden: pd.DataFrame) -> pd.DataFrame:
    g = golden.set_index("id")[["lang", "difficulty", "user_utterance",
                                "expected_intent", "source_of_truth"]].rename(
        columns={"user_utterance": "utterance_ref"})

    rows = []
    for r in results:
        row = {
            "row_id":            r.row_id,
            "lang":              r.row_id[:2].lower(),
            "difficulty":        "",
            "utterance":         r.utterance[:120],
            "response_text":     r.response_text[:200],
            "total_latency_ms":  r.total_latency_ms,
            "backend_wall_ms":   r.backend_wall_ms,
            "rag_ms":            r.rag_ms,
            "llm_ms":            r.llm_ms,
            "tool_ms":           r.tool_ms,
            "accuracy_score":    r.accuracy_score,
            "facts_hit":         r.facts_hit,
            "forbidden_hit":     r.forbidden_hit,
            "constraints_ok":    r.constraints_ok,
            "error":             r.error,
            "timestamp":         r.timestamp,
        }
        # Enrich from golden
        if r.row_id in g.index:
            row["lang"]       = g.at[r.row_id, "lang"]
            row["difficulty"] = g.at[r.row_id, "difficulty"]
        rows.append(row)

    return pd.DataFrame(rows, columns=COLS + ["expected_intent", "source_of_truth"]
                        if "expected_intent" in pd.DataFrame(rows).columns else COLS)


def _summary_df(be_df: pd.DataFrame, ui_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path_label, df in [("Backend", be_df), ("UI", ui_df)]:
        if df.empty:
            continue
        lat = df["total_latency_ms"].dropna()
        rows.append({
            "path": path_label,
            "n_rows":       len(df),
            "mean_ms":      round(lat.mean(), 1),
            "median_ms":    round(lat.median(), 1),
            "p75_ms":       round(lat.quantile(0.75), 1),
            "p95_ms":       round(lat.quantile(0.95), 1),
            "p99_ms":       round(lat.quantile(0.99), 1),
            "mean_backend_wall_ms": round(df["backend_wall_ms"].mean(), 1),
            "mean_rag_ms":  round(df["rag_ms"].mean(), 1),
            "mean_llm_ms":  round(df["llm_ms"].mean(), 1),
            "mean_acc":     round(df["accuracy_score"].mean(), 1),
            "errors":       int((df["error"] != "").sum()),
        })

        # Per-language breakdown
        for lang, grp in df.groupby("lang"):
            lat_g = grp["total_latency_ms"].dropna()
            if lat_g.empty:
                continue
            rows.append({
                "path": f"{path_label} ({lang})",
                "n_rows":       len(grp),
                "mean_ms":      round(lat_g.mean(), 1),
                "median_ms":    round(lat_g.median(), 1),
                "p75_ms":       round(lat_g.quantile(0.75), 1),
                "p95_ms":       round(lat_g.quantile(0.95), 1),
                "p99_ms":       round(lat_g.quantile(0.99), 1),
                "mean_backend_wall_ms": round(grp["backend_wall_ms"].mean(), 1),
                "mean_rag_ms":  round(grp["rag_ms"].mean(), 1),
                "mean_llm_ms":  round(grp["llm_ms"].mean(), 1),
                "mean_acc":     round(grp["accuracy_score"].mean(), 1),
                "errors":       int((grp["error"] != "").sum()),
            })

        # Per-difficulty breakdown
        for diff, grp in df.groupby("difficulty"):
            lat_g = grp["total_latency_ms"].dropna()
            if lat_g.empty:
                continue
            rows.append({
                "path": f"{path_label} (diff={diff})",
                "n_rows":       len(grp),
                "mean_ms":      round(lat_g.mean(), 1),
                "median_ms":    round(lat_g.median(), 1),
                "p75_ms":       round(lat_g.quantile(0.75), 1),
                "p95_ms":       round(lat_g.quantile(0.95), 1),
                "p99_ms":       round(lat_g.quantile(0.99), 1),
                "mean_backend_wall_ms": round(grp["backend_wall_ms"].mean(), 1),
                "mean_rag_ms":  round(grp["rag_ms"].mean(), 1),
                "mean_llm_ms":  round(grp["llm_ms"].mean(), 1),
                "mean_acc":     round(grp["accuracy_score"].mean(), 1),
                "errors":       int((grp["error"] != "").sum()),
            })

    return pd.DataFrame(rows)


def _style_sheet(ws, df: pd.DataFrame, header_color: str = "1F4E79"):
    """Apply professional formatting to a worksheet."""
    thin = Side(style="thin", color="D0D0D0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Header row
    for col_idx, col_name in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font      = Font(bold=True, color="FFFFFF", size=10)
        cell.fill      = PatternFill("solid", fgColor=header_color)
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border    = border

    # Data rows
    for r_idx, row_data in enumerate(df.itertuples(index=False), 2):
        for c_idx, val in enumerate(row_data, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            cell.border    = border
            cell.alignment = Alignment(vertical="top", wrap_text=False)
            cell.font      = Font(size=9)

            col_name = df.columns[c_idx - 1]
            # Colour accuracy scores
            if col_name == "accuracy_score" and isinstance(val, (int, float)):
                if val >= 80:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                elif val >= 50:
                    cell.fill = PatternFill("solid", fgColor="FFEB9C")
                elif val >= 0:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
            # Colour latency
            if col_name == "total_latency_ms" and isinstance(val, (int, float)):
                if val > 5000:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
                elif val > 2000:
                    cell.fill = PatternFill("solid", fgColor="FFEB9C")
                else:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
            # Colour errors
            if col_name == "error" and val:
                cell.fill = PatternFill("solid", fgColor="FFC7CE")

    # Column widths
    width_map = {
        "row_id": 8, "lang": 6, "difficulty": 9, "utterance": 40,
        "response_text": 50, "total_latency_ms": 14, "backend_wall_ms": 15,
        "rag_ms": 10, "llm_ms": 10, "tool_ms": 10, "accuracy_score": 12,
        "facts_hit": 35, "forbidden_hit": 30, "error": 30, "timestamp": 20,
    }
    for col_idx, col_name in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width_map.get(col_name, 14)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def write_excel(be_results: list[TurnResult], ui_results: list[TurnResult],
                golden: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    be_df  = _results_to_df(be_results, golden)
    ui_df  = _results_to_df(ui_results, golden)
    sum_df = _summary_df(be_df, ui_df)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Summary sheet (first)
    ws_sum = wb.create_sheet("Summary")
    for r_idx, row in enumerate([sum_df.columns.tolist()] + sum_df.values.tolist(), 1):
        for c_idx, val in enumerate(row, 1):
            ws_sum.cell(row=r_idx, column=c_idx, value=val)
    _style_sheet(ws_sum, sum_df, header_color="375623")

    # Backend sheet
    if be_df is not None and not be_df.empty:
        ws_be = wb.create_sheet("Backend_Results")
        for r_idx, row in enumerate([be_df.columns.tolist()] + be_df.values.tolist(), 1):
            for c_idx, val in enumerate(row, 1):
                ws_be.cell(row=r_idx, column=c_idx, value=val)
        _style_sheet(ws_be, be_df, header_color="1F4E79")

    # UI sheet
    if ui_df is not None and not ui_df.empty:
        ws_ui = wb.create_sheet("UI_Results")
        for r_idx, row in enumerate([ui_df.columns.tolist()] + ui_df.values.tolist(), 1):
            for c_idx, val in enumerate(row, 1):
                ws_ui.cell(row=r_idx, column=c_idx, value=val)
        _style_sheet(ws_ui, ui_df, header_color="833C00")

    wb.save(out_path)
    print(f"\n[report] saved → {out_path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load golden dataset
    print("[golden] loading dataset…")
    golden = pd.read_excel(GOLDEN_XLSX, sheet_name="Golden_SingleTurn")
    golden["memory_state"] = golden["memory_state"].fillna("{}").astype(str)
    golden["required_facts"] = golden["required_facts"].fillna("").astype(str)
    golden["forbidden_facts"] = golden["forbidden_facts"].fillna("").astype(str)
    golden["answer_constraints"] = golden["answer_constraints"].fillna("").astype(str)
    golden["difficulty"] = golden["difficulty"].fillna("").astype(str)
    golden["source_of_truth"] = golden["source_of_truth"].fillna("").astype(str)
    if MAX_ROWS:
        golden = golden.head(MAX_ROWS)

    rows: list[EvalRow] = [
        EvalRow(
            row_id=str(r["id"]),
            lang=str(r["lang"]),
            channel=str(r.get("channel", "inbound")),
            difficulty=str(r.get("difficulty", "")),
            utterance=str(r["user_utterance"]),
            expected_intent=str(r.get("expected_intent", "")),
            required_facts=str(r.get("required_facts", "")),
            forbidden_facts=str(r.get("forbidden_facts", "")),
            answer_constraints=str(r.get("answer_constraints", "")),
            source_of_truth=str(r.get("source_of_truth", "")),
            memory_state=str(r.get("memory_state", "{}")),
        )
        for _, r in golden.iterrows()
    ]
    print(f"[golden] loaded {len(rows)} rows")

    # Check backend
    print(f"\n[backend] connecting to {BACKEND_BASE}…")
    async with httpx.AsyncClient(timeout=10) as probe:
        try:
            h = await probe.get(f"{BACKEND_BASE}/health")
            print(f"[backend] health → {h.status_code} {h.text[:80]}")
        except Exception as e:
            print(f"[backend] UNREACHABLE: {e}")
            print("  Start the backend first:  cd core && uvicorn app.main:app --reload")
            print("  Then re-run this script.\n")

    # Auth
    print("[auth] logging in…")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            token = await get_token(client)
            print("[auth] token acquired")
        except Exception as e:
            print(f"[auth] FAILED: {e}")
            print(f"  Check credentials: {EMAIL} / {PASSWORD}")
            print("  Or seed with:  python scripts/seed.py --tenant Advora --email admin@advora.ai --password Admin@123")
            return

        # Ensure knowledge base
        print("\n[knowledge] checking knowledge base…")
        await ensure_knowledge(client, token)

    # ── Backend tests ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"BACKEND TESTS  ({len(rows)} rows)")
    print(f"{'='*60}")

    be_results: list[TurnResult] = []
    async with httpx.AsyncClient(timeout=120) as client:
        token = await get_token(client)  # fresh token per phase
        for i, row in enumerate(rows, 1):
            print(f"  [{i}/{len(rows)}] {row.row_id}  {row.utterance[:60]!r}")
            result = await run_backend_turn(client, token, row)
            be_results.append(result)

    # ── UI tests ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"UI TESTS  ({len(rows)} rows)")
    print(f"{'='*60}")

    # Check if UI is up
    async with httpx.AsyncClient(timeout=10) as probe:
        try:
            h = await probe.get(UI_BASE)
            print(f"[ui] health → {h.status_code}")
            ui_available = True
        except Exception as e:
            print(f"[ui] UNREACHABLE: {e}")
            print("  Start the UI first:  cd AI-Voice-Agent-UI-main && npm run dev")
            ui_available = False

    ui_results: list[TurnResult] = []
    if ui_available:
        async with httpx.AsyncClient(timeout=30) as client:
            token = await get_token(client)
        for i, row in enumerate(rows, 1):
            print(f"  [{i}/{len(rows)}] {row.row_id}  {row.utterance[:60]!r}")
            result = await run_ui_turn(row, token)
            ui_results.append(result)
    else:
        print("[ui] skipping UI tests — start the UI server and re-run")

    # ── Report ───────────────────────────────────────────────────────────
    out = RESULTS_DIR / "latency_accuracy_report.xlsx"
    print(f"\n[report] writing Excel report…")
    write_excel(be_results, ui_results, golden, out)

    # Print quick summary to console
    if be_results:
        lats = [r.total_latency_ms for r in be_results if not r.error]
        accs = [r.accuracy_score   for r in be_results if not r.error]
        if lats:
            print(f"\nBACKEND  mean={sum(lats)/len(lats):.0f}ms  "
                  f"max={max(lats):.0f}ms  "
                  f"acc_mean={sum(accs)/len(accs):.1f}")

    if ui_results:
        lats = [r.total_latency_ms for r in ui_results if not r.error]
        accs = [r.accuracy_score   for r in ui_results if not r.error]
        if lats:
            print(f"UI       mean={sum(lats)/len(lats):.0f}ms  "
                  f"max={max(lats):.0f}ms  "
                  f"acc_mean={sum(accs)/len(accs):.1f}")

    print(f"\nDone. Open: {out}\n")


if __name__ == "__main__":
    asyncio.run(main())
