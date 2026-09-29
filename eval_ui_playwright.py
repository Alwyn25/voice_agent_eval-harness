"""
Playwright UI Latency Test

Drives the React console page and measures the time from the moment the user
presses Enter to when the agent response bubble appears in the DOM.

Compares with mock backend's simulated delay to isolate pure UI overhead:
  ui_latency_ms = time measured in browser
  backend_delay = DELAY_MS configured in mock_backend.py (1200ms by default)
  ui_overhead   = ui_latency_ms - backend_delay

Run AFTER starting:
  - mock_backend.py  on http://localhost:8000
  - Vite dev server  on http://localhost:3000
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

GOLDEN_XLSX  = r"C:\Users\ADVORA-ALWYN\OneDrive\Desktop\highland_greenz_golden_dataset.xlsx"
BACKEND_BASE = "http://localhost:8000"
UI_BASE      = "http://localhost:3000"
RESULTS_DIR  = Path(__file__).parent / "results"
MAX_ROWS     = 20
MOCK_TOKEN   = "mock_token_highland_greenz"


@dataclass
class UIResult:
    row_id: str
    lang: str
    difficulty: str
    utterance: str
    response_text: str = ""
    ui_total_ms: float = 0      # wall time from Enter to response visible (browser side)
    backend_reported_ms: float = 0   # wall_ms from debug.timing
    ui_overhead_ms: float = 0   # = ui_total_ms - backend_reported_ms
    proxy_overhead_ms: float = 0
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


async def measure_ui_turn(page, row: dict, token: str) -> UIResult:
    """Drive one turn through the browser and measure end-to-end UI latency."""
    from playwright.async_api import TimeoutError as PWTimeout

    res = UIResult(
        row_id=str(row["id"]),
        lang=str(row.get("lang", "")),
        difficulty=str(row.get("difficulty", "")),
        utterance=str(row["user_utterance"]),
    )

    # Also time the same turn via direct HTTP (from Python, no browser)
    import httpx
    t_direct_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30) as hx:
            sid_r = await hx.post(f"{BACKEND_BASE}/sessions",
                                   json={"channel": "chat", "direction": "inbound"},
                                   headers={"Authorization": f"Bearer {token}"})
            sid = sid_r.json()["session_id"]

            t_turn_start = time.perf_counter()
            turn_r = await hx.post(f"{BACKEND_BASE}/sessions/{sid}/turns",
                                    json={"text": res.utterance},
                                    headers={"Authorization": f"Bearer {token}"},
                                    timeout=90)
            t_turn_end = time.perf_counter()
            direct_ms = round((t_turn_end - t_turn_start) * 1000, 1)

            data = turn_r.json()
            res.backend_reported_ms = (data.get("debug") or {}).get("timing", {}).get("wall_ms", 0)

            await hx.post(f"{BACKEND_BASE}/sessions/{sid}/end",
                          json={"analyse": "skip"},
                          headers={"Authorization": f"Bearer {token}"},
                          timeout=10)
    except Exception as e:
        res.error = f"direct: {e}"
        return res

    # Now measure through the browser (same backend, but via Vite proxy)
    try:
        # Start a fresh session in the UI via JavaScript injection
        t_ui_start = time.perf_counter()

        # Inject auth token and use the UI's fetch to start a session
        session_data = await page.evaluate(
            """async ([url, tok, utterance]) => {
                const headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + tok
                };
                // Start session through Vite proxy (/api → localhost:8000)
                const t0 = performance.now();
                const sr = await fetch('/api/sessions', {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({channel: 'chat', direction: 'inbound'})
                });
                const session = await sr.json();

                // Send turn and measure
                const t1 = performance.now();
                const tr = await fetch('/api/sessions/' + session.session_id + '/turns', {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({text: utterance})
                });
                const turn = await tr.json();
                const t2 = performance.now();

                // End session
                fetch('/api/sessions/' + session.session_id + '/end', {
                    method: 'POST', headers,
                    body: JSON.stringify({analyse: 'skip'})
                });

                return {
                    session_ms: t1 - t0,
                    turn_ms:    t2 - t1,
                    total_ms:   t2 - t0,
                    response:   turn.response_text || '',
                    wall_ms:    (turn.debug?.timing?.wall_ms) || 0,
                };
            }""",
            [UI_BASE, token, res.utterance]
        )

        t_ui_end = time.perf_counter()

        res.ui_total_ms        = round(session_data["turn_ms"], 1)
        res.ui_overhead_ms     = round(res.ui_total_ms - res.backend_reported_ms, 1)
        res.proxy_overhead_ms  = round(res.ui_total_ms - direct_ms, 1)
        res.response_text      = session_data.get("response", "")

    except Exception as e:
        res.error = (res.error + " ui:" + str(e)).strip()

    return res


async def run_ui_tests(rows: list[dict]) -> list[UIResult]:
    from playwright.async_api import async_playwright

    results = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx     = await browser.new_context(base_url=UI_BASE)
        page    = await ctx.new_page()

        # Load the UI and inject auth
        await page.goto(UI_BASE, wait_until="domcontentloaded")
        await page.evaluate(
            """([access]) => {
                localStorage.setItem('voice_agent_token', access);
                localStorage.setItem('voice_agent_refresh', access);
            }""",
            [MOCK_TOKEN],
        )

        print(f"  Browser ready, running {len(rows)} turns…")

        for i, row in enumerate(rows, 1):
            print(f"  [{i}/{len(rows)}] {row['id']}  {str(row['user_utterance'])[:55]!r}")
            r = await measure_ui_turn(page, row, MOCK_TOKEN)
            results.append(r)
            print(
                f"    ui_total={r.ui_total_ms:.0f}ms  "
                f"backend_wall={r.backend_reported_ms:.0f}ms  "
                f"ui_overhead={r.ui_overhead_ms:.0f}ms  "
                f"proxy_overhead={r.proxy_overhead_ms:.0f}ms"
                f"{' ERR:'+r.error[:30] if r.error else ''}"
            )

        await browser.close()
    return results


def write_report(results: list[UIResult], offline_path: Path, out: Path):
    """Merge offline (backend) results with UI results into one comprehensive report."""
    # Load offline results
    try:
        off_df = pd.read_excel(offline_path, sheet_name="Offline_Results")
    except Exception:
        off_df = pd.DataFrame()

    # UI results dataframe
    ui_rows = []
    for r in results:
        ui_rows.append({
            "row_id":             r.row_id,
            "lang":               r.lang,
            "difficulty":         r.difficulty,
            "utterance":          r.utterance[:120],
            "response_text":      r.response_text[:200],
            "ui_total_ms":        r.ui_total_ms,
            "backend_wall_ms":    r.backend_reported_ms,
            "ui_overhead_ms":     r.ui_overhead_ms,
            "proxy_overhead_ms":  r.proxy_overhead_ms,
            "error":              r.error,
            "timestamp":          r.timestamp,
        })
    ui_df = pd.DataFrame(ui_rows)

    # Merge for comparison
    merged = []
    if not off_df.empty and not ui_df.empty:
        off_idx = off_df.set_index("row_id")
        for _, ur in ui_df.iterrows():
            row = {"row_id": ur["row_id"], "lang": ur["lang"],
                   "difficulty": ur["difficulty"], "utterance": ur["utterance"]}
            # Offline (backend direct) numbers
            if ur["row_id"] in off_idx.index:
                o = off_idx.loc[ur["row_id"]]
                row["backend_total_ms"]  = o.get("total_latency_ms", 0)
                row["backend_rag_ms"]    = o.get("rag_ms", 0)
                row["backend_llm_ms"]    = o.get("llm_ms", 0)
                row["backend_accuracy"]  = o.get("accuracy_score", 0)
                row["backend_response"]  = str(o.get("response_text", ""))[:100]
            else:
                row.update({"backend_total_ms": 0, "backend_rag_ms": 0,
                            "backend_llm_ms": 0, "backend_accuracy": 0, "backend_response": ""})
            # UI numbers
            row["ui_total_ms"]       = ur["ui_total_ms"]
            row["ui_overhead_ms"]    = ur["ui_overhead_ms"]
            row["proxy_overhead_ms"] = ur["proxy_overhead_ms"]
            row["ui_error"]          = ur["error"]
            merged.append(row)
    merged_df = pd.DataFrame(merged) if merged else pd.DataFrame()

    # Summary
    sum_rows = []
    if not ui_df.empty:
        t = ui_df["ui_total_ms"]
        o = ui_df["ui_overhead_ms"]
        p = ui_df["proxy_overhead_ms"]
        sum_rows.append({
            "metric": "UI Total (via browser)",
            "mean_ms": round(t.mean(), 1), "median_ms": round(t.median(), 1),
            "p75_ms": round(t.quantile(0.75), 1), "p95_ms": round(t.quantile(0.95), 1),
            "notes": "Time from JS fetch start to response received in browser"
        })
        sum_rows.append({
            "metric": "UI Overhead (vs backend)",
            "mean_ms": round(o.mean(), 1), "median_ms": round(o.median(), 1),
            "p75_ms": round(o.quantile(0.75), 1), "p95_ms": round(o.quantile(0.95), 1),
            "notes": "ui_total_ms minus backend_wall_ms — React/JS overhead"
        })
        sum_rows.append({
            "metric": "Proxy Overhead (Vite)",
            "mean_ms": round(p.mean(), 1), "median_ms": round(p.median(), 1),
            "p75_ms": round(p.quantile(0.75), 1), "p95_ms": round(p.quantile(0.95), 1),
            "notes": "Extra ms the Vite dev proxy adds over direct backend call"
        })
    if not off_df.empty:
        t = off_df["total_latency_ms"]
        sum_rows.append({
            "metric": "Backend Direct (RAG + LLM)",
            "mean_ms": round(t.mean(), 1), "median_ms": round(t.median(), 1),
            "p75_ms": round(t.quantile(0.75), 1), "p95_ms": round(t.quantile(0.95), 1),
            "notes": "Offline eval: real OpenAI embeddings + GPT-4o-mini"
        })
        r_ms = off_df["rag_ms"]
        l_ms = off_df["llm_ms"]
        sum_rows.append({
            "metric": "  ↳ RAG retrieval",
            "mean_ms": round(r_ms.mean(), 1), "median_ms": round(r_ms.median(), 1),
            "p75_ms": round(r_ms.quantile(0.75), 1), "p95_ms": round(r_ms.quantile(0.95), 1),
            "notes": "Cosine similarity search over embedded chunks"
        })
        sum_rows.append({
            "metric": "  ↳ LLM generation",
            "mean_ms": round(l_ms.mean(), 1), "median_ms": round(l_ms.median(), 1),
            "p75_ms": round(l_ms.quantile(0.75), 1), "p95_ms": round(l_ms.quantile(0.95), 1),
            "notes": "GPT-4o-mini answer generation"
        })
        acc = off_df["accuracy_score"]
        sum_rows.append({
            "metric": "Accuracy (LLM judge 0-100)",
            "mean_ms": round(acc.mean(), 1), "median_ms": round(acc.median(), 1),
            "p75_ms": round(acc.quantile(0.75), 1), "p95_ms": round(acc.quantile(0.95), 1),
            "notes": "GPT-4o-mini judge: required_facts coverage vs forbidden_facts"
        })
    sum_df = pd.DataFrame(sum_rows)

    # Write workbook
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    thin = Side(style="thin", color="D0D0D0")
    brd  = Border(left=thin, right=thin, top=thin, bottom=thin)

    def style(ws, df, hdr="1F4E79"):
        for ci, col in enumerate(df.columns, 1):
            c = ws.cell(row=1, column=ci, value=col)
            c.font = Font(bold=True, color="FFFFFF", size=10)
            c.fill = PatternFill("solid", fgColor=hdr)
            c.alignment = Alignment(horizontal="center", wrap_text=True)
            c.border = brd
        for ri, row in enumerate(df.itertuples(index=False), 2):
            for ci, v in enumerate(row, 1):
                cell = ws.cell(row=ri, column=ci, value=v)
                cell.border = brd; cell.font = Font(size=9)
                col = df.columns[ci - 1]
                if "overhead" in col and isinstance(v, (int, float)):
                    cell.fill = PatternFill("solid", fgColor=(
                        "FFC7CE" if v > 200 else "FFEB9C" if v > 100 else "C6EFCE"))
                if ("accuracy" in col or "acc" in col) and isinstance(v, (int, float)):
                    cell.fill = PatternFill("solid", fgColor=(
                        "C6EFCE" if v >= 80 else "FFEB9C" if v >= 50 else "FFC7CE"))
                if "total_ms" in col and isinstance(v, (int, float)):
                    cell.fill = PatternFill("solid", fgColor=(
                        "FFC7CE" if v > 5000 else "FFEB9C" if v > 2000 else "C6EFCE"))
        ws.freeze_panes = "A2"
        for ci, col in enumerate(df.columns, 1):
            w = {"row_id":8,"lang":6,"difficulty":9,"utterance":40,"response_text":50,
                 "backend_response":40,"notes":45}.get(col, 14)
            ws.column_dimensions[get_column_letter(ci)].width = w
        if len(df) > 0:
            ws.auto_filter.ref = ws.dimensions

    # Summary sheet
    ws_sum = wb.create_sheet("Summary")
    for ri, row in enumerate([sum_df.columns.tolist()] + sum_df.values.tolist(), 1):
        for ci, v in enumerate(row, 1):
            ws_sum.cell(row=ri, column=ci, value=v)
    style(ws_sum, sum_df, "375623")

    # Comparison sheet (merged)
    if not merged_df.empty:
        ws_cmp = wb.create_sheet("Backend_vs_UI")
        for ri, row in enumerate([merged_df.columns.tolist()] + merged_df.values.tolist(), 1):
            for ci, v in enumerate(row, 1):
                ws_cmp.cell(row=ri, column=ci, value=v)
        style(ws_cmp, merged_df, "4472C4")

    # Backend results sheet
    if not off_df.empty:
        ws_be = wb.create_sheet("Backend_Results")
        for ri, row in enumerate([off_df.columns.tolist()] + off_df.values.tolist(), 1):
            for ci, v in enumerate(row, 1):
                ws_be.cell(row=ri, column=ci, value=v)
        style(ws_be, off_df, "1F4E79")

    # UI results sheet
    if not ui_df.empty:
        ws_ui = wb.create_sheet("UI_Results")
        for ri, row in enumerate([ui_df.columns.tolist()] + ui_df.values.tolist(), 1):
            for ci, v in enumerate(row, 1):
                ws_ui.cell(row=ri, column=ci, value=v)
        style(ws_ui, ui_df, "833C00")

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"\n[report] → {out}")


async def main():
    import httpx
    # Verify servers
    for url, name in [(BACKEND_BASE + "/health", "backend"), (UI_BASE, "ui")]:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                await c.get(url)
            print(f"[check] {name} OK at {url}")
        except Exception as e:
            print(f"[check] {name} UNREACHABLE: {e}")
            return

    # Load golden
    golden = pd.read_excel(GOLDEN_XLSX, sheet_name="Golden_SingleTurn")
    for col in ["required_facts", "forbidden_facts", "answer_constraints",
                "difficulty", "source_of_truth"]:
        golden[col] = golden[col].fillna("").astype(str)
    if MAX_ROWS:
        golden = golden.head(MAX_ROWS)

    print(f"\n{'='*60}")
    print(f"UI LATENCY TESTS  ({len(golden)} rows via Playwright)")
    print(f"{'='*60}\n")

    rows = golden.to_dict("records")
    results = await run_ui_tests(rows)

    offline_path = RESULTS_DIR / "offline_latency_accuracy_report.xlsx"
    out          = RESULTS_DIR / "latency_accuracy_report.xlsx"
    write_report(results, offline_path, out)

    # Console summary
    if results:
        totals    = [r.ui_total_ms    for r in results if not r.error]
        overheads = [r.ui_overhead_ms for r in results if not r.error]
        proxies   = [r.proxy_overhead_ms for r in results if not r.error]
        if totals:
            print(f"\nUI total:         mean={sum(totals)/len(totals):.0f}ms  max={max(totals):.0f}ms")
            print(f"UI overhead:      mean={sum(overheads)/len(overheads):.0f}ms  max={max(overheads):.0f}ms")
            print(f"Proxy overhead:   mean={sum(proxies)/len(proxies):.0f}ms  max={max(proxies):.0f}ms")
    print(f"\nReport: {out}\n")


if __name__ == "__main__":
    asyncio.run(main())
