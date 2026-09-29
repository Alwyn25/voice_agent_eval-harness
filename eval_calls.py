"""
Highland Greenz — Real Backend Call Latency + Accuracy Eval (v2)
================================================================
Runs golden-dataset rows through the real FastAPI backend (Docker Compose)
and the real Vite UI proxy, across four scenarios:

  ① INBOUND  backend  — direct to backend API (channel=phone, direction=inbound)
  ② INBOUND  ui       — through Vite /api proxy
  ③ OUTBOUND backend  — direct; identity confirmation turn prepended
  ④ OUTBOUND ui

Each row is classified BEFORE running as:
  with_tool    — expected_tool_calls is non-empty in the golden dataset
  without_tool — pure Q&A / RAG, no tools expected

Timing fields from debug.timing:
  planner_ms   — intent understanding + routing (LLM call)
  retrieval_ms — pgvector similarity search
  tool_ms      — tool execution time (0 when no tool was called)
  responder_ms — response generation (LLM call)
  wall_ms      — total backend wall time

Actual tool execution is confirmed via executed_tools in the API response.

Accuracy judged by GPT-4o-mini comparing response vs required_facts / forbidden_facts.

Summary reports mean AND median for all timing metrics, split by tool/no-tool.
"""

import ast
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import openai
import pandas as pd
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ── Load credentials from backend .env (no key on command line) ───────────────
_backend_env = Path(__file__).parent.parent / "AI-Voice-Agent-Backend-main" / ".env"
if _backend_env.exists():
    for _line in _backend_env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()

# ── Config ─────────────────────────────────────────────────────────────────────
GOLDEN_XLSX  = r"C:\Users\ADVORA-ALWYN\OneDrive\Desktop\highland_greenz_golden_dataset.xlsx"
BACKEND_BASE = "http://localhost:8000"
UI_BASE      = "http://localhost:3000"
RESULTS_DIR  = Path(__file__).parent / "results"

LOGIN_EMAIL  = "admin@advora.ai"
LOGIN_PASS   = "Admin@123"

# Rows PER CATEGORY (with_tool / without_tool). Total rows = 2 × MAX_PER_CAT.
MAX_PER_CAT  = int(os.getenv("MAX_PER_CAT", "15"))

_oai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


# ── Data model ─────────────────────────────────────────────────────────────────
@dataclass
class CallResult:
    row_id:        str
    lang:          str
    difficulty:    str
    utterance:     str
    direction:     str          # "inbound" | "outbound"
    path:          str          # "backend" | "ui"
    tool_category: str = ""     # "with_tool" | "without_tool"
    expected_tools: str = ""    # from golden dataset
    channel:       str = "phone"
    response_text: str = ""

    # Timing from debug.timing
    planner_ms:    float = 0
    retrieval_ms:  float = 0
    tool_ms:       float = 0    # 0 when no tool was called
    responder_ms:  float = 0
    total_ms:      float = 0    # Python-measured (includes network round-trip)
    wall_ms:       float = 0    # backend-reported

    # Actual tool execution from API response
    executed_tools: str = ""    # comma-separated list of tools that ran
    tool_called:    bool = False

    # Accuracy
    accuracy_score: float = 0
    facts_hit:      str = ""
    forbidden_hit:  str = ""
    error:          str = ""
    timestamp:      str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Auth + customer pool ───────────────────────────────────────────────────────
async def get_token(base_url: str = BACKEND_BASE) -> str:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{base_url}/auth/login",
                         json={"email": LOGIN_EMAIL, "password": LOGIN_PASS})
        r.raise_for_status()
        return r.json()["access_token"]


async def get_customer_ids(token: str) -> list[str]:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{BACKEND_BASE}/customers?page_size=50",
                        headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        items = r.json().get("items", [])
        return [it["id"] for it in items if it.get("id")]


# ── LLM accuracy judge ─────────────────────────────────────────────────────────
def judge(utterance: str, response: str, required: str,
          forbidden: str, constraints: str) -> dict:
    system = (
        "You are a strict evaluator for a real-estate voice-agent. "
        "Score the agent response 0-100 based on REQUIRED_FACTS coverage "
        "and FORBIDDEN_FACTS avoidance. "
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
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=30,
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {"score": -1, "reason": str(e), "facts_hit": [], "forbidden_hit": []}


# ── Single turn ────────────────────────────────────────────────────────────────
async def run_turn(client: httpx.AsyncClient,
                   row: dict,
                   direction: str,
                   path: str,
                   token: str,
                   base_url: str,
                   customer_ids: list[str] | None = None,
                   customer_index: int = 0) -> CallResult:
    res = CallResult(
        row_id=str(row["id"]),
        lang=str(row.get("lang", "")),
        difficulty=str(row.get("difficulty", "")),
        utterance=str(row["user_utterance"]),
        direction=direction,
        path=path,
        tool_category=str(row.get("_tool_category", "")),
        expected_tools=str(row.get("_expected_tools", "")),
    )
    hdrs = {"Authorization": f"Bearer {token}"}
    api = f"{base_url}/api" if path == "ui" else base_url
    session_api = BACKEND_BASE if (path == "ui" and direction == "outbound") else api

    try:
        sess_body: dict = {"channel": "phone", "direction": direction, "purpose": "sales"}
        if direction == "outbound" and customer_ids:
            sess_body["customer_id"] = customer_ids[customer_index % len(customer_ids)]

        sr = await client.post(f"{session_api}/sessions", json=sess_body,
                               headers=hdrs, timeout=20)
        sr.raise_for_status()
        sid = sr.json()["session_id"]

        if direction == "outbound":
            await client.post(f"{api}/sessions/{sid}/turns",
                              json={"text": "Yes, this is me. I was interested in 3BHK options."},
                              headers=hdrs, timeout=60)

        t0 = time.perf_counter()
        tr = await client.post(f"{api}/sessions/{sid}/turns",
                               json={"text": res.utterance},
                               headers=hdrs, timeout=120)
        tr.raise_for_status()
        t1 = time.perf_counter()

        data = tr.json()
        res.response_text = data.get("response_text", "")
        res.total_ms      = round((t1 - t0) * 1000, 1)

        timing = (data.get("debug") or {}).get("timing") or {}
        res.wall_ms      = timing.get("wall_ms", 0)
        res.planner_ms   = timing.get("planner_ms", 0)
        res.retrieval_ms = timing.get("retrieval_ms", 0)
        res.tool_ms      = timing.get("tool_ms", 0)
        res.responder_ms = timing.get("responder_ms", 0)

        # Actual tool execution from response
        executed = data.get("executed_tools") or []
        if isinstance(executed, list):
            tool_names = [
                (t.get("tool") or t.get("name") or str(t)) if isinstance(t, dict) else str(t)
                for t in executed
            ]
            res.executed_tools = ", ".join(tool_names)
            res.tool_called    = len(tool_names) > 0
        else:
            res.executed_tools = str(executed) if executed else ""
            res.tool_called    = bool(executed)

        await client.post(f"{session_api}/sessions/{sid}/end",
                          json={"analyse": "skip"}, headers=hdrs, timeout=10)

    except Exception as e:
        res.error = str(e)[:120]
        return res

    verdict = judge(
        res.utterance, res.response_text,
        str(row.get("required_facts") or ""),
        str(row.get("forbidden_facts") or ""),
        str(row.get("answer_constraints") or ""),
    )
    res.accuracy_score = verdict.get("score", 0) or 0
    res.facts_hit      = "; ".join(verdict.get("facts_hit") or [])
    res.forbidden_hit  = "; ".join(verdict.get("forbidden_hit") or [])
    return res


# ── Run a scenario ─────────────────────────────────────────────────────────────
async def run_scenario(rows: list[dict],
                       direction: str,
                       path: str,
                       token: str,
                       base_url: str,
                       customer_ids: list[str] | None = None) -> list[CallResult]:
    results = []
    label = f"{direction.upper()} via {path.upper()}"
    async with httpx.AsyncClient(timeout=180) as client:
        for i, row in enumerate(rows, 1):
            cat = row.get("_tool_category", "")
            print(f"  [{label} {i}/{len(rows)}] {row['id']} [{cat}]  "
                  f"{str(row['user_utterance'])[:48]!r}")
            r = await run_turn(client, row, direction, path, token, base_url,
                               customer_ids=customer_ids, customer_index=i - 1)
            results.append(r)
            tool_flag = f" TOOL={r.executed_tools[:30]}" if r.tool_called else ""
            print(
                f"    plan={r.planner_ms:.0f}  ret={r.retrieval_ms:.0f}  "
                f"tool={r.tool_ms:.0f}  resp={r.responder_ms:.0f}  "
                f"wall={r.wall_ms:.0f} ms  acc={r.accuracy_score:.0f}"
                f"{tool_flag}"
                f"{(' ERR:' + r.error[:50]) if r.error else ''}"
            )
    return results


# ── Stats helpers ──────────────────────────────────────────────────────────────
def _stat(vals: list[float]) -> tuple[float, float, float]:
    """Return (mean, median, p95) rounded to 1 dp."""
    if not vals:
        return 0.0, 0.0, 0.0
    s = sorted(vals)
    mean   = round(sum(s) / len(s), 1)
    median = round(statistics.median(s), 1)
    p95    = round(s[int(len(s) * 0.95)], 1)
    return mean, median, p95


def _grp_stats(results: list[CallResult], field_name: str) -> tuple[float, float, float]:
    vals = [getattr(r, field_name) for r in results if not r.error]
    return _stat(vals)


# ── Excel output ───────────────────────────────────────────────────────────────
COLS = [
    "row_id", "lang", "difficulty", "utterance",
    "direction", "path", "tool_category", "expected_tools",
    "planner_ms", "retrieval_ms", "tool_ms", "responder_ms", "total_ms", "wall_ms",
    "tool_called", "executed_tools",
    "accuracy_score", "facts_hit", "forbidden_hit",
    "response_text", "error", "timestamp",
]

COL_W = {
    "row_id": 8, "lang": 6, "difficulty": 9, "utterance": 42,
    "direction": 10, "path": 9, "tool_category": 14, "expected_tools": 28,
    "planner_ms": 11, "retrieval_ms": 12, "tool_ms": 10, "responder_ms": 12,
    "total_ms": 11, "wall_ms": 10,
    "tool_called": 10, "executed_tools": 32,
    "accuracy_score": 12, "facts_hit": 36, "forbidden_hit": 28,
    "response_text": 52, "error": 28, "timestamp": 20,
}


def _brd():
    s = Side(style="thin", color="D0D0D0")
    return Border(left=s, right=s, top=s, bottom=s)


def _style_sheet(ws, df: pd.DataFrame, hdr_color: str):
    b = _brd()
    for ci, col in enumerate(df.columns, 1):
        c = ws.cell(row=1, column=ci, value=col)
        c.font = Font(bold=True, color="FFFFFF", size=10)
        c.fill = PatternFill("solid", fgColor=hdr_color)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        c.border = b
    for ri, row in enumerate(df.itertuples(index=False), 2):
        for ci, v in enumerate(row, 1):
            cell = ws.cell(row=ri, column=ci, value=v)
            cell.border = b
            cell.font = Font(size=9)
            col = df.columns[ci - 1]
            if col == "accuracy_score" and isinstance(v, (int, float)):
                cell.fill = PatternFill("solid", fgColor=(
                    "C6EFCE" if v >= 80 else "FFEB9C" if v >= 50 else "FFC7CE"))
            if col in ("total_ms", "wall_ms") and isinstance(v, (int, float)):
                cell.fill = PatternFill("solid", fgColor=(
                    "FFC7CE" if v > 5000 else "FFEB9C" if v > 3000 else "C6EFCE"))
            if col == "tool_ms" and isinstance(v, (int, float)) and v > 0:
                cell.fill = PatternFill("solid", fgColor="DDEEFF")
            if col == "error" and v:
                cell.fill = PatternFill("solid", fgColor="FFC7CE")
            if col == "tool_called":
                cell.fill = PatternFill("solid", fgColor=(
                    "EAF4FF" if v else "F5F5F5"))
    for ci, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(ci)].width = COL_W.get(col, 14)
    ws.freeze_panes = "A2"
    if len(df) > 0:
        ws.auto_filter.ref = ws.dimensions


def write_excel(all_results: list[CallResult], out: Path):
    full_df = pd.DataFrame(
        [{c: getattr(r, c, "") for c in COLS} for r in all_results],
        columns=COLS,
    )

    # ── Summary sheet ──────────────────────────────────────────────────────────
    sum_rows = []
    ok_results = [r for r in all_results if not r.error]
    groups = [
        ("All",          ok_results),
        ("with_tool",    [r for r in ok_results if r.tool_category == "with_tool"]),
        ("without_tool", [r for r in ok_results if r.tool_category == "without_tool"]),
    ]
    for grp_label, grp in groups:
        for direction in ("inbound", "outbound"):
            for path in ("backend", "ui"):
                subset = [r for r in grp if r.direction == direction and r.path == path]
                if not subset:
                    continue
                plan_m, plan_med, plan_p95 = _grp_stats(subset, "planner_ms")
                ret_m,  ret_med,  ret_p95  = _grp_stats(subset, "retrieval_ms")
                tool_m, tool_med, tool_p95 = _grp_stats(subset, "tool_ms")
                resp_m, resp_med, resp_p95 = _grp_stats(subset, "responder_ms")
                tot_m,  tot_med,  tot_p95  = _grp_stats(subset, "total_ms")
                wall_m, wall_med, wall_p95 = _grp_stats(subset, "wall_ms")
                acc_m,  acc_med,  _        = _grp_stats(subset, "accuracy_score")
                n_tool = sum(1 for r in subset if r.tool_called)
                sum_rows.append({
                    "group":              grp_label,
                    "direction":          direction,
                    "path":               path,
                    "n":                  len(subset),
                    "n_tool_called":      n_tool,
                    "wall_mean_ms":       wall_m,
                    "wall_median_ms":     wall_med,
                    "wall_p95_ms":        wall_p95,
                    "planner_mean_ms":    plan_m,
                    "planner_median_ms":  plan_med,
                    "retrieval_mean_ms":  ret_m,
                    "retrieval_median_ms":ret_med,
                    "tool_mean_ms":       tool_m,
                    "tool_median_ms":     tool_med,
                    "responder_mean_ms":  resp_m,
                    "responder_median_ms":resp_med,
                    "accuracy_mean":      acc_m,
                    "accuracy_median":    acc_med,
                    "errors":             len([r for r in all_results
                                              if r.direction == direction and r.path == path
                                              and r.tool_category == grp_label.split("_")[0]
                                              and r.error]),
                })

    sum_df  = pd.DataFrame(sum_rows)
    tool_df = full_df[full_df["tool_category"] == "with_tool"].reset_index(drop=True)
    notool_df = full_df[full_df["tool_category"] == "without_tool"].reset_index(drop=True)
    ib_df   = full_df[full_df["direction"] == "inbound"].reset_index(drop=True)
    ob_df   = full_df[full_df["direction"] == "outbound"].reset_index(drop=True)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Summary
    ws = wb.create_sheet("Summary")
    for ri, row in enumerate([sum_df.columns.tolist()] + sum_df.values.tolist(), 1):
        for ci, v in enumerate(row, 1):
            c = ws.cell(row=ri, column=ci, value=v)
            if ri == 1:
                c.font = Font(bold=True, color="FFFFFF", size=10)
                c.fill = PatternFill("solid", fgColor="375623")
                c.alignment = Alignment(horizontal="center", wrap_text=True)
            c.border = _brd()
    ws.freeze_panes = "A2"
    for ci, col in enumerate(sum_df.columns, 1):
        ws.column_dimensions[get_column_letter(ci)].width = max(12, len(col) + 2)

    # With tool calls
    ws2 = wb.create_sheet("With_Tool_Calls")
    for ri, row in enumerate([tool_df.columns.tolist()] + tool_df.values.tolist(), 1):
        for ci, v in enumerate(row, 1):
            ws2.cell(row=ri, column=ci, value=v)
    _style_sheet(ws2, tool_df, "1F4E79")

    # Without tool calls
    ws3 = wb.create_sheet("Without_Tool_Calls")
    for ri, row in enumerate([notool_df.columns.tolist()] + notool_df.values.tolist(), 1):
        for ci, v in enumerate(row, 1):
            ws3.cell(row=ri, column=ci, value=v)
    _style_sheet(ws3, notool_df, "375623")

    # Inbound
    ws4 = wb.create_sheet("Inbound_Results")
    for ri, row in enumerate([ib_df.columns.tolist()] + ib_df.values.tolist(), 1):
        for ci, v in enumerate(row, 1):
            ws4.cell(row=ri, column=ci, value=v)
    _style_sheet(ws4, ib_df, "4472C4")

    # Outbound
    ws5 = wb.create_sheet("Outbound_Results")
    for ri, row in enumerate([ob_df.columns.tolist()] + ob_df.values.tolist(), 1):
        for ci, v in enumerate(row, 1):
            ws5.cell(row=ri, column=ci, value=v)
    _style_sheet(ws5, ob_df, "833C00")

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"\n[report] → {out}")
    return sum_df


# ── Console summary ────────────────────────────────────────────────────────────
def print_summary(all_results: list[CallResult]):
    ok = [r for r in all_results if not r.error]
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)

    for cat in ("All", "with_tool", "without_tool"):
        grp = ok if cat == "All" else [r for r in ok if r.tool_category == cat]
        if not grp:
            continue
        print(f"\n  ── {cat.upper().replace('_', ' ')} (n={len(grp)}) ──")
        for direction in ("inbound", "outbound"):
            for path in ("backend", "ui"):
                sub = [r for r in grp if r.direction == direction and r.path == path]
                if not sub:
                    continue
                wm, wmed, wp95 = _stat([r.wall_ms for r in sub])
                pm, pmed, _    = _stat([r.planner_ms for r in sub])
                rm, rmed, _    = _stat([r.retrieval_ms for r in sub])
                tm, tmed, _    = _stat([r.tool_ms for r in sub])
                rsm, rsmed, _  = _stat([r.responder_ms for r in sub])
                acc, accmed, _ = _stat([r.accuracy_score for r in sub])
                n_tool = sum(1 for r in sub if r.tool_called)
                print(f"\n    {direction.upper()} via {path.upper()}  n={len(sub)}  tools_ran={n_tool}")
                print(f"      Wall      mean={wm:.0f}  median={wmed:.0f}  p95={wp95:.0f} ms")
                print(f"      Planner   mean={pm:.0f}  median={pmed:.0f} ms")
                print(f"      Retrieval mean={rm:.0f}  median={rmed:.0f} ms")
                print(f"      Tool      mean={tm:.0f}  median={tmed:.0f} ms")
                print(f"      Responder mean={rsm:.0f}  median={rsmed:.0f} ms")
                print(f"      Accuracy  mean={acc:.1f}  median={accmed:.1f}/100")


# ── Entry point ────────────────────────────────────────────────────────────────
async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set."); return

    async with httpx.AsyncClient(timeout=8) as c:
        for url, name in [(f"{BACKEND_BASE}/health", "backend"), (UI_BASE, "ui (optional)")]:
            try:
                r = await c.get(url)
                print(f"[check] {name}: {r.status_code} OK")
            except Exception as e:
                if name == "backend":
                    print(f"[check] backend UNREACHABLE: {e}"); return
                else:
                    print(f"[check] ui not reachable — skipping UI path")

    print("[auth] logging in...")
    token = await get_token(BACKEND_BASE)
    print(f"[auth] token: {token[:30]}...")

    customer_ids = await get_customer_ids(token)
    print(f"[customers] {len(customer_ids)} available for outbound")

    # ── Load and split golden dataset ──────────────────────────────────────────
    # Work from a temp copy to avoid OneDrive file lock
    import shutil
    tmp = Path(r"C:\Users\ADVORA~1\AppData\Local\Temp\golden_eval_tmp.xlsx")
    try:
        shutil.copy2(GOLDEN_XLSX, tmp)
    except PermissionError:
        fallback = Path(r"C:\Users\ADVORA~1\AppData\Local\Temp\golden_tmp.xlsx")
        if fallback.exists():
            tmp = fallback
            print(f"[golden] using cached copy: {tmp}")
        else:
            print("[golden] ERROR: cannot read golden dataset (file locked, no cache)"); return
    golden = pd.read_excel(tmp, sheet_name="Golden_SingleTurn")
    for col in ["required_facts", "forbidden_facts", "answer_constraints",
                "difficulty", "source_of_truth", "expected_tool_calls"]:
        golden[col] = golden[col].fillna("").astype(str)

    def _parse_tools(v: str) -> list:
        v = v.strip()
        if not v or v in ("[]", "nan"): return []
        try: return ast.literal_eval(v)
        except: return [v]

    golden["_tools_list"]    = golden["expected_tool_calls"].apply(_parse_tools)
    golden["_has_tool"]      = golden["_tools_list"].apply(len) > 0
    golden["_tool_category"] = golden["_has_tool"].map({True: "with_tool", False: "without_tool"})
    golden["_expected_tools"]= golden["_tools_list"].apply(
        lambda t: ", ".join(t) if t else "")

    tool_rows   = golden[golden["_has_tool"]].head(MAX_PER_CAT).to_dict("records")
    notool_rows = golden[~golden["_has_tool"]].head(MAX_PER_CAT).to_dict("records")
    rows        = tool_rows + notool_rows

    print(f"\n[golden] {len(rows)} rows total: "
          f"{len(tool_rows)} with_tool + {len(notool_rows)} without_tool")
    print(f"  Tool types: "
          f"{', '.join(set(t for r in tool_rows for t in r['_tools_list']))}")
    print()

    all_results: list[CallResult] = []

    for direction in ("inbound", "outbound"):
        for path, base_url in (("backend", BACKEND_BASE), ("ui", UI_BASE)):
            print("=" * 68)
            print(f"{direction.upper()} via {path.upper()}  ({len(rows)} rows: "
                  f"{len(tool_rows)} tool + {len(notool_rows)} no-tool)")
            print("=" * 68)
            res = await run_scenario(
                rows, direction, path, token, base_url,
                customer_ids=customer_ids if direction == "outbound" else None,
            )
            all_results.extend(res)
            print()

    print_summary(all_results)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out = RESULTS_DIR / f"eval_tool_vs_notool_{ts}.xlsx"
    write_excel(all_results, out)
    print(f"\nReport: {out}")

    # Return summary data for HTML report
    return all_results


if __name__ == "__main__":
    asyncio.run(main())
