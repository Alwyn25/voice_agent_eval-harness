"""
Highland Greenz — Golden Eval Final (v1)
=========================================
Runs BOTH single-turn AND multi-turn thread evaluations via the UI path
(Vite :3000 proxy → backend :8000).

Test matrix:
  ① INBOUND  single-turn  with_tool rows    (up to MAX_ST each)
  ② INBOUND  single-turn  without_tool rows (up to MAX_ST each)
  ③ OUTBOUND single-turn  with_tool rows    (up to MAX_ST each)
  ④ OUTBOUND single-turn  without_tool rows (up to MAX_ST each)
  ⑤ INBOUND  thread       with_tool threads (up to MAX_TH_TOOL threads)
  ⑥ INBOUND  thread       without_tool threads (up to MAX_TH_NOTOOL)
  ⑦ OUTBOUND thread       with_tool threads (up to MAX_TH_TOOL threads)
  ⑧ OUTBOUND thread       without_tool threads (up to MAX_TH_NOTOOL)

Output:
  results/golden_eval_final.xlsx   — 7-sheet workbook
  Console: per-stage latency + accuracy + tool-fire rate
"""

import ast
import asyncio
import json
import os
import re
import shutil
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

# ── Load .env ──────────────────────────────────────────────────────────────────
_env = Path(__file__).parent.parent / "AI-Voice-Agent-Backend-main" / ".env"
if _env.exists():
    for _ln in _env.read_text(encoding="utf-8").splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _, _v = _ln.partition("=")
            k = _k.strip()
            if k not in os.environ:
                os.environ[k] = _v.strip()

# ── Config ─────────────────────────────────────────────────────────────────────
GOLDEN_XLSX  = r"C:\Users\ADVORA-ALWYN\OneDrive\Desktop\highland_greenz_golden_dataset.xlsx"
BACKEND_BASE = "http://localhost:8000"
UI_BASE      = "http://localhost:3000"
RESULTS_DIR  = Path(__file__).parent / "results"
LOGIN_EMAIL  = "admin@advora.ai"
LOGIN_PASS   = "Admin@123"

MAX_ST       = int(os.getenv("MAX_ST",       "10"))   # single-turn rows per category
MAX_TH_TOOL  = int(os.getenv("MAX_TH_TOOL",  "5"))    # thread sessions with tool calls
MAX_TH_NOTOOL= int(os.getenv("MAX_TH_NOTOOL","2"))    # thread sessions without tool calls

_oai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

_CONFIRM_RE = re.compile(
    r"(shall i|should i|would you like me to|want me to|go ahead|confirm|proceed|"
    r"is that right|can i go ahead|shall we|do you want me)",
    re.IGNORECASE,
)


# ── Data models ────────────────────────────────────────────────────────────────
@dataclass
class EvalRow:
    # identity
    test_type:      str   = "single_turn"   # "single_turn" | "thread"
    thread_id:      str   = ""
    archetype:      str   = ""
    turn_no:        int   = 1
    row_id:         str   = ""
    lang:           str   = "en"
    difficulty:     str   = ""
    utterance:      str   = ""
    direction:      str   = "inbound"
    tool_category:  str   = ""              # "with_tool" | "without_tool"
    expected_tools: str   = ""
    tool_expected:  bool  = False
    # timing
    planner_ms:     float = 0
    retrieval_ms:   float = 0
    tool_ms:        float = 0
    responder_ms:   float = 0
    wall_ms:        float = 0
    # execution
    executed_tools: str   = ""
    tool_called:    bool  = False
    tool_match:     bool  = False           # expected == actual
    confirmed_turn: bool  = False
    # quality
    accuracy_score: float = 0
    facts_hit:      str   = ""
    forbidden_hit:  str   = ""
    response_text:  str   = ""
    error:          str   = ""
    timestamp:      str   = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Auth ───────────────────────────────────────────────────────────────────────
async def get_token() -> str:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{BACKEND_BASE}/auth/login",
                         json={"email": LOGIN_EMAIL, "password": LOGIN_PASS})
        r.raise_for_status()
        return r.json()["access_token"]


# ── Customers ──────────────────────────────────────────────────────────────────
async def get_all_customers(token: str) -> list[str]:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{BACKEND_BASE}/customers?page_size=50",
                        headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        ids = [it["id"] for it in r.json().get("items", []) if it.get("id")]
    print(f"[customers] {len(ids)} available (policy enforcement bypassed)")
    return ids


# ── Tool parsing (handles unquoted bracket lists from golden dataset) ─────────
def parse_tools(v) -> list[str]:
    if not v or (isinstance(v, float) and pd.isna(v)):
        return []
    s = str(v).strip()
    if not s or s in ("[]", "nan", "None", ""):
        return []
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return [str(x) for x in result]
    except Exception:
        pass
    # Fallback: extract identifiers from [tool_a, tool_b] notation
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", s)
    return tokens if tokens else []


# ── Accuracy judge ─────────────────────────────────────────────────────────────
def judge(utterance: str, response: str, required: str,
          forbidden: str, constraints: str = "") -> dict:
    if not required and not forbidden:
        return {"score": -1, "reason": "no facts to judge", "facts_hit": [], "forbidden_hit": []}
    system = (
        "You are a strict real-estate voice-agent evaluator. "
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
        rsp = _oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=30,
        )
        return json.loads(rsp.choices[0].message.content)
    except Exception as e:
        return {"score": -1, "reason": str(e), "facts_hit": [], "forbidden_hit": []}


# ── Response parser ───────────────────────────────────────────────────────────
def _parse_response(data: dict, row: EvalRow, elapsed_ms: float):
    row.response_text = data.get("response_text", "")
    timing = (data.get("debug") or {}).get("timing") or {}
    row.wall_ms      = timing.get("wall_ms", 0) or elapsed_ms
    row.planner_ms   = timing.get("planner_ms", 0)
    row.retrieval_ms = timing.get("retrieval_ms", 0)
    row.tool_ms      = timing.get("tool_ms", 0)
    row.responder_ms = timing.get("responder_ms", 0)
    executed = data.get("executed_tools") or []
    if isinstance(executed, list):
        names = [
            (t.get("tool") or t.get("name") or str(t)) if isinstance(t, dict) else str(t)
            for t in executed
        ]
        row.executed_tools = ", ".join(names)
        row.tool_called    = len(names) > 0
    else:
        row.executed_tools = str(executed) if executed else ""
        row.tool_called    = bool(executed)
    row.tool_match = (row.tool_called == row.tool_expected)


# ── Single-turn run ────────────────────────────────────────────────────────────
async def run_single_turn(client: httpx.AsyncClient, golden_row: dict,
                          direction: str, token: str,
                          customer_ids: list[str], cust_idx: int) -> EvalRow:
    expected = parse_tools(golden_row.get("expected_tool_calls", ""))
    row = EvalRow(
        test_type="single_turn",
        row_id=str(golden_row.get("id", "")),
        lang=str(golden_row.get("lang", "en")),
        difficulty=str(golden_row.get("difficulty", "")),
        utterance=str(golden_row["user_utterance"]),
        direction=direction,
        tool_category=str(golden_row.get("_tool_category", "")),
        expected_tools=", ".join(expected),
        tool_expected=len(expected) > 0,
    )
    hdrs = {"Authorization": f"Bearer {token}"}
    try:
        sess_body: dict = {"channel": "phone", "direction": direction,
                           "purpose": "sales", "enforce_outbound_policy": False}
        if direction == "outbound" and customer_ids:
            sess_body["customer_id"] = customer_ids[cust_idx % len(customer_ids)]

        sr = await client.post(f"{BACKEND_BASE}/sessions", json=sess_body,
                               headers=hdrs, timeout=20)
        sr.raise_for_status()
        sid = sr.json()["session_id"]

        if direction == "outbound":
            await client.post(f"{UI_BASE}/api/sessions/{sid}/turns",
                              json={"text": "Yes this is me. I was interested in 3BHK."},
                              headers=hdrs, timeout=60)

        t0 = time.perf_counter()
        tr = await client.post(f"{UI_BASE}/api/sessions/{sid}/turns",
                               json={"text": row.utterance},
                               headers=hdrs, timeout=120)
        tr.raise_for_status()
        t1 = time.perf_counter()

        _parse_response(tr.json(), row, (t1 - t0) * 1000)

        # Confirmation turn if agent asks and tool not yet fired
        if not row.tool_called and _CONFIRM_RE.search(row.response_text):
            row.confirmed_turn = True
            t2 = time.perf_counter()
            cr = await client.post(f"{UI_BASE}/api/sessions/{sid}/turns",
                                   json={"text": "Yes, go ahead please."},
                                   headers=hdrs, timeout=120)
            t3 = time.perf_counter()
            if cr.status_code == 200:
                cd = cr.json()
                ct = (cd.get("debug") or {}).get("timing") or {}
                row.tool_ms      += ct.get("tool_ms", 0)
                row.responder_ms += ct.get("responder_ms", 0)
                row.wall_ms      += ct.get("wall_ms", 0) or ((t3 - t2) * 1000)
                ex2 = cd.get("executed_tools") or []
                if isinstance(ex2, list) and ex2:
                    n2 = [(t.get("tool") or t.get("name") or str(t)) if isinstance(t, dict)
                          else str(t) for t in ex2]
                    prev = [x for x in row.executed_tools.split(", ") if x]
                    row.executed_tools = ", ".join(dict.fromkeys(prev + n2))
                    row.tool_called = True
                row.response_text = cd.get("response_text", row.response_text)
                row.tool_match = (row.tool_called == row.tool_expected)

        await client.post(f"{BACKEND_BASE}/sessions/{sid}/end",
                          json={"analyse": "skip"}, headers=hdrs, timeout=10)

    except Exception as e:
        row.error = str(e)[:150]
        return row

    v = judge(row.utterance, row.response_text,
              str(golden_row.get("required_facts") or ""),
              str(golden_row.get("forbidden_facts") or ""),
              str(golden_row.get("answer_constraints") or ""))
    row.accuracy_score = float(v.get("score", 0) or 0)
    row.facts_hit      = "; ".join(v.get("facts_hit") or [])
    row.forbidden_hit  = "; ".join(v.get("forbidden_hit") or [])
    return row


# ── Thread run (all turns in one session) ─────────────────────────────────────
async def run_thread(client: httpx.AsyncClient, thread_rows: list[dict],
                     direction: str, token: str, tool_category: str,
                     customer_ids: list[str], cust_idx: int) -> list[EvalRow]:
    thread_id = str(thread_rows[0].get("thread_id", ""))
    archetype  = str(thread_rows[0].get("archetype", ""))
    hdrs = {"Authorization": f"Bearer {token}"}
    results: list[EvalRow] = []

    try:
        sess_body: dict = {"channel": "phone", "direction": direction,
                           "purpose": "sales", "enforce_outbound_policy": False}
        if direction == "outbound" and customer_ids:
            sess_body["customer_id"] = customer_ids[cust_idx % len(customer_ids)]

        sr = await client.post(f"{BACKEND_BASE}/sessions", json=sess_body,
                               headers=hdrs, timeout=20)
        sr.raise_for_status()
        sid = sr.json()["session_id"]

        if direction == "outbound":
            await client.post(f"{UI_BASE}/api/sessions/{sid}/turns",
                              json={"text": "Yes this is me. I was interested in 3BHK."},
                              headers=hdrs, timeout=60)

        for tr_row in thread_rows:
            turn_no   = int(tr_row.get("turn_no", 0))
            utterance = str(tr_row.get("utterance", ""))
            expected  = parse_tools(tr_row.get("expected_tool_calls", ""))

            row = EvalRow(
                test_type="thread",
                thread_id=thread_id,
                archetype=archetype,
                turn_no=turn_no,
                lang=str(tr_row.get("lang", "en")),
                utterance=utterance,
                direction=direction,
                tool_category=tool_category,
                expected_tools=", ".join(expected),
                tool_expected=len(expected) > 0,
            )

            t0 = time.perf_counter()
            resp = await client.post(f"{UI_BASE}/api/sessions/{sid}/turns",
                                     json={"text": utterance},
                                     headers=hdrs, timeout=120)
            resp.raise_for_status()
            t1 = time.perf_counter()
            _parse_response(resp.json(), row, (t1 - t0) * 1000)

            # Confirmation turn for tool-expected turns
            if row.tool_expected and not row.tool_called and _CONFIRM_RE.search(row.response_text):
                row.confirmed_turn = True
                t2 = time.perf_counter()
                cr = await client.post(f"{UI_BASE}/api/sessions/{sid}/turns",
                                       json={"text": "Yes, go ahead please."},
                                       headers=hdrs, timeout=120)
                t3 = time.perf_counter()
                if cr.status_code == 200:
                    cd = cr.json()
                    ct = (cd.get("debug") or {}).get("timing") or {}
                    row.tool_ms      += ct.get("tool_ms", 0)
                    row.responder_ms += ct.get("responder_ms", 0)
                    row.wall_ms      += ct.get("wall_ms", 0) or ((t3 - t2) * 1000)
                    ex2 = cd.get("executed_tools") or []
                    if isinstance(ex2, list) and ex2:
                        n2 = [(t.get("tool") or t.get("name") or str(t)) if isinstance(t, dict)
                              else str(t) for t in ex2]
                        prev = [x for x in row.executed_tools.split(", ") if x]
                        row.executed_tools = ", ".join(dict.fromkeys(prev + n2))
                        row.tool_called = True
                    row.response_text = cd.get("response_text", row.response_text)
                    row.tool_match = (row.tool_called == row.tool_expected)

            req = str(tr_row.get("required_facts") or "")
            forb = str(tr_row.get("forbidden_facts") or "")
            if req or forb:
                v = judge(utterance, row.response_text, req, forb)
                row.accuracy_score = float(v.get("score", 0) or 0)
                row.facts_hit      = "; ".join(v.get("facts_hit") or [])
                row.forbidden_hit  = "; ".join(v.get("forbidden_hit") or [])
            else:
                row.accuracy_score = -1  # no facts to judge

            results.append(row)

        await client.post(f"{BACKEND_BASE}/sessions/{sid}/end",
                          json={"analyse": "skip"}, headers=hdrs, timeout=10)

    except Exception as e:
        err = str(e)[:150]
        if results:
            results[-1].error = err
        else:
            results.append(EvalRow(test_type="thread", thread_id=thread_id,
                                   archetype=archetype, direction=direction,
                                   tool_category=tool_category, error=err))

    return results


# ── Stats helpers ─────────────────────────────────────────────────────────────
def _pct(val, n): return round(val / n * 100, 1) if n else 0

def _stat(vals: list[float]):
    s = [v for v in vals if v >= 0]
    if not s: return 0.0, 0.0, 0.0
    srt = sorted(s)
    return (round(sum(srt) / len(srt), 1),
            round(statistics.median(srt), 1),
            round(srt[max(0, int(len(srt) * 0.95) - (0 if len(srt) <= 20 else 1))], 1))

def _gs(rows, attr):
    return _stat([getattr(r, attr) for r in rows if not r.error])


# ── Excel styling ─────────────────────────────────────────────────────────────
COLS = [
    "test_type", "thread_id", "archetype", "turn_no",
    "row_id", "lang", "difficulty", "utterance",
    "direction", "tool_category", "expected_tools", "tool_expected",
    "planner_ms", "retrieval_ms", "tool_ms", "responder_ms", "wall_ms",
    "tool_called", "tool_match", "executed_tools", "confirmed_turn",
    "accuracy_score", "facts_hit", "forbidden_hit",
    "response_text", "error", "timestamp",
]
COL_W = {
    "test_type":14, "thread_id":10, "archetype":22, "turn_no":7,
    "row_id":8, "lang":5, "difficulty":9, "utterance":42,
    "direction":10, "tool_category":13, "expected_tools":28, "tool_expected":12,
    "planner_ms":11, "retrieval_ms":12, "tool_ms":10, "responder_ms":12, "wall_ms":10,
    "tool_called":10, "tool_match":10, "executed_tools":30, "confirmed_turn":13,
    "accuracy_score":13, "facts_hit":38, "forbidden_hit":28,
    "response_text":50, "error":28, "timestamp":20,
}

def _brd():
    s = Side(style="thin", color="D0D0D0")
    return Border(left=s, right=s, top=s, bottom=s)

def _style_ws(ws, df: pd.DataFrame, hdr_color: str):
    b = _brd()
    for ci, col in enumerate(df.columns, 1):
        c = ws.cell(row=1, column=ci, value=col)
        c.font = Font(bold=True, color="FFFFFF", size=10)
        c.fill = PatternFill("solid", fgColor=hdr_color)
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        c.border = b
    for ri, row_vals in enumerate(df.itertuples(index=False), 2):
        for ci, v in enumerate(row_vals, 1):
            cell = ws.cell(row=ri, column=ci, value=v)
            cell.border = b
            cell.font = Font(size=9)
            col = df.columns[ci - 1]
            if col == "accuracy_score" and isinstance(v, (int, float)) and v >= 0:
                cell.fill = PatternFill("solid", fgColor=(
                    "C6EFCE" if v >= 70 else "FFEB9C" if v >= 40 else "FFC7CE"))
            if col == "wall_ms" and isinstance(v, (int, float)):
                cell.fill = PatternFill("solid", fgColor=(
                    "FFC7CE" if v > 4000 else "FFEB9C" if v > 2000 else "EAF6FF" if v > 1200 else "C6EFCE"))
            if col == "tool_ms" and isinstance(v, (int, float)) and v > 0:
                cell.fill = PatternFill("solid", fgColor="DDEEFF")
            if col == "error" and v:
                cell.fill = PatternFill("solid", fgColor="FFC7CE")
            if col == "tool_called":
                cell.fill = PatternFill("solid", fgColor=("EAF4FF" if v else "F5F5F5"))
            if col == "tool_match":
                cell.fill = PatternFill("solid", fgColor=("C6EFCE" if v else "FFC7CE"))
            if col == "confirmed_turn":
                cell.fill = PatternFill("solid", fgColor=("FFF3CD" if v else "F5F5F5"))
    for ci, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(ci)].width = COL_W.get(col, 14)
    ws.freeze_panes = "A2"
    if len(df) > 0:
        ws.auto_filter.ref = ws.dimensions


def build_summary(all_rows: list[EvalRow]) -> pd.DataFrame:
    records = []
    ok = [r for r in all_rows if not r.error]
    for tt in ("single_turn", "thread", "ALL"):
        g0 = ok if tt == "ALL" else [r for r in ok if r.test_type == tt]
        if not g0:
            continue
        for cat in ("with_tool", "without_tool", "ALL"):
            g1 = g0 if cat == "ALL" else [r for r in g0 if r.tool_category == cat]
            if not g1:
                continue
            for direction in ("inbound", "outbound", "ALL"):
                sub = g1 if direction == "ALL" else [r for r in g1 if r.direction == direction]
                if not sub:
                    continue
                scored = [r for r in sub if r.accuracy_score >= 0]
                wm, wmed, wp95 = _gs(sub, "wall_ms")
                pm, _, _       = _gs(sub, "planner_ms")
                rm, _, _       = _gs(sub, "retrieval_ms")
                tm, _, _       = _gs(sub, "tool_ms")
                rsm, _, _      = _gs(sub, "responder_ms")
                am, amed, _    = _stat([r.accuracy_score for r in scored]) if scored else (0,0,0)
                n_tool_exp  = sum(1 for r in sub if r.tool_expected)
                n_tool_fire = sum(1 for r in sub if r.tool_called)
                n_confirm   = sum(1 for r in sub if r.confirmed_turn)
                n_match     = sum(1 for r in sub if r.tool_match)
                records.append({
                    "test_type": tt, "tool_category": cat, "direction": direction,
                    "n_turns": len(sub),
                    "n_tool_expected": n_tool_exp,
                    "n_tool_fired": n_tool_fire,
                    "tool_fire_rate_%": _pct(n_tool_fire, n_tool_exp) if n_tool_exp else "—",
                    "n_confirm_turns": n_confirm,
                    "n_tool_match": n_match,
                    "wall_mean_ms": wm, "wall_median_ms": wmed, "wall_p95_ms": wp95,
                    "planner_mean_ms": pm,
                    "retrieval_mean_ms": rm,
                    "tool_mean_ms": tm,
                    "responder_mean_ms": rsm,
                    "n_scored": len(scored),
                    "accuracy_mean": am, "accuracy_median": amed,
                    "n_errors": sum(1 for r in all_rows if r.error
                                    and (tt == "ALL" or r.test_type == tt)
                                    and (cat == "ALL" or r.tool_category == cat)
                                    and (direction == "ALL" or r.direction == direction)),
                })
    return pd.DataFrame(records)


def write_excel(all_rows: list[EvalRow], out: Path) -> pd.DataFrame:
    full_df = pd.DataFrame(
        [{c: getattr(r, c, "") for c in COLS} for r in all_rows], columns=COLS)

    sum_df = build_summary(all_rows)

    # Filtered sub-views
    st_df    = full_df[full_df["test_type"] == "single_turn"].reset_index(drop=True)
    th_df    = full_df[full_df["test_type"] == "thread"].reset_index(drop=True)
    wt_df    = full_df[full_df["tool_category"] == "with_tool"].reset_index(drop=True)
    wot_df   = full_df[full_df["tool_category"] == "without_tool"].reset_index(drop=True)
    ib_df    = full_df[full_df["direction"] == "inbound"].reset_index(drop=True)
    ob_df    = full_df[full_df["direction"] == "outbound"].reset_index(drop=True)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Summary sheet
    ws = wb.create_sheet("Summary")
    for ri, row in enumerate([sum_df.columns.tolist()] + sum_df.values.tolist(), 1):
        for ci, v in enumerate(row, 1):
            c = ws.cell(row=ri, column=ci, value=v)
            if ri == 1:
                c.font = Font(bold=True, color="FFFFFF", size=10)
                c.fill = PatternFill("solid", fgColor="0F3057")
                c.alignment = Alignment(horizontal="center", wrap_text=True)
            c.border = _brd()
            if ri > 1 and isinstance(v, (int, float)):
                col = sum_df.columns[ci - 1]
                if "accuracy" in col and v >= 0:
                    c.fill = PatternFill("solid", fgColor=(
                        "C6EFCE" if v >= 70 else "FFEB9C" if v >= 40 else "FFC7CE"))
    ws.freeze_panes = "A2"
    for ci, col in enumerate(sum_df.columns, 1):
        ws.column_dimensions[get_column_letter(ci)].width = max(14, len(col) + 2)

    _style_ws(wb.create_sheet("SingleTurn_All"),   st_df,  "1F4E79")
    _style_ws(wb.create_sheet("Thread_All"),        th_df,  "375623")
    _style_ws(wb.create_sheet("WithTool_Rows"),     wt_df,  "833C00")
    _style_ws(wb.create_sheet("WithoutTool_Rows"),  wot_df, "404040")
    _style_ws(wb.create_sheet("Inbound"),           ib_df,  "4472C4")
    _style_ws(wb.create_sheet("Outbound"),          ob_df,  "7030A0")

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"\n[excel] -> {out}")
    return sum_df


# ── Console summary ────────────────────────────────────────────────────────────
def print_summary(all_rows: list[EvalRow]):
    ok = [r for r in all_rows if not r.error]
    print("\n" + "=" * 72)
    print("GOLDEN EVAL FINAL — FULL SUMMARY")
    print("=" * 72)
    for tt in ("single_turn", "thread"):
        g0 = [r for r in ok if r.test_type == tt]
        if not g0:
            continue
        print(f"\n{'─' * 72}")
        print(f"  TEST TYPE: {tt.upper().replace('_', ' ')}  (n={len(g0)})")
        print(f"{'─' * 72}")
        for cat in ("with_tool", "without_tool"):
            g1 = [r for r in g0 if r.tool_category == cat]
            if not g1:
                continue
            print(f"\n  ── {cat.upper().replace('_', ' ')} (n={len(g1)}) ──")
            for direction in ("inbound", "outbound"):
                sub = [r for r in g1 if r.direction == direction]
                if not sub:
                    continue
                wm, wmed, wp95 = _stat([r.wall_ms for r in sub])
                pm, _, _       = _stat([r.planner_ms for r in sub])
                rm, _, _       = _stat([r.retrieval_ms for r in sub])
                tm, tmed, _    = _stat([r.tool_ms for r in sub])
                rsm, _, _      = _stat([r.responder_ms for r in sub])
                scored = [r for r in sub if r.accuracy_score >= 0]
                am, amed, _    = _stat([r.accuracy_score for r in scored]) if scored else (0, 0, 0)
                n_tf   = sum(1 for r in sub if r.tool_called)
                n_texp = sum(1 for r in sub if r.tool_expected)
                n_conf = sum(1 for r in sub if r.confirmed_turn)
                n_err  = sum(1 for r in sub if r.error)
                print(f"\n    {direction.upper()} via UI  n={len(sub)}"
                      f"  tools_expected={n_texp}  tools_fired={n_tf}"
                      f"  confirm_turns={n_conf}  errors={n_err}")
                print(f"      Wall      mean={wm:.0f}  median={wmed:.0f}  p95={wp95:.0f} ms")
                print(f"      Planner   mean={pm:.0f} ms")
                print(f"      Retrieval mean={rm:.0f} ms")
                print(f"      Tool      mean={tm:.0f}  median={tmed:.0f} ms")
                print(f"      Responder mean={rsm:.0f} ms")
                if scored:
                    print(f"      Accuracy  mean={am:.1f}  median={amed:.1f}/100  (n_scored={len(scored)})")
                else:
                    print(f"      Accuracy  — no judged turns")


# ── Token keeper (auto-refreshes before 25-min mark) ──────────────────────────
class TokenKeeper:
    def __init__(self, token: str):
        self._token = token
        self._issued = time.monotonic()

    @property
    def token(self) -> str:
        return self._token

    async def fresh(self) -> str:
        if time.monotonic() - self._issued > 1400:   # 23 min — refresh before 30-min TTL
            self._token = await get_token()
            self._issued = time.monotonic()
            print(f"[auth] token auto-refreshed")
        return self._token


# ── JSONL checkpoint (intermediate save so crash = partial results kept) ───────
def _append_jsonl(path: Path, row: EvalRow):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({c: getattr(row, c, "") for c in COLS}, ensure_ascii=False) + "\n")


# ── Concurrent single-turn wrapper with semaphore ──────────────────────────────
async def _st_task(sem: asyncio.Semaphore, client: httpx.AsyncClient,
                   row: dict, direction: str, keeper: TokenKeeper,
                   cids: list[str], idx: int,
                   total: int, jsonl: Path) -> EvalRow:
    async with sem:
        token = await keeper.fresh()
        res = await run_single_turn(client, row, direction, token, cids, idx)
        _append_jsonl(jsonl, res)
        cat = row.get("_tool_category", "")
        lang = row.get("lang", "?")
        print(
            f"  [{direction.upper()} ST {idx+1}/{total}] [{lang}][{cat}] "
            f"wall={res.wall_ms:.0f}ms acc={res.accuracy_score:.0f}"
            f"{' +CONFIRM' if res.confirmed_turn else ''}"
            f"{' TOOL=' + res.executed_tools[:20] if res.tool_called else ''}"
            f"{' ERR:' + res.error[:40] if res.error else ''}"
        )
        return res


# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set"); return

    # Connectivity checks
    async with httpx.AsyncClient(timeout=8) as c:
        try:
            r = await c.get(f"{BACKEND_BASE}/health")
            print(f"[check] backend: {r.status_code}")
        except Exception as e:
            print(f"[check] backend UNREACHABLE: {e}"); return
        try:
            r = await c.get(UI_BASE)
            print(f"[check] UI proxy: {r.status_code}")
        except Exception as e:
            print(f"[check] UI proxy UNREACHABLE: {e}"); return

    token = await get_token()
    print(f"[auth] OK — {token[:28]}...")
    keeper = TokenKeeper(token)
    customers = await get_all_customers(token)

    # ── Load golden data (ALL languages) ──────────────────────────────────────
    tmp = Path(r"C:\Users\ADVORA-ALWYN\AppData\Local\Temp\golden_eval_full_run.xlsx")
    try:
        shutil.copy2(GOLDEN_XLSX, tmp)
        print(f"[golden] copied -> {tmp}")
    except PermissionError:
        fallback = Path(r"C:\Users\ADVORA-ALWYN\AppData\Local\Temp\golden_inspect.xlsx")
        if fallback.exists():
            tmp = fallback
            print(f"[golden] using cached copy: {tmp}")
        else:
            print("[golden] ERROR: cannot read golden dataset"); return

    # Single-turn sheet — ALL 450 rows (EN + HI + Hinglish)
    st_df = pd.read_excel(tmp, sheet_name="Golden_SingleTurn")
    for col in ["required_facts", "forbidden_facts", "answer_constraints",
                "difficulty", "expected_tool_calls"]:
        st_df[col] = st_df[col].fillna("").astype(str)
    st_df["_tools"]         = st_df["expected_tool_calls"].apply(parse_tools)
    st_df["_has_tool"]      = st_df["_tools"].apply(len) > 0
    st_df["_tool_category"] = st_df["_has_tool"].map({True: "with_tool", False: "without_tool"})
    st_df["_expected_tools"] = st_df["_tools"].apply(lambda t: ", ".join(t) if t else "")
    st_rows = st_df.to_dict("records")
    print(f"[golden] single-turn: {len(st_rows)} rows "
          f"({st_df['_has_tool'].sum()} with_tool / {(~st_df['_has_tool']).sum()} without_tool) "
          f"langs={dict(st_df['lang'].value_counts().to_dict())}")

    # Thread sheet — ALL 90 threads (EN + HI + Hinglish)
    th_df = pd.read_excel(tmp, sheet_name="Golden_Threads")
    for col in ["required_facts", "forbidden_facts", "expected_tool_calls"]:
        th_df[col] = th_df[col].fillna("").astype(str)
    th_df["_tools"]    = th_df["expected_tool_calls"].apply(parse_tools)
    th_df["_has_tool"] = th_df["_tools"].apply(len) > 0

    thread_has_tool   = th_df.groupby("thread_id")["_has_tool"].any()
    tool_thread_ids   = list(thread_has_tool[thread_has_tool].index)    # all 84
    notool_thread_ids = list(thread_has_tool[~thread_has_tool].index)   # all 6

    def get_thread_rows(tid):
        return th_df[th_df["thread_id"] == tid].sort_values("turn_no").to_dict("records")

    tool_threads   = [(tid, get_thread_rows(tid), "with_tool")   for tid in tool_thread_ids]
    notool_threads = [(tid, get_thread_rows(tid), "without_tool") for tid in notool_thread_ids]
    all_threads = tool_threads + notool_threads

    total_th_turns = sum(len(rows) for _, rows, _ in all_threads)
    print(f"[golden] threads: {len(all_threads)} threads "
          f"({len(tool_threads)} with_tool / {len(notool_threads)} without_tool) "
          f"turns={total_th_turns}")
    print(f"[golden] TOTAL TURNS to run: "
          f"{len(st_rows)*2} ST + {total_th_turns*2} TH = "
          f"{len(st_rows)*2 + total_th_turns*2}")

    # JSONL checkpoint file (append mode, survives restarts)
    jsonl_path = RESULTS_DIR / "golden_eval_full_rows.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate for a fresh run
    jsonl_path.write_text("", encoding="utf-8")

    all_results: list[EvalRow] = []

    # ── Single-turn eval (concurrent, semaphore=3) ────────────────────────────
    ST_CONCURRENCY = 3
    sem = asyncio.Semaphore(ST_CONCURRENCY)

    for direction in ("inbound", "outbound"):
        cids = customers if direction == "outbound" else []
        print("\n" + "=" * 72)
        print(f"SINGLE-TURN {direction.upper()} ({len(st_rows)} rows, concurrency={ST_CONCURRENCY})")
        print("=" * 72)
        async with httpx.AsyncClient(timeout=180) as client:
            tasks = [
                _st_task(sem, client, row, direction, keeper, cids, i, len(st_rows), jsonl_path)
                for i, row in enumerate(st_rows)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, EvalRow):
                all_results.append(r)
            else:
                print(f"  [ST gather error] {r}")

        st_phase = [r for r in all_results if r.test_type == "single_turn" and r.direction == direction]
        fired = sum(1 for r in st_phase if r.tool_called)
        exp   = sum(1 for r in st_phase if r.tool_expected)
        errs  = sum(1 for r in st_phase if r.error)
        scored = [r for r in st_phase if r.accuracy_score >= 0]
        acc_mean = sum(r.accuracy_score for r in scored) / len(scored) if scored else 0
        print(f"\n  [{direction.upper()} ST done] tool_fired={fired}/{exp}  "
              f"errors={errs}  acc_mean={acc_mean:.1f}  n={len(st_phase)}")

    # ── Refresh token before thread phase ─────────────────────────────────────
    keeper._token = await get_token()
    keeper._issued = time.monotonic()
    print("[auth] token refreshed before thread phase")

    # ── Thread eval (sequential, one at a time) ───────────────────────────────
    for direction in ("inbound", "outbound"):
        cids = customers if direction == "outbound" else []
        print("\n" + "=" * 72)
        print(f"THREAD {direction.upper()} ({len(all_threads)} threads)")
        print("=" * 72)
        async with httpx.AsyncClient(timeout=300) as client:
            for ti, (tid, t_rows, t_cat) in enumerate(all_threads, 1):
                token = await keeper.fresh()
                n_tool_turns = sum(1 for r in t_rows if parse_tools(r.get("expected_tool_calls", "")))
                lang_tag = str(t_rows[0].get("lang", "?")) if t_rows else "?"
                print(f"\n  [{direction.upper()} TH {ti}/{len(all_threads)}] "
                      f"{tid} [{lang_tag}][{t_cat}] {len(t_rows)} turns "
                      f"({n_tool_turns} tool-expected)")
                turn_results = await run_thread(
                    client, t_rows, direction, token, t_cat, cids, ti - 1)
                for tr in turn_results:
                    all_results.append(tr)
                    _append_jsonl(jsonl_path, tr)
                    print(
                        f"    T{tr.turn_no}  plan={tr.planner_ms:.0f}  "
                        f"ret={tr.retrieval_ms:.0f}  tool={tr.tool_ms:.0f}  "
                        f"resp={tr.responder_ms:.0f}  wall={tr.wall_ms:.0f}ms  "
                        f"acc={tr.accuracy_score:.0f}  "
                        f"exp={tr.tool_expected}  fired={tr.tool_called}"
                        f"{' TOOL=' + tr.executed_tools[:20] if tr.tool_called else ''}"
                        f"{' ERR:' + tr.error[:40] if tr.error else ''}"
                    )

    print_summary(all_results)

    # ── Write Excel ───────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "golden_eval_final.xlsx"
    try:
        if out_path.exists():
            try:
                out_path.unlink()
            except PermissionError:
                ts = datetime.utcnow().strftime("%H%M%S")
                out_path = RESULTS_DIR / f"golden_eval_final_{ts}.xlsx"
                print(f"[excel] original locked, writing to: {out_path}")
        write_excel(all_results, out_path)
    except PermissionError:
        ts = datetime.utcnow().strftime("%H%M%S")
        out_path = RESULTS_DIR / f"golden_eval_final_{ts}.xlsx"
        print(f"[excel] permission denied, fallback: {out_path}")
        write_excel(all_results, out_path)
    print(f"\nDone. Report: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
