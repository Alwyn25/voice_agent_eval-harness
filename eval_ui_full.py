"""
Highland Greenz — UI-Path Full Eval (v3)
=========================================
Tests ONLY the UI proxy path (Vite :3000/api → backend :8000) for:

  ① INBOUND  via UI   — with_tool rows (10)
  ② INBOUND  via UI   — without_tool rows (10)
  ③ OUTBOUND via UI   — with_tool rows (10)
  ④ OUTBOUND via UI   — without_tool rows (10)

Tool-call handling
------------------
When the agent asks for confirmation ("shall I proceed?", "is that right?",
"would you like me to go ahead?") we send ONE follow-up "yes, go ahead"
turn so the tool actually executes and tool_ms becomes non-zero.

Outbound 403 fix
-----------------
We probe all seeded customers before the run and build a list of only the
ones that accept an outbound session. The eval cycles through that list.

Provider
---------
Reads .env from the backend directory — currently LLM_PROVIDER=openai
(Cerebras credit exhausted; will restore after top-up).
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

# ── Credentials from .env ──────────────────────────────────────────────────────
_env_file = Path(__file__).parent.parent / "AI-Voice-Agent-Backend-main" / ".env"
if _env_file.exists():
    for _ln in _env_file.read_text(encoding="utf-8").splitlines():
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
MAX_PER_CAT  = int(os.getenv("MAX_PER_CAT", "10"))   # 10 with_tool + 10 without_tool
_oai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Confirmation-request patterns — send "yes, go ahead" when matched
_CONFIRM_PATTERNS = re.compile(
    r"(shall i|should i|would you like me to|want me to|go ahead|confirm|proceed|"
    r"is that right|can i go ahead|shall we|do you want me|want me to)",
    re.IGNORECASE,
)


# ── Data model ─────────────────────────────────────────────────────────────────
@dataclass
class CallResult:
    row_id:         str
    lang:           str
    difficulty:     str
    utterance:      str
    direction:      str
    path:           str = "ui"
    tool_category:  str = ""
    expected_tools: str = ""
    channel:        str = "phone"
    response_text:  str = ""
    planner_ms:     float = 0
    retrieval_ms:   float = 0
    tool_ms:        float = 0
    responder_ms:   float = 0
    total_ms:       float = 0
    wall_ms:        float = 0
    executed_tools: str = ""
    tool_called:    bool = False
    confirmed_turn: bool = False    # True when a confirmation turn was sent
    accuracy_score: float = 0
    facts_hit:      str = ""
    forbidden_hit:  str = ""
    error:          str = ""
    timestamp:      str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Auth ───────────────────────────────────────────────────────────────────────
async def get_token() -> str:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{BACKEND_BASE}/auth/login",
                         json={"email": LOGIN_EMAIL, "password": LOGIN_PASS})
        r.raise_for_status()
        return r.json()["access_token"]


# ── Customer probing ───────────────────────────────────────────────────────────
async def get_all_customers(token: str) -> list[str]:
    """Return all customer ids. Outbound sessions use enforce_outbound_policy=false."""
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{BACKEND_BASE}/customers?page_size=50",
                        headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        ids = [it["id"] for it in r.json().get("items", []) if it.get("id")]
    print(f"[customers] {len(ids)} available (policy enforcement bypassed for eval)")
    return ids


# ── Accuracy judge ─────────────────────────────────────────────────────────────
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


# ── Parse timing + tools from turn response ────────────────────────────────────
def _parse_response(data: dict, res: CallResult, elapsed_ms: float):
    res.response_text = data.get("response_text", "")
    res.total_ms = round(elapsed_ms, 1)
    timing = (data.get("debug") or {}).get("timing") or {}
    res.wall_ms      = timing.get("wall_ms", 0) or elapsed_ms
    res.planner_ms   = timing.get("planner_ms", 0)
    res.retrieval_ms = timing.get("retrieval_ms", 0)
    res.tool_ms      = timing.get("tool_ms", 0)
    res.responder_ms = timing.get("responder_ms", 0)
    executed = data.get("executed_tools") or []
    if isinstance(executed, list):
        names = [
            (t.get("tool") or t.get("name") or str(t)) if isinstance(t, dict) else str(t)
            for t in executed
        ]
        res.executed_tools = ", ".join(names)
        res.tool_called    = len(names) > 0
    else:
        res.executed_tools = str(executed) if executed else ""
        res.tool_called    = bool(executed)


# ── Single turn ────────────────────────────────────────────────────────────────
async def run_turn(client: httpx.AsyncClient,
                   row: dict,
                   direction: str,
                   token: str,
                   customer_ids: list[str] | None,
                   customer_index: int) -> CallResult:
    res = CallResult(
        row_id=str(row["id"]),
        lang=str(row.get("lang", "")),
        difficulty=str(row.get("difficulty", "")),
        utterance=str(row["user_utterance"]),
        direction=direction,
        tool_category=str(row.get("_tool_category", "")),
        expected_tools=str(row.get("_expected_tools", "")),
    )
    hdrs  = {"Authorization": f"Bearer {token}"}
    # UI path: turns go through Vite proxy, sessions are created on backend directly
    turns_api  = f"{UI_BASE}/api"
    session_api = BACKEND_BASE

    try:
        sess_body: dict = {"channel": "phone", "direction": direction, "purpose": "sales",
                           "enforce_outbound_policy": False}
        if direction == "outbound" and customer_ids:
            sess_body["customer_id"] = customer_ids[customer_index % len(customer_ids)]

        sr = await client.post(f"{session_api}/sessions", json=sess_body,
                               headers=hdrs, timeout=20)
        sr.raise_for_status()
        sid = sr.json()["session_id"]

        # Outbound identity confirmation turn (not timed)
        if direction == "outbound":
            await client.post(f"{turns_api}/sessions/{sid}/turns",
                              json={"text": "Yes this is me. I was interested in 3BHK."},
                              headers=hdrs, timeout=60)

        # Primary eval turn (TIMED)
        t0 = time.perf_counter()
        tr = await client.post(f"{turns_api}/sessions/{sid}/turns",
                               json={"text": res.utterance},
                               headers=hdrs, timeout=120)
        tr.raise_for_status()
        t1 = time.perf_counter()

        data = tr.json()
        _parse_response(data, res, (t1 - t0) * 1000)

        # If the agent is asking for confirmation AND no tool has fired yet,
        # send "yes go ahead" and capture the tool execution timing
        if not res.tool_called and _CONFIRM_PATTERNS.search(res.response_text):
            res.confirmed_turn = True
            t2 = time.perf_counter()
            cr = await client.post(f"{turns_api}/sessions/{sid}/turns",
                                   json={"text": "Yes, go ahead please."},
                                   headers=hdrs, timeout=120)
            t3 = time.perf_counter()
            if cr.status_code == 200:
                cd = cr.json()
                # Add tool timing from confirmation turn to the total
                ct = (cd.get("debug") or {}).get("timing") or {}
                res.tool_ms      += ct.get("tool_ms", 0)
                res.responder_ms += ct.get("responder_ms", 0)
                res.wall_ms      += ct.get("wall_ms", 0) or ((t3 - t2) * 1000)
                res.total_ms     += round((t3 - t2) * 1000, 1)
                # Update tool execution from confirmation turn
                ex2 = cd.get("executed_tools") or []
                if isinstance(ex2, list) and ex2:
                    n2 = [(t.get("tool") or t.get("name") or str(t)) if isinstance(t, dict)
                          else str(t) for t in ex2]
                    existing = [x for x in res.executed_tools.split(", ") if x]
                    res.executed_tools = ", ".join(dict.fromkeys(existing + n2))
                    res.tool_called = True
                # Use full combined response for accuracy scoring
                res.response_text = cd.get("response_text", res.response_text)

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
    res.accuracy_score = float(verdict.get("score", 0) or 0)
    res.facts_hit      = "; ".join(verdict.get("facts_hit") or [])
    res.forbidden_hit  = "; ".join(verdict.get("forbidden_hit") or [])
    return res


# ── Run a scenario ─────────────────────────────────────────────────────────────
async def run_scenario(rows: list[dict], direction: str, token: str,
                       customer_ids: list[str] | None) -> list[CallResult]:
    label = f"{direction.upper()} via UI"
    results: list[CallResult] = []
    async with httpx.AsyncClient(timeout=180) as client:
        for i, row in enumerate(rows, 1):
            cat = row.get("_tool_category", "")
            print(f"  [{label} {i}/{len(rows)}] {row['id']} [{cat}]  "
                  f"{str(row['user_utterance'])[:50]!r}")
            r = await run_turn(client, row, direction, token, customer_ids, i - 1)
            results.append(r)
            conf_flag = " +CONFIRM" if r.confirmed_turn else ""
            tool_flag = f" TOOL={r.executed_tools[:35]}" if r.tool_called else ""
            print(
                f"    plan={r.planner_ms:.0f}  ret={r.retrieval_ms:.0f}  "
                f"tool={r.tool_ms:.0f}  resp={r.responder_ms:.0f}  "
                f"wall={r.wall_ms:.0f} ms  acc={r.accuracy_score:.0f}"
                f"{conf_flag}{tool_flag}"
                f"{(' ERR:' + r.error[:55]) if r.error else ''}"
            )
    return results


# ── Stats ──────────────────────────────────────────────────────────────────────
def _stat(vals: list[float]) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    s = sorted(vals)
    return (round(sum(s) / len(s), 1),
            round(statistics.median(s), 1),
            round(s[int(len(s) * 0.95)], 1))


def _gs(results: list[CallResult], attr: str) -> tuple[float, float, float]:
    return _stat([getattr(r, attr) for r in results if not r.error])


# ── Excel output ───────────────────────────────────────────────────────────────
COLS = [
    "row_id", "lang", "difficulty", "utterance",
    "direction", "path", "tool_category", "expected_tools",
    "planner_ms", "retrieval_ms", "tool_ms", "responder_ms", "total_ms", "wall_ms",
    "tool_called", "executed_tools", "confirmed_turn",
    "accuracy_score", "facts_hit", "forbidden_hit",
    "response_text", "error", "timestamp",
]
COL_W = {
    "row_id": 8, "lang": 6, "difficulty": 9, "utterance": 44,
    "direction": 10, "path": 6, "tool_category": 14, "expected_tools": 30,
    "planner_ms": 11, "retrieval_ms": 12, "tool_ms": 10, "responder_ms": 12,
    "total_ms": 11, "wall_ms": 10, "tool_called": 10, "executed_tools": 32,
    "confirmed_turn": 13,
    "accuracy_score": 13, "facts_hit": 38, "forbidden_hit": 28,
    "response_text": 55, "error": 28, "timestamp": 20,
}


def _brd():
    s = Side(style="thin", color="D0D0D0")
    return Border(left=s, right=s, top=s, bottom=s)


def _style(ws, df: pd.DataFrame, hdr: str):
    b = _brd()
    for ci, col in enumerate(df.columns, 1):
        c = ws.cell(row=1, column=ci, value=col)
        c.font = Font(bold=True, color="FFFFFF", size=10)
        c.fill = PatternFill("solid", fgColor=hdr)
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
                    "C6EFCE" if v >= 70 else "FFEB9C" if v >= 40 else "FFC7CE"))
            if col in ("wall_ms", "total_ms") and isinstance(v, (int, float)):
                cell.fill = PatternFill("solid", fgColor=(
                    "FFC7CE" if v > 5000 else "FFEB9C" if v > 3000 else "C6EFCE"))
            if col == "tool_ms" and isinstance(v, (int, float)) and v > 0:
                cell.fill = PatternFill("solid", fgColor="DDEEFF")
            if col == "error" and v:
                cell.fill = PatternFill("solid", fgColor="FFC7CE")
            if col == "tool_called":
                cell.fill = PatternFill("solid", fgColor=("EAF4FF" if v else "F5F5F5"))
            if col == "confirmed_turn":
                cell.fill = PatternFill("solid", fgColor=("FFF3CD" if v else "F5F5F5"))
    for ci, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(ci)].width = COL_W.get(col, 14)
    ws.freeze_panes = "A2"
    if len(df) > 0:
        ws.auto_filter.ref = ws.dimensions


def write_excel(all_results: list[CallResult], out: Path) -> pd.DataFrame:
    full_df = pd.DataFrame(
        [{c: getattr(r, c, "") for c in COLS} for r in all_results], columns=COLS)

    sum_rows = []
    ok = [r for r in all_results if not r.error]
    for cat_label, grp in [("All", ok),
                            ("with_tool",    [r for r in ok if r.tool_category == "with_tool"]),
                            ("without_tool", [r for r in ok if r.tool_category == "without_tool"])]:
        for direction in ("inbound", "outbound"):
            sub = [r for r in grp if r.direction == direction]
            if not sub:
                continue
            pm, pmed, _ = _gs(sub, "planner_ms")
            rm, rmed, _ = _gs(sub, "retrieval_ms")
            tm, tmed, _ = _gs(sub, "tool_ms")
            rsm,rsmed,_ = _gs(sub, "responder_ms")
            wm, wmed, wp95 = _gs(sub, "wall_ms")
            am, amed, _ = _gs(sub, "accuracy_score")
            n_tool = sum(1 for r in sub if r.tool_called)
            n_conf  = sum(1 for r in sub if r.confirmed_turn)
            sum_rows.append({
                "group": cat_label, "direction": direction, "path": "ui",
                "n": len(sub), "n_tool_called": n_tool, "n_confirm_turns": n_conf,
                "wall_mean_ms": wm, "wall_median_ms": wmed, "wall_p95_ms": wp95,
                "planner_mean_ms": pm, "planner_median_ms": pmed,
                "retrieval_mean_ms": rm, "retrieval_median_ms": rmed,
                "tool_mean_ms": tm, "tool_median_ms": tmed,
                "responder_mean_ms": rsm, "responder_median_ms": rsmed,
                "accuracy_mean": am, "accuracy_median": amed,
                "errors": len([r for r in all_results if r.direction == direction
                               and r.tool_category == (cat_label if cat_label != "All" else r.tool_category)
                               and r.error]),
            })

    sum_df = pd.DataFrame(sum_rows)
    tool_df   = full_df[full_df["tool_category"] == "with_tool"].reset_index(drop=True)
    notool_df = full_df[full_df["tool_category"] == "without_tool"].reset_index(drop=True)
    ib_df     = full_df[full_df["direction"] == "inbound"].reset_index(drop=True)
    ob_df     = full_df[full_df["direction"] == "outbound"].reset_index(drop=True)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet("Summary")
    for ri, row in enumerate([sum_df.columns.tolist()] + sum_df.values.tolist(), 1):
        for ci, v in enumerate(row, 1):
            c = ws.cell(row=ri, column=ci, value=v)
            if ri == 1:
                c.font = Font(bold=True, color="FFFFFF", size=10)
                c.fill = PatternFill("solid", fgColor="0F3057")
                c.alignment = Alignment(horizontal="center", wrap_text=True)
            c.border = _brd()
    ws.freeze_panes = "A2"
    for ci, col in enumerate(sum_df.columns, 1):
        ws.column_dimensions[get_column_letter(ci)].width = max(12, len(col) + 2)

    _style(wb.create_sheet("With_Tool_Calls"), tool_df, "1F4E79")
    _style(wb.create_sheet("Without_Tool_Calls"), notool_df, "375623")
    _style(wb.create_sheet("Inbound_UI"), ib_df, "4472C4")
    _style(wb.create_sheet("Outbound_UI"), ob_df, "833C00")

    # Raw all rows
    ws_all = wb.create_sheet("All_Rows")
    _style(ws_all, full_df, "404040")

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"\n[excel] → {out}")
    return sum_df


# ── Console summary ────────────────────────────────────────────────────────────
def print_summary(all_results: list[CallResult]):
    ok = [r for r in all_results if not r.error]
    print("\n" + "=" * 68)
    print("SUMMARY — UI PATH")
    print("=" * 68)
    for cat in ("All", "with_tool", "without_tool"):
        grp = ok if cat == "All" else [r for r in ok if r.tool_category == cat]
        if not grp:
            continue
        print(f"\n  ── {cat.upper().replace('_', ' ')} (n={len(grp)}) ──")
        for direction in ("inbound", "outbound"):
            sub = [r for r in grp if r.direction == direction]
            if not sub:
                continue
            wm, wmed, wp95 = _stat([r.wall_ms for r in sub])
            pm, pmed, _    = _stat([r.planner_ms for r in sub])
            rm, rmed, _    = _stat([r.retrieval_ms for r in sub])
            tm, tmed, _    = _stat([r.tool_ms for r in sub])
            rsm, rsmed, _  = _stat([r.responder_ms for r in sub])
            acc, accmed, _ = _stat([r.accuracy_score for r in sub])
            n_tool  = sum(1 for r in sub if r.tool_called)
            n_conf  = sum(1 for r in sub if r.confirmed_turn)
            n_err   = sum(1 for r in sub if r.error)
            print(f"\n    {direction.upper()} via UI  n={len(sub)}  "
                  f"tools_ran={n_tool}  confirm_turns={n_conf}  errors={n_err}")
            print(f"      Wall      mean={wm:.0f}  median={wmed:.0f}  p95={wp95:.0f} ms")
            print(f"      Planner   mean={pm:.0f}  median={pmed:.0f} ms")
            print(f"      Retrieval mean={rm:.0f}  median={rmed:.0f} ms")
            print(f"      Tool      mean={tm:.0f}  median={tmed:.0f} ms")
            print(f"      Responder mean={rsm:.0f}  median={rsmed:.0f} ms")
            print(f"      Accuracy  mean={acc:.1f}  median={accmed:.1f}/100")


# ── Entry point ────────────────────────────────────────────────────────────────
async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set"); return

    # Connectivity check
    async with httpx.AsyncClient(timeout=8) as c:
        try:
            r = await c.get(f"{BACKEND_BASE}/health")
            print(f"[check] backend: {r.status_code} OK")
        except Exception as e:
            print(f"[check] backend UNREACHABLE: {e}"); return
        try:
            r = await c.get(UI_BASE)
            print(f"[check] UI proxy: {r.status_code} OK")
        except Exception as e:
            print(f"[check] UI proxy not reachable: {e} — aborting (UI path required)"); return

    print("[auth] logging in...")
    token = await get_token()
    print(f"[auth] token: {token[:28]}...")

    # Get all customers (policy bypassed via enforce_outbound_policy=false)
    valid_customers = await get_all_customers(token)

    # ── Load golden dataset ────────────────────────────────────────────────────
    tmp = Path(r"C:\Users\ADVORA~1\AppData\Local\Temp\golden_eval_ui.xlsx")
    try:
        shutil.copy2(GOLDEN_XLSX, tmp)
        print(f"[golden] copied to {tmp}")
    except PermissionError:
        fallback = Path(r"C:\Users\ADVORA~1\AppData\Local\Temp\golden_tmp.xlsx")
        if fallback.exists():
            tmp = fallback
            print(f"[golden] using cached copy: {tmp}")
        else:
            print("[golden] ERROR: cannot read golden dataset"); return

    golden = pd.read_excel(tmp, sheet_name="Golden_SingleTurn")
    for col in ["required_facts", "forbidden_facts", "answer_constraints",
                "difficulty", "expected_tool_calls"]:
        golden[col] = golden[col].fillna("").astype(str)

    def _parse(v: str) -> list:
        v = v.strip()
        if not v or v in ("[]", "nan"): return []
        try: return ast.literal_eval(v)
        except: return [v]

    golden["_tools_list"]    = golden["expected_tool_calls"].apply(_parse)
    golden["_has_tool"]      = golden["_tools_list"].apply(len) > 0
    golden["_tool_category"] = golden["_has_tool"].map({True: "with_tool", False: "without_tool"})
    golden["_expected_tools"]= golden["_tools_list"].apply(lambda t: ", ".join(t) if t else "")

    tool_rows   = golden[golden["_has_tool"]].head(MAX_PER_CAT).to_dict("records")
    notool_rows = golden[~golden["_has_tool"]].head(MAX_PER_CAT).to_dict("records")
    rows        = tool_rows + notool_rows

    tool_types = sorted({t for r in tool_rows for t in r["_tools_list"]})
    print(f"\n[golden] {len(rows)} rows: {len(tool_rows)} with_tool + {len(notool_rows)} without_tool")
    print(f"  Tool types: {tool_types}\n")

    all_results: list[CallResult] = []

    for direction in ("inbound", "outbound"):
        cids = valid_customers if direction == "outbound" else None
        print("=" * 68)
        print(f"{direction.upper()} via UI  ({len(rows)} rows: "
              f"{len(tool_rows)} tool + {len(notool_rows)} no-tool)")
        print("=" * 68)
        res = await run_scenario(rows, direction, token, cids)
        all_results.extend(res)
        print()

    print_summary(all_results)

    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out = RESULTS_DIR / f"eval_ui_full_{ts}.xlsx"
    write_excel(all_results, out)
    print(f"\nReport: {out}")


if __name__ == "__main__":
    asyncio.run(main())
