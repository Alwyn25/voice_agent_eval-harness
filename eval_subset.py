"""
eval_subset.py  —  Highland Greenz Golden Subset Eval
100 single-turn rows × 2 directions + 40 threads × 2 directions = 468 total calls.
All traffic goes directly to the backend at http://localhost:8000.

Usage:
    python eval_harness/eval_subset.py
"""
import asyncio, json, re, time, os
from dataclasses import dataclass
from pathlib import Path
import httpx
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
BASE      = "http://localhost:8000"
DATASET   = Path(__file__).parent / "golden_subset_100st_40th.xlsx"
OUT_DIR   = Path(__file__).parent / "results"
JSONL     = OUT_DIR / "subset_eval_rows.jsonl"
XLSX_OUT  = OUT_DIR / "subset_eval_results.xlsx"
OUT_DIR.mkdir(exist_ok=True)

EMAIL = "admin@advora.ai"
PASS  = "Admin@123"

# load .env so OPENAI_API_KEY is available for the judge
_env = Path(__file__).parent.parent / "AI-Voice-Agent-Backend-main" / ".env"
if _env.exists():
    for _ln in _env.read_text(encoding="utf-8").splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            k, _, v = _ln.partition("=")
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()

CONFIRM_RE = re.compile(
    r"(shall i|should i|would you like me to|want me to|go ahead|confirm|proceed|"
    r"is that right|can i go ahead|shall we|do you want me)",
    re.IGNORECASE,
)

# ── Auth ──────────────────────────────────────────────────────────────────────
class TokenKeeper:
    def __init__(self): self._token = ""; self._issued = 0.0

    async def fresh(self, client: httpx.AsyncClient) -> str:
        if time.monotonic() - self._issued > 1400:
            r = await client.post(f"{BASE}/auth/login",
                                  json={"email": EMAIL, "password": PASS}, timeout=15)
            r.raise_for_status()
            self._token = r.json()["access_token"]
            self._issued = time.monotonic()
            print("[auth] token refreshed")
        return self._token

# ── Session helpers ───────────────────────────────────────────────────────────
async def new_session(client: httpx.AsyncClient, token: str, direction: str,
                      customer_id: str | None = None) -> str:
    body: dict = {"channel": "phone", "direction": direction,
                  "purpose": "sales", "enforce_outbound_policy": False}
    if customer_id:
        body["customer_id"] = customer_id
    r = await client.post(f"{BASE}/sessions", json=body,
                          headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if r.status_code not in (200, 201):
        r.raise_for_status()
    return r.json()["session_id"]

async def send_turn(client: httpx.AsyncClient, token: str,
                    sid: str, text: str) -> tuple[dict, int]:
    t0 = time.perf_counter()
    r  = await client.post(f"{BASE}/sessions/{sid}/turns",
                           json={"text": text},
                           headers={"Authorization": f"Bearer {token}"},
                           timeout=120)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    r.raise_for_status()
    return r.json(), wall_ms

async def end_session(client: httpx.AsyncClient, token: str, sid: str):
    try:
        await client.post(f"{BASE}/sessions/{sid}/end",
                          json={"analyse": "skip"},
                          headers={"Authorization": f"Bearer {token}"}, timeout=10)
    except Exception:
        pass

# ── Response parser ───────────────────────────────────────────────────────────
def parse_response(data: dict, wall_ms: int) -> dict:
    debug  = data.get("debug") or {}
    timing = debug.get("timing") or {}
    executed = data.get("executed_tools") or []
    if isinstance(executed, list):
        tool_names = [(t.get("tool") or t.get("name") or str(t)) if isinstance(t, dict) else str(t)
                      for t in executed]
    else:
        tool_names = [str(executed)] if executed else []
    return {
        "response_text":  data.get("response_text", ""),
        "wall_ms":        timing.get("wall_ms", 0) or wall_ms,
        "planner_ms":     timing.get("planner_ms", 0) or 0,
        "retrieval_ms":   timing.get("retrieval_ms", 0) or 0,
        "tool_ms":        timing.get("tool_ms", 0) or 0,
        "responder_ms":   timing.get("responder_ms", 0) or 0,
        "tool_called":    len(tool_names) > 0,
        "executed_tools": ", ".join(tool_names),
        "pending_confirmation": bool(data.get("pending_confirmation")),
    }

# ── Judge ─────────────────────────────────────────────────────────────────────
_JUDGE_URL = "https://api.openai.com/v1/chat/completions"

async def judge(client: httpx.AsyncClient, utterance: str, response: str,
                required: str, forbidden: str) -> float:
    key = os.environ.get("CI_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key or (not required.strip() and not forbidden.strip()):
        return -1.0
    prompt = (f"Score the agent response 0-100.\n"
              f"Required facts (ALL must appear): {required or 'none'}\n"
              f"Forbidden facts (must NOT appear): {forbidden or 'none'}\n"
              f"Customer: {utterance}\nAgent: {response}\n"
              f'JSON only: {{"score":<int>,"reason":"<10 words>"}}')
    try:
        r = await client.post(_JUDGE_URL,
            json={"model":"gpt-4o-mini","messages":[{"role":"user","content":prompt}],
                  "max_tokens":80,"temperature":0},
            headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        return float(json.loads(r.json()["choices"][0]["message"]["content"])["score"])
    except Exception:
        return -1.0

# ── EvalRow ───────────────────────────────────────────────────────────────────
@dataclass
class EvalRow:
    test_type: str = ""; thread_id: str = ""; archetype: str = ""
    turn_no: int = 1; row_id: str = ""; lang: str = ""; difficulty: str = ""
    slice: str = ""; utterance: str = ""; direction: str = ""
    tool_category: str = ""; expected_tools: str = ""
    tool_expected: bool = False
    planner_ms: int = 0; retrieval_ms: int = 0; tool_ms: int = 0
    responder_ms: int = 0; wall_ms: int = 0
    tool_called: bool = False; tool_match: bool = False
    executed_tools: str = ""; confirmed_turn: bool = False
    accuracy_score: float = -1.0
    response_text: str = ""; error: str = ""; timestamp: str = ""

COLS = list(EvalRow.__dataclass_fields__.keys())

def _save(row: EvalRow):
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({c: getattr(row, c) for c in COLS}, ensure_ascii=False) + "\n")

def _log(row: EvalRow):
    trouble = "⚠ " if "having trouble on my side" in row.response_text.lower() else ""
    err     = f" ERR={row.error[:40]}" if row.error else ""
    print(f"    {trouble}{row.row_id or row.thread_id} [{row.lang}|{row.direction}|{row.difficulty}] "
          f"acc={row.accuracy_score:.0f} wall={row.wall_ms}ms "
          f"tool={'✓' if row.tool_called else '✗'}{err}")

# ── Single-turn ───────────────────────────────────────────────────────────────
async def run_st(sem, client, meta, direction, keeper, idx, total):
    async with sem:
        token  = await keeper.fresh(client)
        exp_tools_raw = meta.get("expected_tool_calls") or ""
        exp_tools = [t.strip() for t in str(exp_tools_raw).split(",") if t.strip()] \
                    if exp_tools_raw else []
        utt    = str(meta.get("user_utterance", ""))
        row    = EvalRow(
            test_type="single_turn", row_id=str(meta.get("id","")),
            lang=meta.get("lang",""), difficulty=meta.get("difficulty",""),
            slice=meta.get("slice",""), utterance=utt, direction=direction,
            tool_category="with_tool" if exp_tools else "without_tool",
            expected_tools=", ".join(exp_tools), tool_expected=bool(exp_tools),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        print(f"  [{idx}/{total}] {row.row_id} {row.lang} {direction}", end=" ", flush=True)
        try:
            sid  = await new_session(client, token, direction)
            # outbound primer
            if direction == "outbound":
                token = await keeper.fresh(client)
                await send_turn(client, token, sid,
                                "Yes this is me. I was interested in a 3BHK.")
                token = await keeper.fresh(client)
            data, wall_ms = await send_turn(client, token, sid, utt)
            p = parse_response(data, wall_ms)
            row.response_text = p["response_text"]
            row.wall_ms = p["wall_ms"]; row.planner_ms = p["planner_ms"]
            row.retrieval_ms = p["retrieval_ms"]; row.tool_ms = p["tool_ms"]
            row.responder_ms = p["responder_ms"]
            row.tool_called = p["tool_called"]; row.executed_tools = p["executed_tools"]
            row.tool_match = (row.tool_called == row.tool_expected)

            # confirmation follow-up if agent is asking
            if not row.tool_called and CONFIRM_RE.search(row.response_text):
                row.confirmed_turn = True
                token = await keeper.fresh(client)
                data2, _ = await send_turn(client, token, sid, "Yes, go ahead please.")
                p2 = parse_response(data2, 0)
                if p2["tool_called"]:
                    row.tool_called = True
                    row.executed_tools = p2["executed_tools"]
                    row.tool_match = (row.tool_called == row.tool_expected)

            row.accuracy_score = await judge(
                client, utt, row.response_text,
                str(meta.get("required_facts","") or ""),
                str(meta.get("forbidden_facts","") or ""))
            await end_session(client, token, sid)
        except Exception as e:
            row.error = str(e)[:200]
        _save(row); _log(row)
        return row

# ── Thread ────────────────────────────────────────────────────────────────────
async def run_thread(client, thread_rows, direction, keeper, t_idx, n_threads):
    token = await keeper.fresh(client)
    tid   = thread_rows[0].get("thread_id","")
    arch  = thread_rows[0].get("archetype","")
    lang  = thread_rows[0].get("lang","")
    print(f"  [T{t_idx}/{n_threads}] {tid} {arch} {lang} {direction}")
    results = []
    try:
        sid = await new_session(client, token, direction)
        if direction == "outbound":
            token = await keeper.fresh(client)
            await send_turn(client, token, sid,
                            "Yes this is me. I was interested in a 3BHK.")
        for row_meta in thread_rows:
            token = await keeper.fresh(client)
            utt = str(row_meta.get("utterance",""))
            row = EvalRow(
                test_type="thread", thread_id=tid, archetype=arch,
                turn_no=int(row_meta.get("turn_no",1)), lang=lang,
                direction=direction, utterance=utt,
                tool_category="with_tool" if (row_meta.get("expected_tool_calls") or "") else "without_tool",
                expected_tools=str(row_meta.get("expected_tool_calls","") or ""),
                tool_expected=bool(row_meta.get("expected_tool_calls","") or ""),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            try:
                data, wall_ms = await send_turn(client, token, sid, utt)
                p = parse_response(data, wall_ms)
                row.response_text = p["response_text"]
                row.wall_ms = p["wall_ms"]; row.planner_ms = p["planner_ms"]
                row.retrieval_ms = p["retrieval_ms"]; row.tool_ms = p["tool_ms"]
                row.responder_ms = p["responder_ms"]
                row.tool_called = p["tool_called"]; row.executed_tools = p["executed_tools"]
                row.tool_match = (row.tool_called == row.tool_expected)
                if not row.tool_called and CONFIRM_RE.search(row.response_text):
                    row.confirmed_turn = True
                    token = await keeper.fresh(client)
                    data2, _ = await send_turn(client, token, sid, "Yes, go ahead please.")
                    p2 = parse_response(data2, 0)
                    if p2["tool_called"]:
                        row.tool_called = True
                        row.executed_tools = p2["executed_tools"]
                        row.tool_match = (row.tool_called == row.tool_expected)
                row.accuracy_score = await judge(
                    client, utt, row.response_text,
                    str(row_meta.get("required_facts","") or ""),
                    str(row_meta.get("forbidden_facts","") or ""))
            except Exception as e:
                row.error = str(e)[:200]
            _save(row); results.append(row)
            trouble = "⚠ " if "having trouble on my side" in row.response_text.lower() else ""
            print(f"      {trouble}turn {row.turn_no} wall={row.wall_ms}ms "
                  f"tool={'✓' if row.tool_called else '✗'} acc={row.accuracy_score:.0f}")
        await end_session(client, token, sid)
    except Exception as e:
        print(f"    THREAD FAILED: {e}")
    return results

# ── Stats ─────────────────────────────────────────────────────────────────────
def print_stats(rows):
    import statistics as st2
    segs = {}
    for r in rows:
        k = (r.test_type, r.direction, r.tool_category)
        segs.setdefault(k, []).append(r)

    W = 72
    print("\n" + "="*W)
    print(" HIGHLAND GREENZ SUBSET EVAL RESULTS")
    print("="*W)
    hdr = f"{'Segment':<34} {'n':>4} {'Fire':>8} {'Acc μ':>6} {'Wall μ':>7} {'p95':>7} {'⚠':>4}"
    print(hdr); print("-"*W)

    all_walls=[]; all_acc=[]; all_exp=0; all_fired=0; all_trouble=0
    for k in sorted(segs):
        rs = segs[k]
        tt, d, tc = k
        lbl = f"{tt[:2].upper()} {d[:2].upper()} {tc[:7]}"
        walls  = [r.wall_ms for r in rs if r.wall_ms > 0]
        accs   = [r.accuracy_score for r in rs if r.accuracy_score >= 0]
        fired  = sum(1 for r in rs if r.tool_called)
        exp    = sum(1 for r in rs if r.tool_expected)
        trouble= sum(1 for r in rs if "having trouble on my side" in r.response_text.lower())
        fire_s = f"{fired}/{exp} ({100*fired/exp:.0f}%)" if exp else "— (FP:0%)" if not fired else f"FP:{fired}"
        acc_s  = f"{st2.mean(accs):.1f}" if accs else "—"
        wall_s = f"{int(st2.mean(walls))}" if walls else "—"
        p95_s  = f"{sorted(walls)[min(int(len(walls)*.95),len(walls)-1)]}" if walls else "—"
        print(f"  {lbl:<32} {len(rs):>4} {fire_s:>10} {acc_s:>6} {wall_s:>6}ms {p95_s:>6}ms {trouble:>4}")
        all_walls.extend(walls); all_acc.extend(accs)
        all_exp+=exp; all_fired+=fired; all_trouble+=trouble

    print("-"*W)
    fire_s = f"{all_fired}/{all_exp} ({100*all_fired/all_exp:.1f}%)" if all_exp else "—"
    acc_s  = f"{st2.mean(all_acc):.1f}" if all_acc else "—"
    wall_s = f"{int(st2.mean(all_walls))}" if all_walls else "—"
    p95_s  = f"{sorted(all_walls)[min(int(len(all_walls)*.95),len(all_walls)-1)]}" if all_walls else "—"
    print(f"  {'OVERALL':<32} {len(rows):>4} {fire_s:>10} {acc_s:>6} {wall_s:>6}ms {p95_s:>6}ms {all_trouble:>4}")
    print(f"\n  ⚠  Trouble-phrase rows: {all_trouble}/{len(rows)} ({100*all_trouble/len(rows):.1f}%)")
    errors = sum(1 for r in rows if r.error)
    print(f"  ✗  Errors: {errors}/{len(rows)}")

    # lang breakdown (ST only)
    st_rows = [r for r in rows if r.test_type=="single_turn"]
    if st_rows:
        print("\n  Language (ST only):")
        for lang in ["en","hi","hinglish"]:
            lr = [r for r in st_rows if r.lang==lang]
            walls = [r.wall_ms for r in lr if r.wall_ms>0]
            accs  = [r.accuracy_score for r in lr if r.accuracy_score>=0]
            wt    = [r for r in lr if r.tool_expected]
            fired = sum(1 for r in wt if r.tool_called)
            acc_s = f"{st2.mean(accs):.1f}" if accs else "—"
            wall_s= f"{int(st2.mean(walls))}" if walls else "—"
            fire_s= f"{fired}/{len(wt)} ({100*fired/len(wt):.0f}%)" if wt else "—"
            print(f"    {lang:<10} n={len(lr):3d}  fire={fire_s:>14}  acc={acc_s:>5}  wall={wall_s}ms")
    print("="*W)

# ── Excel output ──────────────────────────────────────────────────────────────
def write_xlsx(rows):
    df = pd.DataFrame([{c: getattr(r, c) for c in COLS} for r in rows])
    with pd.ExcelWriter(XLSX_OUT, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="All_Rows", index=False)
        df[df["test_type"]=="single_turn"].to_excel(w, sheet_name="SingleTurn", index=False)
        df[df["test_type"]=="thread"].to_excel(w, sheet_name="Threads", index=False)
        df[df["tool_category"]=="with_tool"].to_excel(w, sheet_name="WithTool", index=False)
        df[df["tool_category"]=="without_tool"].to_excel(w, sheet_name="WithoutTool", index=False)

        # summary pivot
        rows_no_err = [r for r in rows if not r.error]
        import statistics as st2
        pivot = []
        for (tt,d,tc), rs in sorted({
            (r.test_type,r.direction,r.tool_category): None for r in rows_no_err}.items()):
            rs = [r for r in rows_no_err
                  if r.test_type==tt and r.direction==d and r.tool_category==tc]
            walls=[r.wall_ms for r in rs if r.wall_ms>0]
            accs=[r.accuracy_score for r in rs if r.accuracy_score>=0]
            fired=sum(1 for r in rs if r.tool_called)
            exp=sum(1 for r in rs if r.tool_expected)
            trouble=sum(1 for r in rs if "having trouble on my side" in r.response_text.lower())
            pivot.append({
                "type":tt,"direction":d,"category":tc,"n":len(rs),
                "tool_exp":exp,"tool_fired":fired,
                "fire_pct":round(100*fired/exp,1) if exp else None,
                "false_pos":round(100*fired/len(rs),1) if not exp and fired else None,
                "acc_mean":round(st2.mean(accs),1) if accs else None,
                "wall_mean":round(st2.mean(walls)) if walls else None,
                "wall_p95":sorted(walls)[min(int(len(walls)*.95),len(walls)-1)] if walls else None,
                "plan_mean":round(st2.mean([r.planner_ms for r in rs if r.planner_ms>0])) if any(r.planner_ms>0 for r in rs) else None,
                "resp_mean":round(st2.mean([r.responder_ms for r in rs if r.responder_ms>0])) if any(r.responder_ms>0 for r in rs) else None,
                "trouble":trouble,
            })
        pd.DataFrame(pivot).to_excel(w, sheet_name="Summary", index=False)
    print(f"\nSaved: {XLSX_OUT}")

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print(f"\nHighland Greenz Subset Eval  ({time.strftime('%Y-%m-%d %H:%M')})")
    print(f"Backend: {BASE}")
    print(f"Dataset: {DATASET.name}\n")

    if not DATASET.exists():
        print("ERROR: Dataset not found. Run build_balanced_dataset.py first.")
        return

    if JSONL.exists():
        JSONL.unlink()

    st_df = pd.read_excel(DATASET, sheet_name="Golden_SingleTurn")
    th_df = pd.read_excel(DATASET, sheet_name="Golden_Threads")
    st_rows  = st_df.to_dict("records")
    th_ids   = th_df["thread_id"].unique().tolist()
    th_turns = len(th_df)
    total_calls = len(st_rows)*2 + th_turns*2
    print(f"ST rows: {len(st_rows)} × 2 dirs = {len(st_rows)*2} calls")
    print(f"Threads: {len(th_ids)} × avg {th_turns/len(th_ids):.1f} turns × 2 dirs = {th_turns*2} calls")
    print(f"Total:   {total_calls} LLM calls\n")

    keeper = TokenKeeper()
    sem    = asyncio.Semaphore(3)
    all_rows = []

    async with httpx.AsyncClient(timeout=120) as client:

        # ── Single-turn ───────────────────────────────────────────────────────
        print("─"*55)
        print("SINGLE-TURN  (100 rows × 2 directions = 200 calls)")
        print("─"*55)
        total_st = len(st_rows) * 2
        tasks = []
        for i, row in enumerate(st_rows):
            for j, direction in enumerate(["inbound","outbound"]):
                idx = i*2 + j + 1
                tasks.append(run_st(sem, client, row, direction, keeper, idx, total_st))
        st_results = await asyncio.gather(*tasks)
        all_rows.extend(st_results)
        ok  = sum(1 for r in st_results if not r.error)
        err = sum(1 for r in st_results if r.error)
        print(f"\nST done: {ok} ok, {err} errors\n")

        # ── Threads ───────────────────────────────────────────────────────────
        print("─"*55)
        print(f"THREADS  ({len(th_ids)} threads × 2 directions = {len(th_ids)*2} runs)")
        print("─"*55)
        for t_idx, tid in enumerate(th_ids, 1):
            t_rows = (th_df[th_df["thread_id"]==tid]
                      .sort_values("turn_no").to_dict("records"))
            for direction in ["inbound","outbound"]:
                keeper._issued = 0   # force token refresh between threads
                res = await run_thread(client, t_rows, direction, keeper, t_idx, len(th_ids))
                all_rows.extend(res)

    print_stats(all_rows)
    write_xlsx(all_rows)

if __name__ == "__main__":
    asyncio.run(main())
