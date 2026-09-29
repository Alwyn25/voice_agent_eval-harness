"""
mini_eval.py  —  Quick latency+accuracy probe for incremental changes.
Picks first N ST rows + first N thread ids from the golden dataset,
runs inbound direction only to keep runtime under 5 minutes.

Usage:
    python eval_harness/mini_eval.py                     # defaults: label=run, N=10
    python eval_harness/mini_eval.py --label baseline
    python eval_harness/mini_eval.py --label step1 --compare baseline
    python eval_harness/mini_eval.py --n 15
"""
import argparse, asyncio, json, os, re, statistics, time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
BASE    = "http://localhost:8000"
DATASET = Path(__file__).parent / "golden_subset_100st_40th.xlsx"
OUT_DIR = Path(__file__).parent / "results" / "mini"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMAIL   = "admin@advora.ai"
PASS    = "Admin@123"

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
_JUDGE_URL = "https://api.openai.com/v1/chat/completions"

# ── Helpers ────────────────────────────────────────────────────────────────────
class TokenKeeper:
    def __init__(self): self._token = ""; self._issued = 0.0

    async def fresh(self, client: httpx.AsyncClient) -> str:
        if time.monotonic() - self._issued > 1400:
            r = await client.post(f"{BASE}/auth/login",
                                  json={"email": EMAIL, "password": PASS}, timeout=15)
            r.raise_for_status()
            self._token = r.json()["access_token"]
            self._issued = time.monotonic()
        return self._token

async def new_session(client, token, direction):
    body = {"channel": "phone", "direction": direction,
            "purpose": "sales", "enforce_outbound_policy": False}
    r = await client.post(f"{BASE}/sessions", json=body,
                          headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if r.status_code not in (200, 201):
        r.raise_for_status()
    return r.json()["session_id"]

async def send_turn(client, token, sid, text):
    t0 = time.perf_counter()
    r  = await client.post(f"{BASE}/sessions/{sid}/turns",
                           json={"text": text},
                           headers={"Authorization": f"Bearer {token}"}, timeout=120)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    r.raise_for_status()
    return r.json(), wall_ms

async def end_session(client, token, sid):
    try:
        await client.post(f"{BASE}/sessions/{sid}/end",
                          json={"analyse": "skip"},
                          headers={"Authorization": f"Bearer {token}"}, timeout=10)
    except Exception:
        pass

def parse_response(data, wall_ms):
    debug  = data.get("debug") or {}
    timing = debug.get("timing") or {}
    executed = data.get("executed_tools") or []
    if isinstance(executed, list):
        tool_names = [(t.get("tool") or t.get("name") or str(t)) if isinstance(t, dict) else str(t)
                      for t in executed]
    else:
        tool_names = [str(executed)] if executed else []
    return {
        "response_text": data.get("response_text", ""),
        "wall_ms":       timing.get("wall_ms", 0) or wall_ms,
        "planner_ms":    timing.get("planner_ms", 0) or 0,
        "retrieval_ms":  timing.get("retrieval_ms", 0) or 0,
        "tool_ms":       timing.get("tool_ms", 0) or 0,
        "responder_ms":  timing.get("responder_ms", 0) or 0,
        "tool_called":   len(tool_names) > 0,
        "executed_tools": ", ".join(tool_names),
    }

async def judge(client, utterance, response, required, forbidden):
    key = os.environ.get("CI_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key or (not str(required).strip() and not str(forbidden).strip()):
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

# ── Row ────────────────────────────────────────────────────────────────────────
@dataclass
class Row:
    test_type: str = ""; label: str = ""; row_id: str = ""; thread_id: str = ""
    turn_no: int = 1; utterance: str = ""; tool_expected: bool = False
    wall_ms: int = 0; planner_ms: int = 0; retrieval_ms: int = 0
    tool_ms: int = 0; responder_ms: int = 0
    tool_called: bool = False; tool_match: bool = False
    accuracy_score: float = -1.0; error: str = ""

# ── Single-turn ────────────────────────────────────────────────────────────────
async def run_st(sem, client, meta, keeper, idx, total, label):
    async with sem:
        token     = await keeper.fresh(client)
        utt       = str(meta.get("user_utterance", ""))
        exp_tools = [t.strip() for t in str(meta.get("expected_tool_calls","") or "").split(",") if t.strip()]
        row       = Row(test_type="st", label=label, row_id=str(meta.get("id","")),
                        utterance=utt, tool_expected=bool(exp_tools))
        print(f"  ST [{idx}/{total}] {row.row_id}", end=" ", flush=True)
        try:
            sid  = await new_session(client, token, "inbound")
            data, wall_ms = await send_turn(client, token, sid, utt)
            p = parse_response(data, wall_ms)
            row.wall_ms = p["wall_ms"]; row.planner_ms = p["planner_ms"]
            row.retrieval_ms = p["retrieval_ms"]; row.tool_ms = p["tool_ms"]
            row.responder_ms = p["responder_ms"]; row.tool_called = p["tool_called"]
            row.tool_match = (row.tool_called == row.tool_expected)
            # follow up confirmation if needed
            if not row.tool_called and CONFIRM_RE.search(p["response_text"]):
                token = await keeper.fresh(client)
                data2, _ = await send_turn(client, token, sid, "Yes, go ahead please.")
                p2 = parse_response(data2, 0)
                if p2["tool_called"]:
                    row.tool_called = True; row.tool_match = row.tool_expected
            row.accuracy_score = await judge(client, utt, p["response_text"],
                meta.get("required_facts",""), meta.get("forbidden_facts",""))
            await end_session(client, token, sid)
        except Exception as e:
            row.error = str(e)[:200]
        trouble = "⚠ " if "having trouble on my side" in (p.get("response_text","") if 'p' in dir() else "").lower() else ""
        print(f"{trouble}acc={row.accuracy_score:.0f} wall={row.wall_ms}ms")
        return row

# ── Thread ─────────────────────────────────────────────────────────────────────
async def run_thread(client, thread_rows, keeper, t_idx, n_threads, label):
    token  = await keeper.fresh(client)
    tid    = thread_rows[0].get("thread_id","")
    print(f"  TH [{t_idx}/{n_threads}] {tid}")
    results = []
    try:
        sid = await new_session(client, token, "inbound")
        for row_meta in thread_rows:
            token = await keeper.fresh(client)
            utt   = str(row_meta.get("utterance",""))
            row   = Row(test_type="th", label=label, thread_id=tid,
                        turn_no=int(row_meta.get("turn_no",1)), utterance=utt,
                        tool_expected=bool(row_meta.get("expected_tool_calls","") or ""))
            try:
                data, wall_ms = await send_turn(client, token, sid, utt)
                p = parse_response(data, wall_ms)
                row.wall_ms = p["wall_ms"]; row.planner_ms = p["planner_ms"]
                row.retrieval_ms = p["retrieval_ms"]; row.tool_ms = p["tool_ms"]
                row.responder_ms = p["responder_ms"]; row.tool_called = p["tool_called"]
                row.tool_match = (row.tool_called == row.tool_expected)
                if not row.tool_called and CONFIRM_RE.search(p["response_text"]):
                    token = await keeper.fresh(client)
                    data2, _ = await send_turn(client, token, sid, "Yes, go ahead please.")
                    p2 = parse_response(data2, 0)
                    if p2["tool_called"]:
                        row.tool_called = True; row.tool_match = row.tool_expected
                row.accuracy_score = await judge(client, utt, p["response_text"],
                    row_meta.get("required_facts",""), row_meta.get("forbidden_facts",""))
            except Exception as e:
                row.error = str(e)[:200]
            trouble = "⚠ " if "having trouble on my side" in (p.get("response_text","") if 'p' in dir() else "").lower() else ""
            print(f"      {trouble}turn {row.turn_no} wall={row.wall_ms}ms acc={row.accuracy_score:.0f}")
            results.append(row)
        await end_session(client, token, sid)
    except Exception as e:
        print(f"    THREAD FAILED: {e}")
    return results

# ── Stats ──────────────────────────────────────────────────────────────────────
def compute_stats(rows):
    walls  = [r.wall_ms for r in rows if r.wall_ms > 0 and not r.error]
    accs   = [r.accuracy_score for r in rows if r.accuracy_score >= 0]
    errors = sum(1 for r in rows if r.error)
    trouble= sum(1 for r in rows if "having trouble on my side" in r.utterance.lower())
    p95    = sorted(walls)[min(int(len(walls) * .95), len(walls) - 1)] if walls else 0
    return {
        "n": len(rows), "errors": errors, "trouble": trouble,
        "wall_mean": round(statistics.mean(walls)) if walls else 0,
        "wall_p50":  sorted(walls)[len(walls)//2] if walls else 0,
        "wall_p95":  p95,
        "acc_mean":  round(statistics.mean(accs), 1) if accs else -1,
        "tool_fire_pct": round(100 * sum(1 for r in rows if r.tool_called) /
                               max(1, sum(1 for r in rows if r.tool_expected)), 1)
                         if any(r.tool_expected for r in rows) else None,
    }

def print_summary(rows, label):
    st_rows = [r for r in rows if r.test_type == "st"]
    th_rows = [r for r in rows if r.test_type == "th"]
    W = 68
    print("\n" + "=" * W)
    print(f"  MINI EVAL — {label.upper()}")
    print("=" * W)
    for seg, rs in [("Single-Turn", st_rows), ("Thread", th_rows), ("OVERALL", rows)]:
        if not rs: continue
        s = compute_stats(rs)
        print(f"  {seg:<14}  n={s['n']:>3}  wall μ={s['wall_mean']:>5}ms  "
              f"p95={s['wall_p95']:>5}ms  acc={s['acc_mean']:>5}  err={s['errors']}")
    print("=" * W)

def compare(base_file, new_file, new_label):
    def load(path):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(Row(**json.loads(line)))
        return rows

    base = load(base_file)
    new  = load(new_file)

    W = 72
    print("\n" + "=" * W)
    print(f"  COMPARISON  baseline  →  {new_label}")
    print("=" * W)
    hdr = f"  {'Segment':<14} {'wall μ':>8} {'p95':>8} {'acc':>6} {'Δwall':>8} {'Δacc':>6} {'pass?':>6}"
    print(hdr); print("-" * W)

    all_pass = True
    for seg, base_rows, new_rows in [
        ("Single-Turn",
         [r for r in base if r.test_type == "st"],
         [r for r in new  if r.test_type == "st"]),
        ("Thread",
         [r for r in base if r.test_type == "th"],
         [r for r in new  if r.test_type == "th"]),
        ("OVERALL", base, new),
    ]:
        if not base_rows or not new_rows:
            continue
        bs = compute_stats(base_rows)
        ns = compute_stats(new_rows)
        dw = ns["wall_mean"] - bs["wall_mean"]
        da = (ns["acc_mean"] - bs["acc_mean"]) if (bs["acc_mean"] >= 0 and ns["acc_mean"] >= 0) else None
        # PASS if wall improved (negative delta) AND accuracy did not drop more than 1 point
        passed = (dw < 0) and (da is None or da >= -1.0)
        if seg == "OVERALL":
            all_pass = passed
        sign_w = "+" if dw >= 0 else ""
        sign_a = ("+" if (da or 0) >= 0 else "") if da is not None else ""
        da_s   = f"{sign_a}{da:.1f}" if da is not None else "  —"
        ok     = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {seg:<14} {bs['wall_mean']:>5}→{ns['wall_mean']:<5}ms "
              f"{bs['wall_p95']:>5}→{ns['wall_p95']:<5}ms "
              f"{bs['acc_mean']:>4}→{ns['acc_mean']:<4} "
              f"{sign_w}{dw:>4}ms {da_s:>6}  {ok}")
    print("=" * W)
    return all_pass

# ── Main ───────────────────────────────────────────────────────────────────────
async def main(label: str, n: int, compare_label: str | None, dataset: Path):
    print(f"\nMini Eval  label={label}  n={n}  dataset={dataset.name}  ({time.strftime('%Y-%m-%d %H:%M')})")
    jsonl_path = OUT_DIR / f"mini_{label}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    st_df = pd.read_excel(dataset, sheet_name="Golden_SingleTurn")
    th_df = pd.read_excel(dataset, sheet_name="Golden_Threads")

    st_rows  = st_df.to_dict("records")[:n]
    th_ids   = th_df["thread_id"].unique().tolist()[:n]

    print(f"ST: {len(st_rows)} rows  |  Threads: {len(th_ids)}\n")

    keeper   = TokenKeeper()
    sem      = asyncio.Semaphore(3)
    all_rows: list[Row] = []

    async with httpx.AsyncClient(timeout=120) as client:
        # ── ST ──────────────────────────────────────────────────────────────
        print("── Single-Turn ──")
        tasks = [run_st(sem, client, m, keeper, i+1, len(st_rows), label)
                 for i, m in enumerate(st_rows)]
        st_results = await asyncio.gather(*tasks)
        all_rows.extend(st_results)

        # ── Threads ──────────────────────────────────────────────────────────
        print("\n── Threads ──")
        for t_idx, tid in enumerate(th_ids, 1):
            t_rows = (th_df[th_df["thread_id"] == tid]
                      .sort_values("turn_no").to_dict("records"))
            keeper._issued = 0
            res = await run_thread(client, t_rows, keeper, t_idx, len(th_ids), label)
            all_rows.extend(res)

    # Save
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

    print_summary(all_rows, label)

    if compare_label:
        base_path = OUT_DIR / f"mini_{compare_label}.jsonl"
        if base_path.exists():
            passed = compare(base_path, jsonl_path, label)
            print(f"\n  → Overall verdict: {'PASS ✓' if passed else 'FAIL ✗'}")
            return passed
        else:
            print(f"  [compare] baseline file not found: {base_path}")
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label",   default="run")
    ap.add_argument("--n",       type=int, default=10)
    ap.add_argument("--compare", default=None, metavar="LABEL",
                    help="compare against this label's results")
    ap.add_argument("--dataset", default=None, metavar="PATH",
                    help="path to golden dataset xlsx (default: golden_subset_100st_40th.xlsx)")
    args = ap.parse_args()
    ds = Path(args.dataset) if args.dataset else DATASET
    result = asyncio.run(main(args.label, args.n, args.compare, ds))
    if result is False:
        raise SystemExit(1)
