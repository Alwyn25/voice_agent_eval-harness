"""
build_golden_balanced.py
────────────────────────────────────────────────────────────────────────────
Builds golden_balanced_200st_40th.xlsx from highland_greenz_golden_dataset.xlsx.

Sampling rules
──────────────
Single-Turn (200 rows):
  • 67 en / 67 hi / 66 hinglish
  • Within each language, proportional to difficulty (easy / medium / hard)
  • Within each (lang × difficulty) stratum, ensure tool-type coverage by
    cycling through rows sorted by expected_tool_calls

Threads (40 threads):
  • 14 en / 13 hi / 13 hinglish
  • Within each language, maximise archetype diversity
  • Mix of 2-turn, 3-turn and 4-turn threads

Usage:
    python eval_harness/build_golden_balanced.py
Writes to: eval_harness/golden_balanced_200st_40th.xlsx
"""
import math, random
from collections import defaultdict
from pathlib import Path

import openpyxl
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

SRC  = Path(__file__).parent.parent / "highland_greenz_golden_dataset.xlsx"
# also accept Desktop path as fallback
if not SRC.exists():
    SRC = Path(r"C:\Users\ADVORA-ALWYN\OneDrive\Desktop\highland_greenz_golden_dataset.xlsx")
DST  = Path(__file__).parent / "golden_balanced_200st_40th.xlsx"

random.seed(42)

# ── Load source ───────────────────────────────────────────────────────────────
wb_src = load_workbook(SRC, data_only=True)

def sheet_to_rows(ws):
    headers = [c.value for c in ws[1]]
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            rows.append(dict(zip(headers, row)))
    return headers, rows

st_headers, st_all = sheet_to_rows(wb_src["Golden_SingleTurn"])
th_headers, th_all = sheet_to_rows(wb_src["Golden_Threads"])

# ── Single-turn sampling ──────────────────────────────────────────────────────
LANG_TARGETS = {"en": 67, "hi": 67, "hinglish": 66}

def tool_key(row):
    """Stable short key for a row's tool signature."""
    t = str(row.get("expected_tool_calls") or "").strip()
    if not t or t in ("[]", "None", ""):
        return "none"
    return t

selected_st = []

for lang, lang_n in LANG_TARGETS.items():
    lang_rows = [r for r in st_all if r["lang"] == lang]

    # Count per difficulty in this language slice
    diff_counts = defaultdict(int)
    for r in lang_rows:
        diff_counts[r["difficulty"]] += 1
    total = sum(diff_counts.values())

    # Allocate proportionally, then round-adjust to hit lang_n exactly
    alloc = {}
    for d in ("easy", "medium", "hard"):
        alloc[d] = max(1, round(diff_counts[d] / total * lang_n))
    # Trim/expand to hit lang_n exactly
    keys = sorted(alloc, key=lambda x: -alloc[x])
    delta = lang_n - sum(alloc.values())
    alloc[keys[0]] += delta

    for diff, k in alloc.items():
        pool = [r for r in lang_rows if r["difficulty"] == diff]
        # Sort by tool_key so cycling gives tool coverage
        pool_sorted = sorted(pool, key=lambda r: (tool_key(r), r["id"] or ""))
        # Systematic sample: take every (len/k)-th item to spread coverage
        step = max(1, len(pool_sorted) // k)
        picked = [pool_sorted[i * step % len(pool_sorted)] for i in range(k)]
        # De-dup (in case step=1 causes repeat at wrap)
        seen_ids = set()
        deduped = []
        for r in picked:
            rid = r["id"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                deduped.append(r)
        # Pad if dedup lost rows
        extras = [r for r in pool_sorted if r["id"] not in seen_ids]
        deduped.extend(extras[: k - len(deduped)])
        selected_st.extend(deduped[:k])

print(f"ST selected: {len(selected_st)}")
from collections import Counter
print("  lang      :", Counter(r["lang"]       for r in selected_st))
print("  difficulty:", Counter(r["difficulty"] for r in selected_st))
print("  tool      :", Counter(tool_key(r)     for r in selected_st).most_common(8))

# ── Thread sampling ────────────────────────────────────────────────────────────
# Build per-thread summary
by_thread = defaultdict(list)
for r in th_all:
    by_thread[r["thread_id"]].append(r)

def thread_meta(tid):
    rows = sorted(by_thread[tid], key=lambda r: r["turn_no"])
    return {
        "thread_id": tid,
        "lang": rows[0]["lang"],
        "archetype": rows[0].get("archetype", ""),
        "turns": len(rows),
    }

all_threads = [thread_meta(tid) for tid in by_thread]

TH_LANG_TARGETS = {"en": 14, "hi": 13, "hinglish": 13}
selected_th_ids = []

for lang, lang_n in TH_LANG_TARGETS.items():
    pool = [t for t in all_threads if t["lang"] == lang]
    # Sort by archetype to maximise coverage; within archetype vary turn-length
    pool_sorted = sorted(pool, key=lambda t: (t["archetype"], -t["turns"]))
    # Systematic sample
    step = max(1, len(pool_sorted) // lang_n)
    picked = [pool_sorted[i * step % len(pool_sorted)]
              for i in range(lang_n)]
    selected_th_ids.extend(t["thread_id"] for t in picked)

print(f"\nThreads selected: {len(selected_th_ids)}")
from collections import Counter as C2
lang_dist = C2(by_thread[tid][0]["lang"] for tid in selected_th_ids)
print("  lang     :", lang_dist)
arch_dist = C2(by_thread[tid][0].get("archetype","?") for tid in selected_th_ids)
print("  archetype:", arch_dist.most_common(6))
turn_dist = C2(len(by_thread[tid]) for tid in selected_th_ids)
print("  turns/th :", dict(sorted(turn_dist.items())))

# ── Build output workbook ─────────────────────────────────────────────────────
wb_dst = openpyxl.Workbook()
wb_dst.remove(wb_dst.active)  # remove default sheet

# Header style
HDR_FILL = PatternFill("solid", fgColor="1F4E79")
HDR_FONT = Font(color="FFFFFF", bold=True, name="Arial", size=10)
BODY_FONT = Font(name="Arial", size=10)
CENTER    = Alignment(horizontal="center", vertical="top", wrap_text=False)
LEFT      = Alignment(horizontal="left",   vertical="top", wrap_text=True)

def write_sheet(ws, headers, rows):
    ws.append(headers)
    for cell in ws[1]:
        cell.font = HDR_FONT
        cell.fill = HDR_FILL
        cell.alignment = CENTER
    for row in rows:
        ws.append([row.get(h) for h in headers])
    for col_idx, _ in enumerate(headers, 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = 22
    # apply body font
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font  = BODY_FONT
            cell.alignment = LEFT

# SingleTurn sheet  — add lang/difficulty explicitly if not present
ws_st = wb_dst.create_sheet("Golden_SingleTurn")
write_sheet(ws_st, st_headers, selected_st)

# Threads sheet
selected_th_rows = []
for tid in selected_th_ids:
    selected_th_rows.extend(sorted(by_thread[tid], key=lambda r: r["turn_no"]))
ws_th = wb_dst.create_sheet("Golden_Threads")
write_sheet(ws_th, th_headers, selected_th_rows)

# Coverage sheet
ws_cov = wb_dst.create_sheet("Coverage")
ws_cov.append(["Metric", "Value"])
ws_cov.append(["Single-turn rows",   len(selected_st)])
ws_cov.append(["Thread count",       len(selected_th_ids)])
ws_cov.append(["Thread turns total", len(selected_th_rows)])
ws_cov.append(["Total eval rows",    len(selected_st) + len(selected_th_rows)])
ws_cov.append(["──────────────────", "──────────────────"])
ws_cov.append(["ST · en",        sum(1 for r in selected_st if r["lang"]=="en")])
ws_cov.append(["ST · hi",        sum(1 for r in selected_st if r["lang"]=="hi")])
ws_cov.append(["ST · hinglish",  sum(1 for r in selected_st if r["lang"]=="hinglish")])
ws_cov.append(["ST · easy",      sum(1 for r in selected_st if r["difficulty"]=="easy")])
ws_cov.append(["ST · medium",    sum(1 for r in selected_st if r["difficulty"]=="medium")])
ws_cov.append(["ST · hard",      sum(1 for r in selected_st if r["difficulty"]=="hard")])
ws_cov.append(["ST · no-tool",   sum(1 for r in selected_st if tool_key(r)=="none")])
ws_cov.append(["ST · has-tool",  sum(1 for r in selected_st if tool_key(r)!="none")])
ws_cov.append(["TH · en",        sum(1 for tid in selected_th_ids if by_thread[tid][0]["lang"]=="en")])
ws_cov.append(["TH · hi",        sum(1 for tid in selected_th_ids if by_thread[tid][0]["lang"]=="hi")])
ws_cov.append(["TH · hinglish",  sum(1 for tid in selected_th_ids if by_thread[tid][0]["lang"]=="hinglish")])

wb_dst.save(DST)
print(f"\nSaved → {DST}")
