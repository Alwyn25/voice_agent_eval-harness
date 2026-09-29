"""
Build a small, latency-stratified, tail-preserving regression set from an
already-completed eval run, so future mini/full eval runs only re-spend
credits on a fixed set of scenarios that are known to span the whole
latency spectrum -- instead of `records[:n]` (first N rows of the dataset,
which is what mini_eval.py currently does), which happens to never include
any scenario from the 4-6s / 6-8s / >8s buckets.

Zero API calls / zero credits: this only re-groups rows that were already
paid for in prior runs (jsonl or golden_eval_final.xlsx reports). A
scenario's bucket is the WORST wall_ms seen for it across every source
given, so a case that was merely slow once still counts as a risk.

Usage:
    python build_stratified_subset.py \
        --budget 48 \
        --master-dataset "C:/Users/.../highland_greenz_golden_dataset.xlsx" \
        --out results/stratified_regression_set.json
    # (--sources defaults to the known-good real-backend runs; see DEFAULT_SOURCES)

The output JSON is a flat list of {type, id, bucket, ...} you can filter
golden_subset_100st_40th.xlsx / golden_balanced_200st_40th.xlsx against
(by `id` for Golden_SingleTurn rows, by `thread_id` for Golden_Threads rows)
instead of taking the first N rows.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import openpyxl

BUCKETS = [
    (0, 1000, "<1s"),
    (1000, 2000, "1-2s"),
    (2000, 3000, "2-3s"),
    (3000, 4000, "3-4s"),
    (4000, 6000, "4-6s"),
    (6000, 8000, "6-8s"),
    (8000, float("inf"), ">8s"),
]

# Rarer / higher-signal buckets get more of the budget; the dominant
# fast buckets get just enough to confirm nothing regressed there.
# Actual counts are capped by each bucket's real population.
BUCKET_WEIGHT = {
    "<1s": 1.0,
    "1-2s": 1.0,
    "2-3s": 1.0,
    "3-4s": 1.0,
    "4-6s": 1.5,
    "6-8s": 1.5,
    ">8s": 1.5,
}

# Below this population, take every scenario in the bucket (it's already
# small enough that sampling it further throws away free coverage).
CENSUS_THRESHOLD = 6


def bucket_of(ms: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= ms < hi:
            return name
    return "?"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _xlsx_sheet_to_dicts(ws) -> list[dict]:
    rows_iter = ws.iter_rows(values_only=True)
    header = list(next(rows_iter))
    return [dict(zip(header, row)) for row in rows_iter]


def load_xlsx_flat(path: Path, sheets=("SingleTurn_All", "Thread_All")) -> list[dict]:
    """Schema where every sheet already has test_type/thread_id/row_id/
    wall_ms/accuracy_score columns (the *_All sheets in golden_eval_final.xlsx
    reports #2-4 share this shape)."""
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = []
    for sheet in sheets:
        if sheet in wb.sheetnames:
            rows.extend(_xlsx_sheet_to_dicts(wb[sheet]))
    return rows


def load_xlsx_split(path: Path, st_sheet="SingleTurn_Results", th_sheet="Thread_Results") -> list[dict]:
    """Schema where single-turn and thread rows live in separate sheets with
    row_id / thread_id respectively and no test_type column (report #1's shape)."""
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = []
    if st_sheet in wb.sheetnames:
        for r in _xlsx_sheet_to_dicts(wb[st_sheet]):
            r["test_type"] = "st"
            rows.append(r)
    if th_sheet in wb.sheetnames:
        for r in _xlsx_sheet_to_dicts(wb[th_sheet]):
            r["test_type"] = "th"
            rows.append(r)
    return rows


def load_source(path: Path) -> list[dict]:
    """Auto-detect jsonl vs either xlsx report shape and return normalized
    turn-level dicts with at least: test_type, row_id/thread_id, wall_ms,
    accuracy_score, lang, difficulty, error."""
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    names = set(wb.sheetnames)
    wb.close()
    if "SingleTurn_All" in names or "Thread_All" in names:
        return load_xlsx_flat(path)
    if "SingleTurn_Results" in names or "Thread_Results" in names:
        return load_xlsx_split(path)
    raise ValueError(f"{path}: unrecognized sheet layout {sorted(names)}")


def build_scenarios(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Group turn-level rows into scenario-level records.

    A scenario's latency class is its WORST turn (max wall_ms) -- that's
    the turn a latency regression check actually needs to catch.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("test_type") == "th":
            key = ("th", r.get("thread_id") or "")
        else:
            key = ("st", r.get("row_id") or "")
        if not key[1]:
            continue
        groups[key].append(r)

    scenarios = {}
    for key, turns in groups.items():
        wall = [t.get("wall_ms", 0) or 0 for t in turns]
        acc = [t.get("accuracy_score") for t in turns if isinstance(t.get("accuracy_score"), (int, float))]
        first = turns[0]
        scenarios[key] = {
            "type": key[0],
            "id": key[1],
            "n_turns": len(turns),
            "max_wall_ms": max(wall) if wall else 0,
            "mean_wall_ms": statistics.mean(wall) if wall else 0,
            "min_accuracy": min(acc) if acc else None,
            "mean_accuracy": statistics.mean(acc) if acc else None,
            "has_error": any(t.get("error") for t in turns),
            "lang": first.get("lang"),
            "difficulty": next((t.get("difficulty") for t in turns if t.get("difficulty")), ""),
            "tool_category": next((t.get("tool_category") for t in turns if t.get("tool_category")), ""),
            "bucket": bucket_of(max(wall) if wall else 0),
        }
    return scenarios


def allocate_budget(bucket_pops: dict[str, int], budget: int) -> dict[str, int]:
    """Split `budget` across buckets, weighted, capped by real population,
    guaranteeing a census of any bucket at/under CENSUS_THRESHOLD."""
    target = {}
    remaining_budget = budget
    remaining_buckets = []

    for _, _, name in BUCKETS:
        pop = bucket_pops.get(name, 0)
        if pop == 0:
            target[name] = 0
        elif pop <= CENSUS_THRESHOLD:
            target[name] = pop
            remaining_budget -= pop
        else:
            remaining_buckets.append(name)

    if remaining_buckets and remaining_budget > 0:
        total_weight = sum(BUCKET_WEIGHT[n] for n in remaining_buckets)
        for name in remaining_buckets:
            share = round(remaining_budget * BUCKET_WEIGHT[name] / total_weight)
            target[name] = min(share, bucket_pops[name])

    return target


def pick_from_bucket(candidates: list[dict], k: int) -> list[dict]:
    """Prioritize the most diagnostic scenarios first: errors, then low
    accuracy, then spread across language so the sample isn't all-English."""
    def sort_key(s):
        return (
            0 if s["has_error"] else 1,
            s["mean_accuracy"] if s["mean_accuracy"] is not None else 999,
        )

    ranked = sorted(candidates, key=sort_key)
    if k >= len(ranked):
        return ranked

    # Round-robin by language over the accuracy-ranked list so the top-k
    # isn't dominated by whichever language happens to sort first.
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for s in ranked:
        by_lang[s["lang"]].append(s)
    langs = list(by_lang.keys())

    picked, i = [], 0
    while len(picked) < k:
        lang = langs[i % len(langs)]
        if by_lang[lang]:
            picked.append(by_lang[lang].pop(0))
        i += 1
        if all(not v for v in by_lang.values()):
            break
    return picked


def write_dataset_xlsx(master_path: Path, selected: list[dict], out_path: Path) -> None:
    """Write a Golden_SingleTurn / Golden_Threads workbook containing only
    the selected scenarios, in the same shape as the master dataset, so it
    can be pointed to directly via `mini_eval.py --dataset <this file>`
    with no code changes to the eval harness."""
    st_ids = {s["id"] for s in selected if s["type"] == "st"}
    th_ids = {s["id"] for s in selected if s["type"] == "th"}

    src = openpyxl.load_workbook(master_path, read_only=True, data_only=True)
    out = openpyxl.Workbook()
    out.remove(out.active)

    def copy_filtered(sheet_name: str, id_col: str, keep_ids: set):
        src_ws = src[sheet_name]
        rows_iter = src_ws.iter_rows(values_only=True)
        header = list(next(rows_iter))
        id_idx = header.index(id_col)
        dst_ws = out.create_sheet(sheet_name)
        dst_ws.append(header)
        n = 0
        for row in rows_iter:
            if row[id_idx] in keep_ids:
                dst_ws.append(list(row))
                n += 1
        return n

    n_st = copy_filtered("Golden_SingleTurn", "id", st_ids) if st_ids else 0
    n_th = copy_filtered("Golden_Threads", "thread_id", th_ids) if th_ids else 0

    cov = out.create_sheet("Coverage")
    cov.append(["Metric", "Value"])
    cov.append(["Single-turn rows", n_st])
    cov.append(["Thread rows (turns)", n_th])
    cov.append(["Source master dataset", str(master_path)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"Wrote {out_path}  (Golden_SingleTurn={n_st} rows, Golden_Threads={n_th} rows)")


# Real per-turn, real-backend runs with genuine latency/accuracy variance,
# oldest to newest. backup/4 (== results/golden_eval_final.xlsx) is
# deliberately excluded: 97% of its rows land in a single 1-2s bucket and
# its accuracy_score averages 7.6 with negative values, both signs of a
# mock/offline backend or a broken scorer on that run rather than real
# production behavior -- including it would contaminate the latency signal.
DEFAULT_SOURCES = [
    "results/backup/1/golden_eval_final.xlsx",
    "results/backup/2/golden_eval_final.xlsx",
    "results/backup/3/golden_eval_final.xlsx",
    "results/full/full_inbound_full_v2.jsonl",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                     help="one or more prior-run files (.jsonl or golden_eval_final.xlsx, either sheet "
                          "layout) to merge; a scenario's bucket is its WORST wall_ms seen across all of them")
    ap.add_argument("--budget", type=int, default=48, help="target total scenario count")
    ap.add_argument("--out", default="results/stratified_regression_set.json")
    ap.add_argument("--master-dataset", default=None,
                     help="master golden dataset xlsx (e.g. highland_greenz_golden_dataset.xlsx) "
                          "to slice into a ready-to-run --dataset-out workbook")
    ap.add_argument("--dataset-out", default="results/stratified_regression_set.xlsx",
                     help="where to write the filtered workbook (only used with --master-dataset)")
    args = ap.parse_args()

    rows = []
    print("Sources:")
    for s in args.sources:
        p = Path(s)
        src_rows = load_source(p)
        for r in src_rows:
            r["_source"] = p.name
        rows.extend(src_rows)
        print(f"  {p}  (+{len(src_rows)} rows)")
    scenarios = build_scenarios(rows)

    bucket_pops = defaultdict(int)
    by_bucket = defaultdict(list)
    for s in scenarios.values():
        bucket_pops[s["bucket"]] += 1
        by_bucket[s["bucket"]].append(s)

    targets = allocate_budget(bucket_pops, args.budget)

    selected = []
    print(f"\nMerged: {len(rows)} rows across {len(args.sources)} sources -> {len(scenarios)} unique scenarios\n")
    print(f"{'bucket':6s} {'population':>10s} {'selected':>9s} {'avg_acc(sel)':>13s} {'langs(sel)'}")
    for _, _, name in BUCKETS:
        pop = bucket_pops.get(name, 0)
        k = targets.get(name, 0)
        picked = pick_from_bucket(by_bucket.get(name, []), k)
        selected.extend(picked)
        accs = [s["mean_accuracy"] for s in picked if s["mean_accuracy"] is not None]
        avg_acc = f"{statistics.mean(accs):.1f}" if accs else "-"
        langs = sorted({s["lang"] for s in picked})
        print(f"{name:6s} {pop:>10d} {len(picked):>9d} {avg_acc:>13s}  {','.join(langs)}")

    print(f"\nTotal selected: {len(selected)} / budget {args.budget}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sources": args.sources,
                "budget": args.budget,
                "generated_scenarios": len(scenarios),
                "selected": selected,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {out_path}")

    if args.master_dataset:
        write_dataset_xlsx(Path(args.master_dataset), selected, Path(args.dataset_out))


if __name__ == "__main__":
    main()
