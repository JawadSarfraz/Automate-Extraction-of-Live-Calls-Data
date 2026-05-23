"""Aggregate CallRecords into the stopwatch report (STREAMING_DESIGN.md §7).

Note on no_contact: it is not a "committal" class, so the detector never flags it —
for no_contact the meaningful number is the FALSE-flag rate (how often a silent call
was wrongly flagged as fair/VM/bot), not a catch rate.
"""
from __future__ import annotations
import csv
import json
import os
from collections import Counter
from dataclasses import asdict

CLASSES = ["fair", "simple_vm", "smart_vm", "no_contact"]
VM_BOT = ("simple_vm", "smart_vm")


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _median(vals):
    return _pct(sorted(vals), 0.5)


def aggregate(records: list) -> dict:
    by_true = {c: [r for r in records if r.true_class == c] for c in CLASSES}
    out = {"n": len(records), "per_class": {}}
    for c in CLASSES:
        rs = by_true[c]
        if not rs:
            continue
        fired = [r for r in rs if r.fired]
        correct = [r for r in rs if r.correct]
        ft = sorted(r.fire_time for r in correct if r.fire_time is not None)
        leads = sorted(r.lead_time for r in correct)
        out["per_class"][c] = {
            "n": len(rs),
            "n_fired": len(fired),
            "n_correct": len(correct),
            "flag_rate": round(len(fired) / len(rs), 3),
            "catch_rate": round(len(correct) / len(rs), 3),
            "accuracy_at_flag": round(len(correct) / len(fired), 3) if fired else None,
            "median_fire_time_s": round(_median(ft), 2) if ft else None,
            "p90_fire_time_s": round(_pct(ft, 0.9), 2) if ft else None,
            "median_lead_time_s": round(_median(leads), 2) if leads else None,
            "median_call_len_s": round(_median([r.call_len for r in rs]), 2),
            "fired_as": dict(Counter(r.fired_class for r in fired if r.fired_class)),
        }
    fair = by_true.get("fair", [])
    ff = [r for r in fair if r.false_flag]
    ff_times = sorted(r.fire_time for r in ff if r.fire_time is not None)
    out["fair_safety"] = {
        "n_fair": len(fair),
        "false_flags": len(ff),
        "false_flag_rate": round(len(ff) / len(fair), 4) if fair else None,
        "earliest_false_flag_s": round(ff_times[0], 2) if ff_times else None,
        "median_false_flag_s": round(_median(ff_times), 2) if ff_times else None,
    }
    return out


def print_report(records: list, agg: dict, cfg: dict) -> None:
    print("\n" + "=" * 78)
    print(f"STREAMING STOPWATCH — {cfg.get('date','?')}   "
          f"(classifier={cfg.get('classifier')}, step={cfg.get('step')}s, "
          f"tau={cfg.get('tau')}, k={cfg.get('k')})")
    print(f"graded {agg['n']} calls | detector is BLIND; labels opened only here")
    print("=" * 78)
    hdr = (f"{'true class':11} {'n':>5} {'flag%':>6} {'correct%':>9} {'acc@flag':>9} "
           f"{'med t':>7} {'p90 t':>7} {'saved':>7} {'len':>6}")
    print(hdr)
    print("-" * len(hdr))
    for c in CLASSES:
        s = agg["per_class"].get(c)
        if not s:
            continue
        def f(x, suf=""):
            return "-" if x is None else f"{x}{suf}"
        print(f"{c:11} {s['n']:>5} {round(100*s['flag_rate'],1):>6} "
              f"{round(100*s['catch_rate'],1):>9} {f(s['accuracy_at_flag']):>9} "
              f"{f(s['median_fire_time_s'],'s'):>7} {f(s['p90_fire_time_s'],'s'):>7} "
              f"{f(s['median_lead_time_s'],'s'):>7} {f(s['median_call_len_s'],'s'):>6}")
    fs = agg["fair_safety"]
    print("-" * len(hdr))
    print(f"SAFETY — true Fair wrongly flagged as VM/bot: {fs['false_flags']}/{fs['n_fair']} "
          f"(rate {fs['false_flag_rate']}); earliest at {fs['earliest_false_flag_s']}s")
    print("legend: flag% = % of class the detector flagged (any committal class) | "
          "correct% = flagged as its TRUE class")
    print("        acc@flag = of flagged calls, fraction right | med t = median secs to "
          "flag | saved = call_len − flag time")
    print("        no_contact never fires by design → its flag% is the FALSE-positive rate.")


def write_outputs(records: list, agg: dict, cfg: dict, out_dir: str) -> tuple[str, str]:
    date = cfg.get("date", "run")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"stopwatch_{date}.csv")
    json_path = os.path.join(out_dir, f"stopwatch_{date}.json")
    if records:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()))
            w.writeheader()
            for r in records:
                w.writerow(asdict(r))
    with open(json_path, "w") as fh:
        json.dump({"config": cfg, "summary": agg}, fh, indent=2)
    return csv_path, json_path
