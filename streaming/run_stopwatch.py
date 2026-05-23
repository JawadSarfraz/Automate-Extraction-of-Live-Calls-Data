"""Run the streaming stopwatch on a day's calls (oracle ASR, v2 brain).

Loads the saved word-timestamped transcripts and replays each as a live prefix; the
sealed answer key is the platform weak label from classify_v1's output. The detector
is blind; we only score afterwards.

Usage:
  .venv/bin/python -m streaming.run_stopwatch --date 2026-05-14
  .venv/bin/python -m streaming.run_stopwatch --date 2026-05-14 --step 0.5 --tau 0.6 --k 2 --limit 300
"""
from __future__ import annotations
import argparse
import json
import os
import time

from . import _paths
from .sources import OracleTranscriptSource
from .classifiers import V2RulesClassifier
from .trigger import ConfidenceTrigger
from .detector import StreamingDetector
from .harness import Stopwatch
from .report import aggregate, print_report, write_outputs, CLASSES

from separate import learn_bot_bank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-14")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--classifier", default="v2", choices=["v2"])
    args = ap.parse_args()

    tdir = os.path.join(_paths.DATA, "transcripts", args.date)
    v1_path = os.path.join(_paths.DATA, "analysis", f"classified_{args.date}.json")
    if not os.path.isdir(tdir):
        raise SystemExit(f"no transcripts at {tdir} — run ml/transcribe.py first")
    if not os.path.exists(v1_path):
        raise SystemExit(f"no {v1_path} — run ml/classify_v1.py first (it has the weak labels)")

    weak = {r["file"]: r.get("weak_label", "") for r in json.load(open(v1_path))}

    print(f"[stopwatch] learning bot bank from {tdir} ...")
    bank, n_docs, _ = learn_bot_bank(tdir)
    print(f"[stopwatch] bot bank: {len(bank)} n-grams from {n_docs} transcripts")

    classifier = V2RulesClassifier()
    detector = StreamingDetector(
        classifier,
        trigger_factory=lambda: ConfidenceTrigger(tau=args.tau, k=args.k),
    )
    sw = Stopwatch(detector)

    files = sorted(f for f in os.listdir(tdir) if f.endswith(".json"))
    records = []
    skipped = 0
    t0 = time.time()
    for fn in files:
        tj = json.load(open(os.path.join(tdir, fn)))
        true_class = weak.get(tj.get("file", ""), "")
        if true_class not in CLASSES:   # only score calls with a usable sealed label
            skipped += 1
            continue
        src = OracleTranscriptSource(tj, bank, step=args.step)
        records.append(sw.grade(src, true_class))
        if args.limit and len(records) >= args.limit:
            break

    cfg = {"date": args.date, "classifier": args.classifier, "step": args.step,
           "tau": args.tau, "k": args.k}
    agg = aggregate(records)
    print_report(records, agg, cfg)
    print(f"\n[stopwatch] graded {len(records)} calls ({skipped} skipped: no usable label) "
          f"in {time.time()-t0:.1f}s")
    csv_path, json_path = write_outputs(records, agg, cfg, os.path.join(_paths.DATA, "analysis"))
    print(f"[stopwatch] wrote {os.path.relpath(csv_path, _paths.ROOT)} and "
          f"{os.path.relpath(json_path, _paths.ROOT)}")


if __name__ == "__main__":
    main()
