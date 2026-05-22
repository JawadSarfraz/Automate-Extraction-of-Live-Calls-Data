"""Fair / Simple-VM / Smart-VM classifier v1 for one day of recordings.

Pipeline:
  1. Load recordings + calls JSON, join each WAV to its call record (weak label).
  2. Extract acoustic features from every WAV.
  3. Map the platform category -> our taxonomy (weak ground truth).
  4. Run an *audio-only* heuristic classifier (the seed of a real-time detector).
  5. Write a classified CSV/JSON, a validation report, and a review HTML player.

Usage:
  .venv/bin/python ml/classify_v1.py --date 2026-05-14 [--limit N]
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from features import extract  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# Platform category -> our taxonomy (the v1 weak-label mapping; tweak freely).
CATEGORY_MAP = {
    "Answering Machine": "simple_vm",
    "Honeypot": "smart_vm",
    "Qualified": "fair", "Not Interested": "fair", "DNC": "fair", "DNQ": "fair",
    "Call Back": "fair", "Neutral": "fair", "Unclear Response": "fair",
    "User Silent": "no_contact", "DAIR": "no_contact", "Inaudible": "no_contact",
    "User Hangup": "no_contact",
}


def ts_to_sec(timestamp: str) -> int:
    # "05/14/2026, 20:00:50" -> seconds since midnight
    try:
        hms = timestamp.split(", ")[1]
        h, m, s = (int(p) for p in hms.split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return -10 ** 9


def rec_sec(t: str) -> int:
    h, m, s = (int(p) for p in t.split(":"))
    return h * 3600 + m * 60 + s


def build_index(calls):
    idx = defaultdict(list)
    for c in calls:
        phone = str(c.get("number", "")).lstrip("+")
        if len(phone) == 11 and phone.startswith("1"):
            phone = phone[1:]
        idx[phone].append(c)
    return idx


def join_call(rec, idx):
    phone = rec["phone_number"]
    rs = rec_sec(rec["time"])
    best, best_d = None, 10 ** 9
    for c in idx.get(phone, []):
        d = abs(ts_to_sec(c["timestamp"]) - rs)
        if d < best_d:
            best, best_d = c, d
    if best is not None and best_d <= 90:
        return best, best_d
    return None, None


def audio_predict(f: dict) -> tuple[str, str]:
    """Audio-ONLY heuristic -> (label, reason). Independent of the platform label."""
    if f["duration"] < 3 or f["speech_ratio"] < 0.12:
        return "no_contact", f"near-silent (dur={f['duration']}s, speech={f['speech_ratio']})"
    if f["beep"]:
        return "simple_vm", f"answering-machine beep ({f['beep_score']}s tone)"
    if f["longest_monologue"] >= 6 and f["n_speech_segments"] <= 3:
        return "simple_vm", f"long monologue {f['longest_monologue']}s, {f['n_speech_segments']} segment(s) (recording)"
    if f["duration"] >= 90 and f["n_speech_segments"] >= 6:
        return "smart_vm", f"prolonged turn-taking (dur={f['duration']}s, {f['n_speech_segments']} segments)"
    if f["n_speech_segments"] >= 3 and f["speech_ratio"] > 0.20:
        return "fair", f"conversational turn-taking ({f['n_speech_segments']} segments, speech={f['speech_ratio']})"
    return "uncertain", f"no strong signal (dur={f['duration']}s, segs={f['n_speech_segments']})"


def feat_for_path(args):
    path, key = args
    try:
        return key, extract(path)
    except Exception as e:  # noqa: BLE001
        return key, {"error": str(e), "duration": 0.0, "speech_ratio": 0.0,
                     "n_speech_segments": 0, "longest_monologue": 0.0, "longest_silence": 0.0,
                     "beep": False, "beep_score": 0.0, "spectral_flatness": 0.0,
                     "bg_ratio": 0.0, "centroid_hz": 0.0, "rms_db": -99.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-14")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    date = args.date

    recs = json.load(open(os.path.join(DATA, f"recordings_{date}.json")))["recordings"]
    calls = json.load(open(os.path.join(DATA, f"calls_{date}.json")))["calls"]
    if args.limit:
        recs = recs[: args.limit]
    idx = build_index(calls)
    audio_dir = os.path.join(DATA, "audio", date)

    # Build work list: (wav_path, recording)
    work = []
    for r in recs:
        fname = os.path.basename(r["file_url"].split("?")[0])
        path = os.path.join(audio_dir, fname)
        if os.path.exists(path):
            work.append((path, fname, r))

    print(f"[v1] {date}: {len(work)} WAVs to analyze (of {len(recs)} recordings)")
    feats = {}
    paths = [(p, fn) for p, fn, _ in work]
    done = 0
    with ProcessPoolExecutor() as ex:
        for key, f in ex.map(feat_for_path, paths, chunksize=16):
            feats[key] = f
            done += 1
            if done % 200 == 0 or done == len(paths):
                print(f"  features {done}/{len(paths)}")

    rows = []
    matched = 0
    for path, fname, r in work:
        f = feats[fname]
        call, dt = join_call(r, idx)
        category = call["category"] if call else ""
        weak = CATEGORY_MAP.get(category, "unknown") if call else "unmatched"
        if call:
            matched += 1
        pred, reason = audio_predict(f)
        rows.append({
            "phone": r["phone_number"],
            "rec_time": r["time"],
            "call_ts": call["timestamp"] if call else "",
            "join_dt_s": dt if dt is not None else "",
            "category": category,
            "snippet": (call["transcription"] if call else "")[:120],
            "transferred": call["transferred"] if call else "",
            "weak_label": weak,
            "audio_pred": pred,
            "audio_reason": reason,
            "duration": f["duration"], "speech_ratio": f["speech_ratio"],
            "n_segments": f["n_speech_segments"], "longest_monologue": f["longest_monologue"],
            "longest_silence": f["longest_silence"], "beep": f["beep"], "beep_score": f["beep_score"],
            "spectral_flatness": f["spectral_flatness"], "bg_ratio": f["bg_ratio"],
            "centroid_hz": f["centroid_hz"], "rms_db": f["rms_db"], "file": fname,
        })

    out_dir = os.path.join(DATA, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"classified_{date}.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    with open(os.path.join(out_dir, f"classified_{date}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- Validation report ----
    print("\n" + "=" * 70)
    print(f"VALIDATION REPORT — {date}  ({len(rows)} recordings, {matched} joined to a call)")
    print("=" * 70)

    print("\nWeak-label distribution (from platform category):")
    for k, v in Counter(r["weak_label"] for r in rows).most_common():
        print(f"  {k:12} {v:5}  ({100*v/len(rows):.1f}%)")

    print("\nAudio-only prediction distribution:")
    for k, v in Counter(r["audio_pred"] for r in rows).most_common():
        print(f"  {k:12} {v:5}  ({100*v/len(rows):.1f}%)")

    print("\nConfusion: weak_label (rows) x audio_pred (cols)")
    labels = ["fair", "simple_vm", "smart_vm", "no_contact", "uncertain", "unknown"]
    cols = ["fair", "simple_vm", "smart_vm", "no_contact", "uncertain"]
    conf = defaultdict(Counter)
    for r in rows:
        conf[r["weak_label"]][r["audio_pred"]] += 1
    header = "  " + "weak\\audio".ljust(12) + "".join(c[:9].rjust(11) for c in cols)
    print(header)
    for lab in labels:
        if sum(conf[lab].values()) == 0:
            continue
        line = "  " + lab.ljust(12) + "".join(str(conf[lab][c]).rjust(11) for c in cols)
        print(line)

    # Key signal validation: does the beep actually flag Answering Machines?
    am = [r for r in rows if r["weak_label"] == "simple_vm"]
    if am:
        beep_hit = sum(1 for r in am if r["beep"])
        mono = sum(1 for r in am if r["longest_monologue"] >= 6)
        print(f"\nSimple-VM (Answering Machine) audio signature: "
              f"beep in {beep_hit}/{len(am)} ({100*beep_hit/len(am):.0f}%), "
              f"long-monologue in {mono}/{len(am)} ({100*mono/len(am):.0f}%)")
    smart = [r for r in rows if r["weak_label"] == "smart_vm"]
    if smart:
        import statistics
        durs = [r["duration"] for r in smart]
        print(f"Smart-VM (Honeypot) durations: median={statistics.median(durs):.0f}s "
              f"max={max(durs):.0f}s  vs  all-call median="
              f"{statistics.median([r['duration'] for r in rows]):.0f}s")

    print(f"\nWrote: data/analysis/classified_{date}.csv / .json")
    return rows, date


if __name__ == "__main__":
    main()
