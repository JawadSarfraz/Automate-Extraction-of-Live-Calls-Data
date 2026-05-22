"""Classifier v2 — content-based Fair / Simple-VM / Smart-VM, on customer words.

Pipeline:
  1. Learn the bot script fingerprint, isolate each call's CUSTOMER words.
  2. Run transparent content detectors (voicemail/IVR, abuse, stalling-bot,
     looping, genuine engagement) on the customer text.
  3. Combine with acoustic features (from v1) -> class + confidence + reasons.
  4. Validate against the platform category.

Taxonomy follows the USER's definitions:
  * Simple VM = automated voicemail/IVR message  OR  abusive bashing
  * Smart VM  = another bot acting like a customer / stalling / prolonging
  * Fair      = genuine human conversation
  * no_contact= no/empty customer response

Usage: .venv/bin/python ml/classify_v2.py --date 2026-05-14
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from separate import learn_bot_bank, customer_words  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

VOICEMAIL = [
    "leave a message", "leave your message", "after the beep", "after the tone", "at the tone",
    "press 1", "press one", "press 2", "press 3", "press pound", "not available", "is unavailable",
    "your call has been", "has been forwarded", "voicemail", "voice mail", "mailbox",
    "record your message", "finished your message", "more options", "please hang up", "google voice",
    "please leave", "reached the", "the person you are trying", "automated", "this call may be recorded",
    "return your call", "we will return", "we'll return", "return your", "you can also reach",
    "reach me at", "as soon as we can", "your message after", "leave a name", "leave your name",
    "name and number", "at the sound of",
]
ABUSE = [
    "fuck", "fucking", "cunt", "bitch", "asshole", "ass hole", "bastard", "dick", "prick",
    "shut up", "scam", "scammer", "go to hell", "piss off", "son of a", "motherf",
]
STALL = [
    "what can i do for you", "where are you from", "who is this", "who's this", "whos this",
    "can you repeat", "repeat that", "say that again", "i can't hear you", "i cannot hear you",
    "can't hear you", "what company", "what do you want", "who are you", "what's your name",
    "are you a robot", "is this a robot", "speak up", "breaking up", "didn't catch",
    "come again", "pardon", "hello hello hello", "are you there", "still there",
    "how can i help", "how may i help",
]
ENGAGE = [
    "not interested", "have a plan", "already have", "take me off", "do not call", "don't call",
    "remove me", "no thank you", "i'm covered", "already covered", "have insurance",
    "appreciate your time", "have a good", "have a wonderful", "i'm fine",
]
AGE_RE = re.compile(r"\b(4[0-9]|5[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9]|10[0-9])\b")


def hits(text, phrases):
    return [p for p in phrases if p in text]


def max_repeat(text):
    """Largest count of any repeated word-trigram (loop/stall detector)."""
    toks = text.split()
    if len(toks) < 6:
        return 1, ""
    tri = Counter(tuple(toks[i:i + 3]) for i in range(len(toks) - 2))
    g, c = tri.most_common(1)[0]
    return c, " ".join(g)


def classify(cust, feats):
    """Return (label, confidence, reasons[])."""
    cust = cust.strip()
    wc = len(cust.split())
    reasons = []

    vm = hits(cust, VOICEMAIL)
    ab = hits(cust, ABUSE)
    st = hits(cust, STALL)
    en = hits(cust, ENGAGE)
    rep_c, rep_g = max_repeat(cust)
    age = bool(AGE_RE.search(cust))

    # no contact: essentially no customer words (just greeting / empty)
    if wc <= 2 and not (vm or ab or st):
        return "no_contact", 0.7, [f"no customer speech (cust='{cust}')"]

    # Simple VM: automated message / IVR
    if vm:
        reasons.append("voicemail/IVR phrase: " + ", ".join(vm[:3]))
        return "simple_vm", 0.85, reasons
    # Simple VM: abusive bashing
    if ab:
        reasons.append("abuse: " + ", ".join(ab[:3]))
        return "simple_vm", 0.8, reasons

    # Genuine engagement (age stated, "have a plan", "not interested") marks a real
    # human — this WINS over a lone stall phrase, because real people also ask
    # "who is this?" etc. Smart VM requires stalling WITHOUT genuine engagement.
    strong_fair = bool(en or age)

    if st and not strong_fair:
        reasons.append("stalling/honeypot phrase: " + ", ".join(st[:3]))
        conf = 0.8 if (feats.get("n_segments", 0) >= 6 or feats.get("duration", 0) >= 30) else 0.6
        return "smart_vm", conf, reasons
    if rep_c >= 3 and not strong_fair:
        reasons.append(f"looping x{rep_c}: '{rep_g}'")
        return "smart_vm", 0.6, reasons

    if strong_fair:
        if en:
            reasons.append("engagement: " + ", ".join(en[:3]))
        if age:
            reasons.append("states age")
        if st:
            reasons.append("(also asked: " + st[0] + ", but engaged -> fair)")
        return "fair", 0.75, reasons

    # Some customer speech but no strong cue
    if wc >= 5:
        return "fair", 0.45, [f"customer spoke {wc} words, no VM/abuse/stall cue"]
    return "uncertain", 0.3, [f"weak signal (cust='{cust[:60]}')"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-14")
    args = ap.parse_args()
    date = args.date

    v1 = {r["file"]: r for r in json.load(open(os.path.join(DATA, "analysis", f"classified_{date}.json")))}
    tdir = os.path.join(DATA, "transcripts", date)
    bank, n_docs, _ = learn_bot_bank(tdir)
    print(f"[v2] bot bank from {n_docs} transcripts: {len(bank)} n-grams")

    rows = []
    for tf in sorted(os.listdir(tdir)):
        if not tf.endswith(".json"):
            continue
        t = json.load(open(os.path.join(tdir, tf)))
        wav = t["file"]
        cust = customer_words(t.get("text", ""), bank)
        feats = v1.get(wav, {})
        label, conf, reasons = classify(cust, feats)
        rows.append({
            "file": wav,
            "category": feats.get("category", ""),
            "weak_label": feats.get("weak_label", ""),
            "v2_label": label,
            "v2_conf": conf,
            "v2_reasons": " | ".join(reasons),
            "customer_text": cust[:200],
            "full_text": (t.get("text", "") or "")[:300],
            "duration": feats.get("duration", t.get("audio_duration", "")),
            "n_segments": feats.get("n_segments", ""),
        })

    out_dir = os.path.join(DATA, "analysis")
    with open(os.path.join(out_dir, f"classified_v2_{date}.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    with open(os.path.join(out_dir, f"classified_v2_{date}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nv2 label distribution ({len(rows)} calls):")
    for k, v in Counter(r["v2_label"] for r in rows).most_common():
        print(f"  {k:12} {v:5} ({100*v/len(rows):.1f}%)")

    print("\nConfusion: platform weak_label (rows) x v2_label (cols)")
    cols = ["fair", "simple_vm", "smart_vm", "no_contact", "uncertain"]
    conf = defaultdict(Counter)
    for r in rows:
        conf[r["weak_label"]][r["v2_label"]] += 1
    print("  " + "weak\\v2".ljust(12) + "".join(c[:9].rjust(11) for c in cols))
    for lab in ["fair", "simple_vm", "smart_vm", "no_contact", "unknown", "unmatched", ""]:
        if sum(conf[lab].values()) == 0:
            continue
        print("  " + (lab or "(none)").ljust(12) + "".join(str(conf[lab][c]).rjust(11) for c in cols))

    print(f"\nWrote data/analysis/classified_v2_{date}.csv / .json")


if __name__ == "__main__":
    main()
