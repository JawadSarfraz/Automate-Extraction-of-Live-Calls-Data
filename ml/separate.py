"""Bot-vs-customer separation via script fingerprinting.

Our outbound bot ("Kate"/"Hannah") reads a near-identical script on every call,
so its phrases dominate. We learn a "bot phrase bank" = word n-grams that recur
across many calls, then subtract bot-covered spans from each transcript to leave
the *customer's* words. No diarization model needed.
"""
from __future__ import annotations
import glob
import json
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NGRAM = 5

# Bot lines that recur but below the auto-DF threshold (qualifying / closing
# script). Injected into the bank so they get stripped as bot, not customer.
CURATED_BOT = [
    "looks like you qualify i'll bring a senior representative",
    "i'll bring a senior representative on the line",
    "i want to make sure your family is taken care of",
    "could you please confirm are you between the ages of 40 and 80",
    "are you between the ages of 40 and 80 so i can proceed",
    "based on what you've shared i want to make sure",
    "designed to help seniors leave less financial burden on their family",
    "state regulated final expense program",
    "it locks you in so the price never goes up",
    "to handle funeral costs so they don't fall on your family",
]


def norm(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def ngrams(tokens, n=NGRAM):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def learn_bot_bank(transcripts_dir: str, min_df_frac=0.08):
    """Return set of bot n-grams (appear in >= min_df_frac of calls)."""
    df = Counter()
    n_docs = 0
    for f in glob.glob(os.path.join(transcripts_dir, "*.json")):
        t = json.load(open(f))
        toks = norm(t.get("text", ""))
        if len(toks) < NGRAM:
            continue
        n_docs += 1
        for g in set(ngrams(toks)):
            df[g] += 1
    thr = max(3, int(min_df_frac * n_docs))
    bank = {g for g, c in df.items() if c >= thr}
    # Add curated bot lines so their spans are stripped even if below threshold.
    for line in CURATED_BOT:
        toks = norm(line)
        bank.update(ngrams(toks))
    return bank, n_docs, df


def customer_words(text: str, bank: set) -> str:
    """Mask tokens that fall inside any bot n-gram; return the leftover (customer) text."""
    toks = norm(text)
    masked = [False] * len(toks)
    for i in range(len(toks) - NGRAM + 1):
        if tuple(toks[i:i + NGRAM]) in bank:
            for j in range(i, i + NGRAM):
                masked[j] = True
    return " ".join(w for w, m in zip(toks, masked) if not m)


if __name__ == "__main__":
    tdir = os.path.join(DATA, "transcripts", "2026-05-14")
    bank, n_docs, df = learn_bot_bank(tdir)
    print(f"learned bot bank from {n_docs} transcripts: {len(bank)} n-grams")
    print("\nTop recurring (bot-script) 5-grams:")
    for g, c in df.most_common(15):
        print(f"  {c:3} ({100*c/n_docs:.0f}%)  {' '.join(g)}")

    # show customer remainder for a handful of calls
    print("\n=== customer remainder (bot script stripped) ===")
    for f in sorted(glob.glob(os.path.join(tdir, "*.json")))[:25]:
        t = json.load(open(f))
        cust = customer_words(t.get("text", ""), bank)
        if cust.strip():
            print(f"  cust: {cust[:140]}")
