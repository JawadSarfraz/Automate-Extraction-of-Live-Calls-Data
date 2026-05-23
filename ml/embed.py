"""Embed each call's CUSTOMER words (bot script stripped) into a dense vector.

Reuses the v2 bot/customer separation (separate.py), then encodes the customer
residue with a sentence-transformer (default all-MiniLM-L6-v2, 384-d) on the GPU.
The result feeds the supervised v3 classifier alongside the v1 acoustic features.

Cached to data/analysis/cust_emb_<date>.npz (skip if present unless --force).
Defaults to the GPU (this box has a Tesla T4); pass --device cpu to force CPU.
Swap the embedding model with --model (e.g. all-mpnet-base-v2 for a quality bump).

Usage:
  .venv/bin/python ml/embed.py --date 2026-05-14
  .venv/bin/python ml/embed.py --date 2026-05-14 --model sentence-transformers/all-mpnet-base-v2 --force
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from separate import learn_bot_bank, customer_words  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _resolve_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        print("[embed] cuda requested but not available — falling back to cpu")
    except Exception as e:  # noqa: BLE001
        print(f"[embed] torch/cuda check failed ({e}) — falling back to cpu")
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-14")
    ap.add_argument("--device", default="cuda", help="cuda (default, this box has a T4) | cpu")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model id")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    date = args.date

    out_dir = os.path.join(DATA, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cust_emb_{date}.npz")
    if os.path.exists(out_path) and not args.force:
        print(f"[embed] {out_path} exists — use --force to recompute. Skipping.")
        return

    tdir = os.path.join(DATA, "transcripts", date)
    bank, n_docs, _ = learn_bot_bank(tdir)
    print(f"[embed] bot bank from {n_docs} transcripts: {len(bank)} n-grams")

    files, cust = [], []
    for tf in sorted(os.listdir(tdir)):
        if not tf.endswith(".json"):
            continue
        t = json.load(open(os.path.join(tdir, tf)))
        files.append(t["file"])
        cust.append(customer_words(t.get("text", ""), bank))

    n_empty = sum(1 for c in cust if not c.strip())
    device = _resolve_device(args.device)
    print(f"[embed] {len(files)} calls, {n_empty} with empty customer residue "
          f"| model={args.model} device={device}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=device)
    t0 = time.time()
    emb = model.encode(
        cust, batch_size=args.batch, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    print(f"[embed] encoded {len(cust)} residues in {time.time()-t0:.1f}s on {device}")

    np.savez_compressed(
        out_path,
        files=np.array(files, dtype=object),
        emb=emb,
        cust=np.array([c[:300] for c in cust], dtype=object),
        model=args.model,
    )
    print(f"[embed] wrote {out_path}  emb shape={emb.shape}")


if __name__ == "__main__":
    main()
