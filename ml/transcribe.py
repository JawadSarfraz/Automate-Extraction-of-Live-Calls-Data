"""Transcribe a day's recordings with faster-whisper (on-device, M1-friendly).

Produces one JSON per call in data/transcripts/<date>/ with full text + segment
and word timestamps. Resumable (skips files already transcribed).

Telephony-specific care:
  * vad_filter=True  -> skip dead air (56% of our calls are near-silent)
  * condition_on_previous_text=False -> avoid Whisper's repetition/hallucination
    loops on noisy/silent lines
  * language="en" fixed -> no per-file language detection overhead

Usage:
  .venv/bin/python ml/transcribe.py --date 2026-05-14 --model base.en --workers 2
  .venv/bin/python ml/transcribe.py --date 2026-05-14 --limit 20   # quick test
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

_MODEL = None
_MODEL_NAME = None
_COMPUTE = None
_DEVICE = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        kwargs = dict(device=_DEVICE, compute_type=_COMPUTE)
        if _DEVICE == "cpu":
            kwargs["cpu_threads"] = 4
        _MODEL = WhisperModel(_MODEL_NAME, **kwargs)
    return _MODEL


def _init(model_name, compute, device):
    global _MODEL_NAME, _COMPUTE, _DEVICE
    _MODEL_NAME, _COMPUTE, _DEVICE = model_name, compute, device


def transcribe_one(args):
    path, out_path = args
    if os.path.exists(out_path):
        return ("skip", path, 0.0)
    t0 = time.time()
    try:
        model = _get_model()
        segments, info = model.transcribe(
            path, language="en", beam_size=1, vad_filter=True,
            condition_on_previous_text=False, word_timestamps=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        segs = []
        for s in segments:
            words = [{"s": round(w.start, 2), "e": round(w.end, 2), "w": w.word, "p": round(w.probability, 2)}
                     for w in (s.words or [])]
            segs.append({"s": round(s.start, 2), "e": round(s.end, 2), "text": s.text.strip(), "words": words})
        text = " ".join(s["text"] for s in segs).strip()
        out = {
            "file": os.path.basename(path),
            "text": text,
            "n_segments": len(segs),
            "n_words": sum(len(s["words"]) for s in segs),
            "audio_duration": round(info.duration, 2),
            "asr_model": _MODEL_NAME,
            "segments": segs,
        }
        with open(out_path, "w") as fh:
            json.dump(out, fh, ensure_ascii=False)
        return ("ok", path, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        with open(out_path, "w") as fh:
            json.dump({"file": os.path.basename(path), "error": str(e), "text": "", "segments": []}, fh)
        return ("fail:" + str(e)[:80], path, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-14")
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--device", default="cpu", help="cpu | cuda  (use cuda on a GPU box)")
    ap.add_argument("--compute", default="int8", help="int8 (cpu) | float16 (gpu)")
    ap.add_argument("--workers", type=int, default=2, help="parallel procs; use 1 on a single GPU")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    # Sensible GPU default: one worker, float16.
    if args.device == "cuda" and args.compute == "int8":
        args.compute = "float16"

    recs = json.load(open(os.path.join(DATA, f"recordings_{args.date}.json")))["recordings"]
    if args.limit:
        recs = recs[: args.limit]
    audio_dir = os.path.join(DATA, "audio", args.date)
    out_dir = os.path.join(DATA, "transcripts", args.date)
    os.makedirs(out_dir, exist_ok=True)

    work = []
    for r in recs:
        fn = os.path.basename(r["file_url"].split("?")[0])
        p = os.path.join(audio_dir, fn)
        if os.path.exists(p):
            work.append((p, os.path.join(out_dir, fn.replace(".wav", ".json"))))

    todo = [w for w in work if not os.path.exists(w[1])]
    print(f"[asr] {args.date}: {len(work)} files, {len(todo)} to transcribe "
          f"({len(work)-len(todo)} already done) | model={args.model} device={args.device} "
          f"compute={args.compute} workers={args.workers}")
    if not todo:
        print("[asr] nothing to do.")
        return

    t0 = time.time()
    done = ok = skip = fail = 0
    audio_secs = 0.0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.model, args.compute, args.device)) as ex:
        for status, path, dt in ex.map(transcribe_one, todo, chunksize=4):
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  ! {os.path.basename(path)}: {status}")
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(todo) - done) / rate if rate else 0
                print(f"  {done}/{len(todo)}  ok={ok} fail={fail}  {rate:.1f} files/s  ETA {eta/60:.1f} min")
    print(f"[asr] done in {(time.time()-t0)/60:.1f} min  ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
