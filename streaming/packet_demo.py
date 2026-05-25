"""Packet-driven detection demo — prove Simple/Smart VM detection over real UDP.

Two modes:
  --selftest <wav...> : spin up the receiver and stream each wav to it over real
                        UDP (sender runs in a thread), printing the live per-second
                        decision and the moment it flags. Reuses one loaded model.
  --listen            : just the receiver+detector on a fixed port; stream to it
                        from another process with `python -m streaming.packet_sender`.

ASR defaults to CPU (no extra CUDA libs needed). For GPU pass --device cuda after
setting LD_LIBRARY_PATH to the venv's nvidia cublas/cudnn libs (see CLAUDE.md §8).

Usage:
  .venv/bin/python -m streaming.packet_demo --selftest <file.wav> --true-class simple_vm
  .venv/bin/python -m streaming.packet_demo --listen --port 9401
"""
from __future__ import annotations
import argparse
import os
import threading
import time

from . import _paths
from .packet_source import PacketAudioSource
from .packet_sender import stream_wav
from .sources import WhisperPrefixASR
from .classifiers import V2RulesClassifier
from .trigger import ConfidenceTrigger

from separate import learn_bot_bank  # noqa: E402

COMMIT = {"simple_vm", "smart_vm", "fair"}


def run_detect(source: PacketAudioSource, tau: float, k: int, true_class: str | None) -> dict:
    clf = V2RulesClassifier()
    trig = ConfidenceTrigger(tau=tau, k=k)
    fired_at = None
    fired_label = None
    print(f"  ... receiver listening on {source.host}:{source.port}")
    t_wall0 = None
    for obs in source.observations():
        if t_wall0 is None:
            t_wall0 = time.time()
        label, conf, _ = clf.classify(obs)
        print(f"    [audio t={obs.t:5.1f}s] {label:11} conf={conf:.2f}  "
              f"cust=\"{obs.customer_text[:55]}\"")
        if fired_at is None and trig.update(obs.t, label, conf):
            fired_at = trig.fired_t
            fired_label = trig.fired_label
            print(f"    🚩 FLAGGED → {fired_label}  at audio t={fired_at:.1f}s")
    if fired_at is None:
        print("    (no flag — fail-safe: keep talking)")
    verdict = ""
    if true_class:
        ok = fired_label == true_class
        verdict = f"  [true={true_class} -> {'✓ correct' if ok else ('✗ ' + str(fired_label))}]"
    print(f"  result: flagged={fired_label} at {fired_at}s  call_len={source.call_len}s{verdict}")
    return {"fired_label": fired_label, "fired_at": fired_at, "call_len": source.call_len}


def _resolve(path: str, date: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    cand = os.path.join(_paths.DATA, "audio", date, path)
    return cand if os.path.exists(cand) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", nargs="+", help="wav file(s) to stream over UDP and detect")
    ap.add_argument("--listen", action="store_true", help="receiver only (use external sender)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9401)
    ap.add_argument("--device", default="cpu", help="cpu (default) | cuda (needs LD_LIBRARY_PATH)")
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--speed", type=float, default=1.0, help="sender pace; 1.0 = real time")
    ap.add_argument("--date", default="2026-05-14", help="date whose transcripts seed the bot bank")
    ap.add_argument("--true-class", default=None)
    args = ap.parse_args()

    tdir = os.path.join(_paths.DATA, "transcripts", args.date)
    print(f"[demo] learning bot bank from {tdir} ...")
    bank, _, _ = learn_bot_bank(tdir)
    print(f"[demo] loading ASR ({args.model}, {args.device}) ...")
    asr = WhisperPrefixASR(model_name=args.model, device=args.device)

    if args.listen:
        src = PacketAudioSource(asr, bank, args.host, args.port, step=args.step).start()
        run_detect(src, args.tau, args.k, args.true_class)
        src.close()
        return

    if not args.selftest:
        raise SystemExit("pass --selftest <wav...> or --listen")

    for wav in args.selftest:
        path = _resolve(wav, args.date)
        if not os.path.exists(path):
            print(f"[demo] !! not found: {wav}")
            continue
        print(f"\n=== self-test: {os.path.basename(path)} ===")
        src = PacketAudioSource(asr, bank, args.host, 0, step=args.step).start()
        sender = threading.Thread(
            target=stream_wav, args=(path, args.host, src.port, 20, args.speed), daemon=True)
        sender.start()
        run_detect(src, args.tau, args.k, args.true_class)
        sender.join(timeout=2)
        src.close()


if __name__ == "__main__":
    main()
