"""Dummy audio packet sender — stream a .wav as real UDP packets, paced like a call.

Stands in for the dialer/bot media feed: chunks the wav into ~20 ms frames and
sends them over UDP in (near) real time, ending with an end-of-stream packet.

Usage (two-process):
  .venv/bin/python -m streaming.packet_sender --wav data/audio/2026-05-14/<file>.wav --port 9401
"""
from __future__ import annotations
import argparse
import socket
import time

import numpy as np
from scipy.io import wavfile

from .packet_io import encode_frame, eos_frame


def _to_int16(raw):
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    if np.issubdtype(raw.dtype, np.floating):
        return (np.clip(raw, -1, 1) * 32767).astype("<i2")
    return raw.astype("<i2")


def stream_wav(path: str, host: str = "127.0.0.1", port: int = 9401,
               frame_ms: int = 20, speed: float = 1.0, sock=None) -> dict:
    """Send `path` as UDP packets to host:port. Returns a small summary dict."""
    sr, raw = wavfile.read(path)
    samples = _to_int16(raw)
    frame = max(1, int(sr * frame_ms / 1000))
    own = sock is None
    if own:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dt = (frame_ms / 1000.0) / max(speed, 1e-9)
    t0 = time.time()
    seq = 0
    for i in range(0, len(samples), frame):
        sock.sendto(encode_frame(seq, samples[i:i + frame], sr), (host, port))
        seq += 1
        target = t0 + seq * dt
        slp = target - time.time()
        if slp > 0:
            time.sleep(slp)
    sock.sendto(eos_frame(seq, sr), (host, port))
    if own:
        sock.close()
    return {"frames": seq, "sr": int(sr), "seconds": round(len(samples) / sr, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9401)
    ap.add_argument("--frame-ms", type=int, default=20)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time; >1 faster")
    args = ap.parse_args()
    info = stream_wav(args.wav, args.host, args.port, args.frame_ms, args.speed)
    print(f"[sender] sent {info['frames']} frames ({info['seconds']}s @ {info['sr']}Hz) "
          f"to {args.host}:{args.port}")


if __name__ == "__main__":
    main()
