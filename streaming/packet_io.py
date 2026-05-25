"""Tiny RTP-like packet codec for the dummy audio transport.

Each packet carries a sequence number, the sample rate, and a frame of 16-bit PCM
audio (mono). One bit flags end-of-stream. This is a deliberately minimal stand-in
for real RTP — enough to prove the packet -> buffer -> detector path end to end.

Wire format (network byte order header, little-endian PCM payload):
    seq:   uint32   monotonically increasing frame index
    sr:    uint32   sample rate (Hz)
    n:     uint16   number of int16 samples in this frame
    flags: uint8    bit0 = end-of-stream
    payload: n * int16  (little-endian)
"""
from __future__ import annotations
import struct

import numpy as np

_HEADER = struct.Struct("!IIHB")
FLAG_EOS = 1


def encode_frame(seq: int, samples: np.ndarray, sr: int, eos: bool = False) -> bytes:
    s = np.ascontiguousarray(samples, dtype="<i2")
    flags = FLAG_EOS if eos else 0
    return _HEADER.pack(int(seq), int(sr), int(s.size), flags) + s.tobytes()


def decode_frame(data: bytes) -> dict:
    seq, sr, n, flags = _HEADER.unpack_from(data, 0)
    payload = data[_HEADER.size:]
    samples = np.frombuffer(payload, dtype="<i2", count=n) if n else np.zeros(0, dtype="<i2")
    return {"seq": seq, "sr": sr, "n": n, "eos": bool(flags & FLAG_EOS), "samples": samples}


def eos_frame(seq: int, sr: int) -> bytes:
    return encode_frame(seq, np.zeros(0, dtype="<i2"), sr, eos=True)
