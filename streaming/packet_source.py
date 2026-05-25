"""PacketAudioSource — receive UDP audio packets and feed the detector as a Source.

This is the production-shaped "ears": a socket receiver thread reassembles the
incoming PCM frames into a growing buffer, and observations() yields a
PrefixObservation every `step` seconds by transcribing the buffer-so-far. It is a
drop-in for OracleTranscriptSource / RealtimeAudioSource — the detector (brain) is
unchanged, only the audio now arrives over the network.
"""
from __future__ import annotations
import socket
import threading
import time

import numpy as np

from . import _paths  # noqa: F401
from .observation import PrefixObservation
from .packet_io import decode_frame

from separate import customer_words  # noqa: E402


class PacketAudioSource:
    def __init__(self, asr, bank: set, host: str = "127.0.0.1", port: int = 0,
                 step: float = 1.0, idle_timeout: float = 3.0, start_timeout: float = 15.0):
        self.asr = asr
        self.bank = bank
        self.step = float(step)
        self.idle_timeout = float(idle_timeout)
        self.start_timeout = float(start_timeout)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.sock.bind((host, port))
        self.host, self.port = self.sock.getsockname()
        self.file = "<packets>"
        self.call_len = 0.0
        self._chunks: list[np.ndarray] = []
        self._sr: int | None = None
        self._done = False
        self._last_rx: float | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> "PacketAudioSource":
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        return self

    def _recv_loop(self):
        self.sock.settimeout(0.5)
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                if self._last_rx and (time.time() - self._last_rx) > self.idle_timeout:
                    with self._lock:
                        self._done = True
                    return
                continue
            fr = decode_frame(data)
            self._last_rx = time.time()
            with self._lock:
                if self._sr is None:
                    self._sr = fr["sr"]
                if fr["n"]:
                    self._chunks.append(fr["samples"])
                if fr["eos"]:
                    self._done = True
                    return

    def _snapshot(self):
        with self._lock:
            arr = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype="<i2")
            return arr, self._sr, self._done

    def observations(self):
        # Wait for the first audio (call start).
        t0 = time.time()
        while True:
            arr, sr, done = self._snapshot()
            if sr is not None and arr.size > 0:
                break
            if done or (time.time() - t0) > self.start_timeout:
                return
            time.sleep(0.05)

        while True:
            waited = 0.0
            while waited < self.step and not self._snapshot()[2]:
                time.sleep(0.05)
                waited += 0.05
            arr, sr, done = self._snapshot()
            t = arr.size / sr if sr else 0.0
            text, n_seg = self.asr.transcribe_array(arr, sr)
            cust = customer_words(text, self.bank)
            self.call_len = round(t, 2)
            yield PrefixObservation(t=round(t, 2), text=text, customer_text=cust,
                                    n_segments=n_seg, duration=round(t, 2))
            if done:
                return

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
