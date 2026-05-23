"""Sources — produce PrefixObservations at increasing time t.

Two implementations, same interface (STREAMING_DESIGN.md §4-5):

* OracleTranscriptSource — reveals a pre-computed word-timestamped transcript up to
  time t. Deterministic, no ASR at runtime; the Phase-1 default for measuring the
  detection LOGIC latency across all calls cheaply. Optimistic about ASR.

* RealtimeAudioSource — chunks a wav and runs streaming ASR on the growing audio
  prefix each step. For demo + validating the extra delay real ASR adds (a sample,
  not all 2000).
"""
from __future__ import annotations
import os
import tempfile
from typing import Iterator

from . import _paths  # noqa: F401
from .observation import PrefixObservation

from separate import customer_words  # noqa: E402  (ml/separate.py)

EPS = 1e-6


def _step_times(call_len: float, step: float) -> list[float]:
    """Times step, 2*step, ... strictly inside the call, always ending at call_len."""
    if call_len <= EPS:
        return [0.0]
    ts: list[float] = []
    t = step
    while t < call_len - EPS:
        ts.append(round(t, 3))
        t += step
    ts.append(round(call_len, 3))
    return ts


class OracleTranscriptSource:
    """Replay a saved transcript (with word/segment timestamps) as a live prefix.

    Args:
        transcript: a transcripts/<date>/<file>.json dict (keys: audio_duration,
                    segments[{s,e,text,words[{s,e,w}]}]).
        bank:       the learned bot-script n-gram bank (from separate.learn_bot_bank).
        step:       seconds between observations.
    """

    def __init__(self, transcript: dict, bank: set, step: float = 1.0):
        self.bank = bank
        self.step = float(step)
        # Flat (end_time, token) timeline + segment start times.
        self._timeline: list[tuple[float, str]] = []
        self._seg_starts: list[float] = []
        max_end = 0.0
        for seg in transcript.get("segments", []) or []:
            self._seg_starts.append(float(seg.get("s", 0.0)))
            words = seg.get("words") or []
            if words:
                for w in words:
                    e = float(w.get("e", seg.get("e", 0.0)))
                    self._timeline.append((e, str(w.get("w", ""))))
                    max_end = max(max_end, e)
            else:  # no word timestamps — token becomes available at segment end
                e = float(seg.get("e", 0.0))
                self._timeline.append((e, str(seg.get("text", ""))))
                max_end = max(max_end, e)
        dur = transcript.get("audio_duration")
        self.call_len = float(dur) if dur else max_end
        self.file = transcript.get("file", "")

    def observations(self) -> Iterator[PrefixObservation]:
        for t in _step_times(self.call_len, self.step):
            tokens = [tok for (e, tok) in self._timeline if e <= t + EPS]
            text = " ".join(" ".join(tokens).split())
            cust = customer_words(text, self.bank)
            n_seg = sum(1 for s in self._seg_starts if s <= t + EPS)
            yield PrefixObservation(
                t=t, text=text, customer_text=cust,
                n_segments=n_seg, duration=t,
            )


class WhisperPrefixASR:
    """Re-transcribe an audio prefix with faster-whisper (lazy-loaded).

    Mirrors the batch transcribe.py settings so text matches. Writes the prefix to
    a temp wav and passes the path, so ctranslate2 handles decode + resample to
    16 kHz exactly as in batch (telephony wavs are 8 kHz).
    """

    def __init__(self, model_name: str = "base.en", device: str = "cuda",
                 compute_type: str | None = None):
        from faster_whisper import WhisperModel
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.sr_written = None

    def transcribe_array(self, x, sr: int) -> tuple[str, int]:
        import numpy as np
        from scipy.io import wavfile
        if x.size == 0:
            return "", 0
        xi = x if np.issubdtype(x.dtype, np.integer) else (x * 32768.0).astype("int16")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            wavfile.write(path, sr, xi)
            segments, _ = self.model.transcribe(
                path, language="en", beam_size=1, vad_filter=True,
                condition_on_previous_text=False,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            segs = list(segments)
            text = " ".join(s.text.strip() for s in segs).strip()
            return text, len(segs)
        finally:
            os.remove(path)


class RealtimeAudioSource:
    """Stream a wav: every `step` seconds, transcribe the audio prefix and strip bot.

    Used for demo / sample-validation against the optimistic oracle. Heavier (real
    ASR per step), so not run across all 2000 calls.
    """

    def __init__(self, wav_path: str, asr: WhisperPrefixASR, bank: set, step: float = 1.0):
        from scipy.io import wavfile
        import numpy as np
        sr, raw = wavfile.read(wav_path)
        if raw.ndim > 1:
            raw = raw.mean(axis=1).astype(raw.dtype)
        self.sr = int(sr)
        self.samples = raw
        self.asr = asr
        self.bank = bank
        self.step = float(step)
        self.call_len = len(raw) / float(sr) if sr else 0.0
        self.file = os.path.basename(wav_path)
        self._np = np

    def observations(self) -> Iterator[PrefixObservation]:
        for t in _step_times(self.call_len, self.step):
            n = int(round(t * self.sr))
            prefix = self.samples[:n]
            text, n_seg = self.asr.transcribe_array(prefix, self.sr)
            cust = customer_words(text, self.bank)
            yield PrefixObservation(
                t=t, text=text, customer_text=cust,
                n_segments=n_seg, duration=t,
            )
