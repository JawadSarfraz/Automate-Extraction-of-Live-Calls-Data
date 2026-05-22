"""Cheap acoustic feature extraction for telephony WAVs (8 kHz mono, 16-bit PCM).

No neural nets — just DSP with numpy/scipy. Designed so every feature could be
computed on a *streaming* prefix of the call later (for real-time early detection).
"""
from __future__ import annotations
import numpy as np
from scipy.io import wavfile
from scipy import signal

FRAME_MS = 25
HOP_MS = 10


def _frames(x: np.ndarray, sr: int):
    flen = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)
    if len(x) < flen:
        x = np.pad(x, (0, flen - len(x)))
    n = 1 + (len(x) - flen) // hop
    idx = np.arange(flen)[None, :] + hop * np.arange(n)[:, None]
    return x[idx], hop


def _runs(mask: np.ndarray):
    """Yield (start, end) index runs where mask is True."""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))


def extract(path: str) -> dict:
    sr, raw = wavfile.read(path)
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    x = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        x /= 32768.0
    dur = len(x) / sr
    if len(x) < sr * 0.2:  # < 200 ms, basically empty
        return dict(duration=round(dur, 2), speech_ratio=0.0, n_speech_segments=0,
                    longest_monologue=0.0, longest_silence=round(dur, 2), beep=False,
                    beep_score=0.0, spectral_flatness=0.0, bg_ratio=0.0, centroid_hz=0.0,
                    rms_db=-99.0)

    fr, hop = _frames(x, sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-9)

    # Adaptive VAD: threshold sits 30% up from the noise floor toward the loud peak.
    floor = np.percentile(rms_db, 15)
    peak = np.percentile(rms_db, 95)
    thr = floor + 0.30 * max(peak - floor, 6.0)
    speech = rms_db > thr

    # Smooth: drop speech runs < 120 ms and silence gaps < 100 ms.
    fps = sr / hop
    min_speech = int(0.12 * fps)
    min_gap = int(0.10 * fps)
    for s, e in _runs(~speech):
        if e - s < min_gap:
            speech[s:e] = True
    for s, e in _runs(speech):
        if e - s < min_speech:
            speech[s:e] = False

    seg = _runs(speech)
    sil = _runs(~speech)
    speech_ratio = float(speech.mean())
    longest_monologue = max(((e - s) / fps for s, e in seg), default=0.0)
    longest_silence = max(((e - s) / fps for s, e in sil), default=0.0)

    # Spectrogram-based features.
    f, _, Sxx = signal.spectrogram(x, fs=sr, nperseg=256, noverlap=128, mode="psd")
    Sxx += 1e-12
    colsum = Sxx.sum(axis=0)
    # Tonality per frame = fraction of energy in the single strongest bin.
    peak_frac = Sxx.max(axis=0) / colsum
    peak_freq = f[Sxx.argmax(axis=0)]
    loud = colsum > np.percentile(colsum, 60)
    beep_frames = (peak_frac > 0.45) & (peak_freq > 650) & (peak_freq < 1600) & loud
    # Beep = a sustained (>=180 ms) tonal run.
    spc_hop = 128 / sr
    beep_run = max(((e - s) * spc_hop for s, e in _runs(beep_frames)), default=0.0)
    beep = beep_run >= 0.18
    beep_score = round(float(beep_run), 2)

    # Spectral flatness (geo/arith mean): speech is peaky (low), music/noise flatter (high).
    gm = np.exp(np.mean(np.log(Sxx), axis=0))
    am = np.mean(Sxx, axis=0)
    flatness = float(np.mean(gm / am))

    # Background ratio: how loud the "silence" is vs the speech (noisy line / music).
    sp_idx = speech[: len(colsum)] if len(speech) >= len(colsum) else np.resize(speech, len(colsum))
    if sp_idx.any() and (~sp_idx).any():
        bg_ratio = float(np.median(colsum[~sp_idx]) / (np.median(colsum[sp_idx]) + 1e-12))
    else:
        bg_ratio = 0.0

    centroid = float(np.sum(f[:, None] * Sxx) / np.sum(Sxx))

    return dict(
        duration=round(dur, 2),
        speech_ratio=round(speech_ratio, 3),
        n_speech_segments=len(seg),
        longest_monologue=round(longest_monologue, 2),
        longest_silence=round(longest_silence, 2),
        beep=bool(beep),
        beep_score=beep_score,
        spectral_flatness=round(flatness, 4),
        bg_ratio=round(bg_ratio, 3),
        centroid_hz=round(centroid, 1),
        rms_db=round(float(np.median(rms_db)), 1),
    )
