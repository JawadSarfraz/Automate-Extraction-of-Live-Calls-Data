"""PrefixObservation — everything the detector knows at one moment in a live call.

This is the ONLY thing a classifier is allowed to see. It deliberately does NOT
carry the true label (see STREAMING_DESIGN.md §2: the detector is blind).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixObservation:
    """A snapshot of the call after `t` seconds of audio.

    Attributes:
        t:             audio seconds elapsed since call start.
        text:          full ASR text heard so far (bot + customer, mixed mono).
        customer_text: `text` with the bot's script subtracted (customer words only).
        n_segments:    number of speech segments that have started by `t`.
        duration:      audio duration observed so far (== t); kept explicit because
                       the v2 rules read a `duration` feature.
        acoustic:      optional prefix acoustic features (None when not computed).
    """
    t: float
    text: str
    customer_text: str
    n_segments: int
    duration: float
    acoustic: dict | None = None
