"""Prefix classifiers — turn a PrefixObservation into (label, confidence, extra).

A classifier sees ONLY the observation; it never receives the true label. The
Phase-1 classifier wraps the existing transparent v2 rules so the streaming and
batch systems make identical decisions given identical text.
"""
from __future__ import annotations
from typing import Protocol

from . import _paths  # noqa: F401  (side effect: puts ml/ on sys.path)
from .observation import PrefixObservation

from classify_v2 import classify as _v2_classify  # noqa: E402  (ml/classify_v2.py)


class PrefixClassifier(Protocol):
    name: str
    def classify(self, obs: PrefixObservation) -> tuple[str, float, dict]:
        ...


class V2RulesClassifier:
    """Phase-1 brain: the v2 content rules applied to the prefix-so-far.

    v2 reads the bot-stripped customer text plus two scalar features (n_segments,
    duration), all of which the OracleTranscriptSource / RealtimeAudioSource supply
    on the growing prefix.
    """
    name = "v2_rules"

    def classify(self, obs: PrefixObservation) -> tuple[str, float, dict]:
        feats = {"n_segments": obs.n_segments, "duration": obs.duration}
        label, conf, reasons = _v2_classify(obs.customer_text, feats)
        return label, float(conf), {"reasons": reasons}
