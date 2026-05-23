"""ConfidenceTrigger — turn a per-step (label, confidence) stream into ONE flag.

Phase-1 "detect & flag only" policy (STREAMING_DESIGN.md §6):
  * Committal classes = {simple_vm, smart_vm, fair}. no_contact / uncertain mean
    "keep listening" and never fire.
  * Fire the first time a committal class is held for K consecutive steps, each with
    confidence >= tau (hysteresis: one noisy window can't trigger a flag).
  * Fire time = the t at which the K-th confirming step lands (when the bot could act).

Fail-safe: if nothing is ever confirmed, it simply never fires (= keep talking).
"""
from __future__ import annotations

COMMITTAL = frozenset({"simple_vm", "smart_vm", "fair"})


class ConfidenceTrigger:
    def __init__(self, tau: float = 0.6, k: int = 2, committal=COMMITTAL):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.tau = float(tau)
        self.k = int(k)
        self.committal = frozenset(committal)
        self.reset()

    def reset(self) -> None:
        self._run_label: str | None = None
        self._run_count: int = 0
        self.fired: bool = False
        self.fired_t: float | None = None
        self.fired_label: str | None = None

    def update(self, t: float, label: str, conf: float) -> bool:
        """Feed one step. Returns True only on the step where the flag first fires."""
        if self.fired:
            return False
        if label in self.committal and conf >= self.tau:
            if label == self._run_label:
                self._run_count += 1
            else:
                self._run_label = label
                self._run_count = 1
        else:
            self._run_label = None
            self._run_count = 0
        if self._run_count >= self.k:
            self.fired = True
            self.fired_t = t
            self.fired_label = self._run_label
            return True
        return False
