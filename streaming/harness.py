"""Stopwatch — grade the blind detector against a SEALED label.

The label is supplied to grade() (not to the detector) and used only to score the
result after the call. This is where the answer key is opened.
"""
from __future__ import annotations
from dataclasses import dataclass

from .detector import StreamingDetector

VM_BOT = ("simple_vm", "smart_vm")


@dataclass
class CallRecord:
    file: str
    true_class: str        # sealed answer key
    fired: bool
    fired_class: str | None
    fire_time: float | None
    call_len: float
    correct: bool          # fired AND fired_class == true_class
    lead_time: float       # call_len - fire_time (seconds the bot would save), else 0
    false_flag: bool       # true Fair call flagged as a VM/bot (the safety metric)


class Stopwatch:
    def __init__(self, detector: StreamingDetector):
        self.detector = detector

    def grade(self, source, true_class: str) -> CallRecord:
        res = self.detector.run(source)             # detector never sees true_class
        fired = res.fired
        fc = res.fired_label
        correct = bool(fired and fc == true_class)
        lead = float(res.call_len - res.fire_time) if (fired and res.fire_time is not None) else 0.0
        false_flag = bool(true_class == "fair" and fired and fc in VM_BOT)
        return CallRecord(
            file=getattr(source, "file", ""),
            true_class=true_class,
            fired=fired,
            fired_class=fc,
            fire_time=res.fire_time,
            call_len=res.call_len,
            correct=correct,
            lead_time=lead,
            false_flag=false_flag,
        )
