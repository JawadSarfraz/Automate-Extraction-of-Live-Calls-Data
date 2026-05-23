"""StreamingDetector — the blind brain. Source -> Classifier -> Trigger.

Receives ONLY a Source of PrefixObservations and returns a DetectionResult. The
true label has no path into this object (enforced: run() takes a source, not a
label). See STREAMING_DESIGN.md §2.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .trigger import ConfidenceTrigger


@dataclass(frozen=True)
class StepRecord:
    t: float
    label: str
    conf: float
    customer_text: str = ""   # the bot-stripped text revealed by time t (for inspection)


@dataclass
class DetectionResult:
    fired: bool
    fire_time: float | None
    fired_label: str | None
    call_len: float
    trajectory: list[StepRecord] = field(default_factory=list)


class StreamingDetector:
    """Wire a classifier + a fresh-per-call trigger over a Source.

    Args:
        classifier:      a PrefixClassifier (stateless; sees only the observation).
        trigger_factory: a zero-arg callable returning a FRESH ConfidenceTrigger per
                         call (so per-call state never leaks between calls).
        stop_on_fire:    stop consuming once a flag fires (production efficiency).
                         Default False so the full trajectory is available for analysis.
    """

    def __init__(self, classifier, trigger_factory=None, stop_on_fire: bool = False):
        self.classifier = classifier
        self.trigger_factory = trigger_factory or ConfidenceTrigger
        self.stop_on_fire = stop_on_fire

    def run(self, source) -> DetectionResult:
        trigger = self.trigger_factory()
        traj: list[StepRecord] = []
        last_t = 0.0
        fired = False
        fire_time = None
        fired_label = None
        for obs in source.observations():
            label, conf, _extra = self.classifier.classify(obs)
            traj.append(StepRecord(obs.t, label, conf, obs.customer_text))
            last_t = obs.t
            if not fired and trigger.update(obs.t, label, conf):
                fired = True
                fire_time = trigger.fired_t
                fired_label = trigger.fired_label
                if self.stop_on_fire:
                    break
        call_len = getattr(source, "call_len", last_t)
        return DetectionResult(
            fired=fired, fire_time=fire_time, fired_label=fired_label,
            call_len=float(call_len), trajectory=traj,
        )
