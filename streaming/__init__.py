"""Streaming early-detection of call type (Fair / Simple VM / Smart VM / no_contact).

Phase 1: a blind StreamingDetector fed PrefixObservations + a separate Stopwatch
grader. See STREAMING_DESIGN.md for the architecture contract.

Public surface:
    PrefixObservation, OracleTranscriptSource, RealtimeAudioSource,
    V2RulesClassifier, ConfidenceTrigger, StreamingDetector, DetectionResult,
    Stopwatch, CallRecord
"""
from .observation import PrefixObservation
from .classifiers import V2RulesClassifier
from .trigger import ConfidenceTrigger, COMMITTAL
from .detector import StreamingDetector, DetectionResult, StepRecord
from .sources import OracleTranscriptSource
from .harness import Stopwatch, CallRecord

__all__ = [
    "PrefixObservation", "V2RulesClassifier", "ConfidenceTrigger", "COMMITTAL",
    "StreamingDetector", "DetectionResult", "StepRecord", "OracleTranscriptSource",
    "Stopwatch", "CallRecord",
]
