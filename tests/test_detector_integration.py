"""End-to-end: synthetic transcript -> OracleSource -> StreamingDetector -> Stopwatch."""
import inspect

from streaming import (StreamingDetector, V2RulesClassifier, ConfidenceTrigger,
                       OracleTranscriptSource, Stopwatch)


def _tx(tokens, dur):
    """One segment, evenly-spaced word timestamps from 0..len(tokens) seconds."""
    words = [{"s": float(i), "e": float(i + 1), "w": " " + w} for i, w in enumerate(tokens)]
    return {"file": "t.wav", "audio_duration": float(dur), "segments": [
        {"s": 0.0, "e": float(len(tokens)), "text": " ".join(tokens), "words": words}]}


def _detector(tau=0.6, k=2):
    return StreamingDetector(V2RulesClassifier(), lambda: ConfidenceTrigger(tau=tau, k=k))


def test_detects_voicemail_and_reports_lead_time():
    # "please leave your message ..." -> "leave your message" complete by t=4s
    toks = ["please", "leave", "your", "message", "after", "the", "tone"]
    src = OracleTranscriptSource(_tx(toks, dur=10.0), bank=set(), step=1.0)
    res = _detector().run(src)
    assert res.fired and res.fired_label == "simple_vm"
    assert res.fire_time <= 6.0                      # caught early, well before 10s

    rec = Stopwatch(_detector()).grade(src, true_class="simple_vm")
    assert rec.correct is True
    assert rec.lead_time == res.call_len - res.fire_time > 0


def test_silence_never_fires_and_is_safe():
    src = OracleTranscriptSource({"file": "s.wav", "audio_duration": 8.0, "segments": []},
                                 bank=set(), step=1.0)
    res = _detector().run(src)
    assert res.fired is False                         # fail-safe: keep talking
    rec = Stopwatch(_detector()).grade(src, true_class="no_contact")
    assert rec.fired is False and rec.false_flag is False


def test_real_human_not_false_flagged():
    # genuine engagement: should be fair (or not fired), never simple/smart -> no false flag
    toks = ["yeah", "i", "am", "72", "and", "not", "interested", "thanks"]
    src = OracleTranscriptSource(_tx(toks, dur=9.0), bank=set(), step=1.0)
    rec = Stopwatch(_detector()).grade(src, true_class="fair")
    assert rec.false_flag is False


def test_run_takes_no_label_argument():
    # structural guarantee that the label cannot leak into the detector
    params = list(inspect.signature(StreamingDetector.run).parameters)
    assert params == ["self", "source"]


def test_fresh_trigger_per_call_no_state_leak():
    det = _detector(k=1)
    # a call that fires
    fire_src = OracleTranscriptSource(_tx(["leave", "your", "message"], dur=5.0), set(), step=1.0)
    r1 = det.run(fire_src)
    assert r1.fired
    # a subsequent silent call must NOT inherit the fired state
    silent = OracleTranscriptSource({"file": "q.wav", "audio_duration": 4.0, "segments": []}, set(), step=1.0)
    r2 = det.run(silent)
    assert r2.fired is False
