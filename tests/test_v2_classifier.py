import inspect

from streaming.classifiers import V2RulesClassifier
from streaming.observation import PrefixObservation


def _obs(cust, n=1, dur=5.0):
    return PrefixObservation(t=dur, text=cust, customer_text=cust, n_segments=n, duration=dur)


def test_voicemail_phrase_is_simple_vm():
    label, conf, _ = V2RulesClassifier().classify(
        _obs("please leave your message after the tone"))
    assert label == "simple_vm" and conf >= 0.8


def test_abuse_is_simple_vm():
    label, _, _ = V2RulesClassifier().classify(_obs("you scammer go to hell"))
    assert label == "simple_vm"


def test_stall_phrase_is_smart_vm():
    label, _, _ = V2RulesClassifier().classify(_obs("hello who is this"))
    assert label == "smart_vm"


def test_engagement_beats_stall_and_is_fair():
    # stall phrase present, but stated age + "not interested" -> genuine human
    label, _, _ = V2RulesClassifier().classify(
        _obs("who is this i am 75 and not interested"))
    assert label == "fair"


def test_empty_residue_is_no_contact():
    label, _, _ = V2RulesClassifier().classify(_obs("", n=0, dur=1.0))
    assert label == "no_contact"


def test_long_stall_call_more_confident():
    short = V2RulesClassifier().classify(_obs("what do you want", n=1, dur=5.0))
    long = V2RulesClassifier().classify(_obs("what do you want", n=8, dur=35.0))
    assert short[0] == long[0] == "smart_vm"
    assert long[1] >= short[1]   # duration/segments only raises confidence


def test_classifier_is_blind_signature():
    # the classifier must accept ONLY the observation (no label can be passed in)
    params = list(inspect.signature(V2RulesClassifier().classify).parameters)
    assert params == ["obs"]
