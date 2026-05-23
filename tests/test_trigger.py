import pytest

from streaming.trigger import ConfidenceTrigger


def test_fires_after_k_consecutive():
    tr = ConfidenceTrigger(tau=0.6, k=2)
    assert tr.update(1.0, "smart_vm", 0.6) is False
    assert tr.update(2.0, "smart_vm", 0.6) is True
    assert tr.fired and tr.fired_t == 2.0 and tr.fired_label == "smart_vm"


def test_below_tau_does_not_count():
    tr = ConfidenceTrigger(tau=0.7, k=2)
    assert tr.update(1.0, "fair", 0.45) is False
    assert tr.update(2.0, "fair", 0.45) is False
    assert not tr.fired


def test_run_resets_on_label_change():
    tr = ConfidenceTrigger(tau=0.6, k=2)
    tr.update(1.0, "smart_vm", 0.6)            # count=1
    assert tr.update(2.0, "simple_vm", 0.85) is False  # different label -> count=1
    assert tr.update(3.0, "simple_vm", 0.85) is True
    assert tr.fired_label == "simple_vm" and tr.fired_t == 3.0


def test_non_committal_never_fires():
    tr = ConfidenceTrigger(tau=0.5, k=1)
    assert tr.update(1.0, "no_contact", 0.7) is False
    assert tr.update(2.0, "uncertain", 0.99) is False
    assert not tr.fired


def test_k1_fires_immediately():
    tr = ConfidenceTrigger(tau=0.6, k=1)
    assert tr.update(3.0, "simple_vm", 0.85) is True
    assert tr.fired_t == 3.0


def test_no_refire_after_fired():
    tr = ConfidenceTrigger(tau=0.6, k=1)
    tr.update(1.0, "simple_vm", 0.85)
    assert tr.update(2.0, "smart_vm", 0.8) is False
    assert tr.fired_label == "simple_vm"


def test_interrupted_run_does_not_fire():
    tr = ConfidenceTrigger(tau=0.6, k=2)
    assert tr.update(1.0, "smart_vm", 0.6) is False   # count=1
    assert tr.update(2.0, "no_contact", 0.7) is False  # break -> count=0
    assert tr.update(3.0, "smart_vm", 0.6) is False   # count=1 again
    assert not tr.fired


def test_reset_clears_state():
    tr = ConfidenceTrigger(tau=0.6, k=1)
    tr.update(1.0, "fair", 0.75)
    tr.reset()
    assert not tr.fired and tr.fired_t is None


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        ConfidenceTrigger(k=0)
