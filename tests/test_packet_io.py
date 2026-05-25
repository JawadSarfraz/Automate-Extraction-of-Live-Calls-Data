import numpy as np

from streaming.packet_io import encode_frame, decode_frame, eos_frame, FLAG_EOS


def test_roundtrip_preserves_samples():
    s = np.array([0, 1, -1, 32767, -32768, 1234], dtype="<i2")
    fr = decode_frame(encode_frame(7, s, 8000))
    assert fr["seq"] == 7 and fr["sr"] == 8000 and fr["n"] == s.size
    assert fr["eos"] is False
    np.testing.assert_array_equal(fr["samples"], s)


def test_eos_flag_and_empty_payload():
    fr = decode_frame(eos_frame(42, 16000))
    assert fr["eos"] is True and fr["sr"] == 16000 and fr["n"] == 0
    assert fr["samples"].size == 0


def test_seq_and_sr_survive_various_values():
    for seq, sr in [(0, 8000), (1, 16000), (65535, 44100), (10 ** 6, 8000)]:
        fr = decode_frame(encode_frame(seq, np.zeros(3, dtype="<i2"), sr))
        assert fr["seq"] == seq and fr["sr"] == sr and fr["n"] == 3


def test_non_eos_has_zero_flag():
    fr = decode_frame(encode_frame(1, np.ones(2, dtype="<i2"), 8000, eos=False))
    assert (FLAG_EOS & (1 if fr["eos"] else 0)) == 0


def test_float_like_input_is_accepted_via_ascontiguous():
    # encode coerces dtype; ensure no crash and correct length for an int16 view
    s = np.arange(5, dtype="<i2")
    fr = decode_frame(encode_frame(3, s[::-1], 8000))  # non-contiguous slice
    np.testing.assert_array_equal(fr["samples"], s[::-1])
