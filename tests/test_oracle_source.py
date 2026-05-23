from streaming.sources import OracleTranscriptSource, _step_times


def _tx():
    return {
        "file": "x.wav", "audio_duration": 5.0,
        "segments": [
            {"s": 0.5, "e": 1.0, "text": "hello",
             "words": [{"s": 0.5, "e": 1.0, "w": " hello"}]},
            {"s": 3.0, "e": 4.0, "text": "who is this",
             "words": [{"s": 3.0, "e": 3.3, "w": " who"},
                       {"s": 3.3, "e": 3.6, "w": " is"},
                       {"s": 3.6, "e": 4.0, "w": " this"}]},
        ],
    }


def test_step_times_end_inclusive():
    assert _step_times(5.0, 1.0) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _step_times(3.0, 1.0) == [1.0, 2.0, 3.0]
    assert _step_times(0.4, 1.0) == [0.4]      # call shorter than a step
    assert _step_times(0.0, 1.0) == [0.0]      # empty call


def test_reveal_grows_monotonically():
    src = OracleTranscriptSource(_tx(), bank=set(), step=1.0)
    obs = {round(o.t, 1): o for o in src.observations()}
    assert obs[1.0].text == "hello"
    assert obs[2.0].text == "hello"            # nothing new between 1s and 3s
    assert "who is this" in obs[4.0].text
    # text only ever grows
    seq = [obs[t].text for t in sorted(obs)]
    for a, b in zip(seq, seq[1:]):
        assert b.startswith(a) or a == ""


def test_n_segments_and_duration():
    src = OracleTranscriptSource(_tx(), bank=set(), step=1.0)
    obs = {round(o.t, 1): o for o in src.observations()}
    assert obs[1.0].n_segments == 1
    assert obs[2.0].n_segments == 1
    assert obs[4.0].n_segments == 2
    assert obs[5.0].duration == 5.0
    assert src.call_len == 5.0


def test_empty_transcript_yields_one_silent_obs():
    src = OracleTranscriptSource({"file": "s.wav", "audio_duration": 0.0, "segments": []},
                                 bank=set(), step=1.0)
    obs = list(src.observations())
    assert len(obs) == 1 and obs[0].text == "" and obs[0].n_segments == 0


def test_bot_script_subtraction():
    # bank holds a bot 5-gram; those tokens must be stripped from customer_text.
    bank = {("hi", "this", "is", "kate", "calling")}
    toks = ["hi", "this", "is", "kate", "calling", "about", "plans"]
    tx = {"file": "y.wav", "audio_duration": 2.0, "segments": [
        {"s": 0.0, "e": 2.0, "text": " ".join(toks),
         "words": [{"s": i * 0.2, "e": i * 0.2 + 0.2, "w": " " + w} for i, w in enumerate(toks)]}]}
    src = OracleTranscriptSource(tx, bank, step=2.0)
    last = list(src.observations())[-1]
    assert last.text == "hi this is kate calling about plans"
    assert last.customer_text == "about plans"   # bot 5-gram masked out
