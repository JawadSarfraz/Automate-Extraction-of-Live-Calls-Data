# Detection Algorithm — how we decide Simple VM vs Smart VM (vs Fair / no_contact)

> This is the exact logic currently executed by the streaming detector (the v2
> rule engine). It is transparent on purpose: every decision traces to a phrase
> list or a threshold, so the team can audit and tune it. Source of truth:
> `ml/classify_v2.py` (`classify()` + the phrase banks) and `ml/separate.py`
> (bot/customer separation), wrapped for streaming in `streaming/`.
>
> Note: there is also a trained ML model (v3, `ml/classify_v3.py`); it is NOT in
> the streaming path. The algorithm below — v2 — is what we execute to detect.

## 0. Input / output

- **Input:** call audio (a live stream, or a recording). Telephony mono, 8 kHz.
- **Output per call:** one of `simple_vm` | `smart_vm` | `fair` | `no_contact`
  (+ `uncertain`), with a confidence in [0,1], plus — in streaming — the *second*
  at which we became confident.

## 1. Transcribe (ASR)

Run Whisper on the audio → text with word-level timestamps. In streaming we
transcribe the growing prefix each step (see §6).

## 2. Isolate the CUSTOMER's words (bot-script subtraction)

Our bot reads a near-identical script every call, so its words dominate every
transcript. We remove them:

1. **Learn a "bot phrase bank"** = every 5-word sequence (5-gram) that appears in
   ≥ 8% of calls, plus a curated list of known bot lines (`CURATED_BOT`).
2. **Mask** any token in the transcript that falls inside a bank 5-gram.
3. What remains = the **customer's words** (`customer_text`). All detection below
   runs on `customer_text`, never the raw transcript.

(`ml/separate.py`: `learn_bot_bank`, `customer_words`. NGRAM=5, min_df_frac=0.08.)

## 3. Compute cues on the customer's words

| cue | meaning |
|---|---|
| `vm`     | customer_text contains any **VOICEMAIL** phrase (substring match) |
| `abuse`  | contains any **ABUSE** word |
| `stall`  | contains any **STALL** phrase |
| `engage` | contains any **ENGAGE** phrase |
| `age`    | regex finds a stated age 40–109 |
| `rep`    | largest count of a repeated 3-word phrase (loop detector) |
| `wc`     | number of customer words |

## 4. Decision cascade — FIRST match wins (this is the whole algorithm)

```
1.  if wc <= 2 and not (vm or abuse or stall):        -> no_contact   (conf 0.70)
2.  if vm:                                            -> SIMPLE_VM    (conf 0.85)
3.  if abuse:                                         -> SIMPLE_VM    (conf 0.80)

    strong_fair = (engage or age)        # a real human signal

4.  if stall and not strong_fair:                     -> SMART_VM
        conf = 0.80 if (n_segments >= 6 or duration >= 30) else 0.60
5.  if rep >= 3 and not strong_fair:                  -> SMART_VM     (conf 0.60)

6.  if strong_fair:                                   -> fair         (conf 0.75)
7.  if wc >= 5:                                        -> fair         (conf 0.45)
8.  else:                                              -> uncertain    (conf 0.30)
```

**So the basis is:**

- **SIMPLE VM** = the customer side *speaks like a machine or abuse*: it contains a
  **voicemail/IVR phrase** ("leave a message", "after the tone", "press 1", …) **or
  profanity/abuse**. (Rules 2–3.)
- **SMART VM** = the customer side *stalls like a bot*: it contains a **stalling/
  evasive phrase** ("who is this", "what do you want", "can you repeat that", "are
  you a robot", …) **OR loops** (same 3 words repeated ≥ 3×) — **AND shows no
  genuine engagement** (no stated age, no "not interested"-type phrase). The
  "no engagement" guard matters because real people also ask "who is this?".
  (Rules 4–5.) Confidence is higher (0.80) if the call also *dragged on* (≥ 6
  speech turns or ≥ 30 s) — the "prolonging" signal.
- **FAIR** = genuine engagement (stated age / "not interested" / "have a plan" /
  "appreciate your time"), which **beats** a lone stall phrase; or simply ≥ 5
  customer words with no machine/bot cue. (Rules 6–7.)
- **no_contact** = essentially no customer words. (Rule 1.)

## 5. The exact phrase lists (`ml/classify_v2.py`)

**VOICEMAIL** → Simple VM:
`leave a message, leave your message, after the beep, after the tone, at the tone,
press 1, press one, press 2, press 3, press pound, not available, is unavailable,
your call has been, has been forwarded, voicemail, voice mail, mailbox, record your
message, finished your message, more options, please hang up, google voice, please
leave, reached the, the person you are trying, automated, this call may be recorded,
return your call, we will return, we'll return, return your, you can also reach,
reach me at, as soon as we can, your message after, leave a name, leave your name,
name and number, at the sound of`

**ABUSE** → Simple VM:
`fuck, fucking, cunt, bitch, asshole, ass hole, bastard, dick, prick, shut up, scam,
scammer, go to hell, piss off, son of a, motherf`

**STALL** → Smart VM (unless engaged):
`what can i do for you, where are you from, who is this, who's this, whos this, can
you repeat, repeat that, say that again, i can't hear you, i cannot hear you, can't
hear you, what company, what do you want, who are you, what's your name, are you a
robot, is this a robot, speak up, breaking up, didn't catch, come again, pardon,
hello hello hello, are you there, still there, how can i help, how may i help`

**ENGAGE** → Fair (overrides a lone stall):
`not interested, have a plan, already have, take me off, do not call, don't call,
remove me, no thank you, i'm covered, already covered, have insurance, appreciate
your time, have a good, have a wonderful, i'm fine`

**AGE** → counts as engagement: any number 40–109 spoken.

## 6. Streaming: turning the per-second decision into one flag

In the live/streaming path (`streaming/`), we run §1–§4 on the **growing prefix**
every `step` seconds (default 1 s), producing `(label, confidence)` each second.
A **ConfidenceTrigger** then decides *when* to raise the flag:

```
fire the flag the first time a committal class (simple_vm | smart_vm | fair)
is produced for K consecutive steps, each with confidence >= tau
   (defaults: tau = 0.60, k = 2)
```

`no_contact` / `uncertain` never fire (they mean "keep listening"). If nothing ever
confirms, no flag fires — the fail-safe is to keep talking.
(`streaming/trigger.py`, `streaming/detector.py`.)

## 7. Known limitations (so the team tunes the right things)

- **Silent / music / beep-only voicemails have no words** → §2-4 can't fire → they
  fall to `no_contact`. (Needs an acoustic branch — roadmap Phase 3.)
- **Phrase lists are finite** — a VM/bot phrased differently is missed.
- **Early stall, late engagement:** a real person who opens "who is this?" can be
  flagged smart_vm before their engagement cue arrives → false-flag. (Phase 2 fixes
  this by suppressing early smart_vm + locking to Fair on any engagement cue.)
- These rules are **weak-supervision-tuned**, not yet validated on human labels.
