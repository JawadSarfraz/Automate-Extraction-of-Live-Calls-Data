# Streaming Early-Detection — Architecture Design

> Status: **Phase 1 (design + simulation harness)**. This document is the contract
> we agreed on before implementation. Read it before touching `streaming/`.

## 1. Goal & scope

Move from **batch** classification (analyze the whole finished recording, as v1/v2/v3
do today) to **streaming early-detection**: consume a live call's audio as it arrives,
and decide the call type **as early as we confidently can**.

The end use is our outbound BOT: during a live call we want to know, within the first
few seconds, whether the far side is a **Fair** human, a **Simple VM** (voicemail/IVR or
abuse), a **Smart VM** (anti-telemarketer bot stalling), or **no_contact** — so the bot
can act (hang up / keep talking). This is roadmap step 5 ("stream-ify"), made concrete.

**Phase 1 is "detect & flag only"** (agreed): no automatic hang-up. We only measure
*when* the system becomes confident and *whether it is right*. The action layer comes
later, once the latency/accuracy numbers justify trusting it.

### Decisions locked with the user
| Topic | Decision |
|---|---|
| Audio channels | **Mixed mono** (bot + customer in one channel, like current recordings) → keep the bot-script subtraction step. |
| Action | **Detect & flag only** for now (no auto-drop). |
| First classifier | **v2 rules** first (transparent, instant), swap to v3 later. |
| Eval data | The existing **2000 calls from 2026-05-14**; platform category = **sealed answer key** (scoring only). |

## 2. Core principle: the detector is BLIND; labels only grade

The single most important correctness property:

```
   audio frames ─►  [ StreamingDetector ]  ─►  flag(class, time)
                          (sees ONLY audio)
                                                      │
   sealed label  ──────────────────────────────►  [ Stopwatch / grader ]
                  (label NEVER enters the detector)     compares, scores
```

The `StreamingDetector` receives **only audio** (or, in the oracle sim, only the
pre-computed transcript revealed up to time *t* — never the label). The label is opened
**after** the call by the grader, purely to answer "was it right, and how early?".
Detector and grader are separate objects; the label has no path into the detector. This
is enforced by construction and covered by tests.

## 3. Architecture diagram

```
 EARS (swappable transport)          BRAIN (StreamingDetector, stateful)            GRADER
 ──────────────────────────          ───────────────────────────────────           ──────
 ┌───────────────────────┐  frames   ┌──────────────────────────────────┐  flag    ┌───────────┐
 │ FileReplaySource       │ ────────► │ rolling buffer (prefix)           │ ───────► │ Stopwatch │
 │  pace=fast  (batch eval)│           │   │                              │          │  per-call │
 │  pace=realtime (demo)   │           │   ▼ every STEP (~1s)              │          │  record + │
 │   ── or (future) ──     │           │ Source → PrefixObservation        │          │  aggregate│
 │ RtpSource / WsSource    │           │   (text, customer_text,           │          └───────────┘
 └───────────────────────┘            │    n_segments, duration, [acoustic])│              ▲
                                       │   ▼                               │              │
                                       │ Classifier.classify(obs)          │       sealed label
                                       │   → (label, conf, probs)          │       (scoring only)
                                       │   ▼                               │
                                       │ ConfidenceTrigger                 │
                                       │   → fire when a committal class   │
                                       │     holds ≥τ for K steps           │
                                       └──────────────────────────────────┘
```

**Guiding principle: decouple transport from detection.** The brain is fed
`PrefixObservation`s and does not care whether they came from a replayed file or a live
RTP/websocket feed. When the real transport is ready, we add one Source adapter and the
brain is unchanged.

## 4. Modules (`streaming/`)

| Module | Responsibility |
|---|---|
| `observation.py` | `PrefixObservation` dataclass: everything the brain knows at time *t* (elapsed *t*, full ASR text so far, bot-stripped customer text, n_segments, duration, optional acoustic dict). |
| `sources.py` | Produce `PrefixObservation`s at increasing *t*. `OracleTranscriptSource` (reveals pre-computed word-timestamped transcript up to *t* — Phase 1 default, fast, no ASR). `RealtimeAudioSource` (chunks audio, runs streaming ASR on the prefix — for demo + validation). |
| `classifiers.py` | `PrefixClassifier` interface + `V2RulesClassifier` (wraps `ml/classify_v2.classify` on the prefix). v3 adapter added later. |
| `trigger.py` | `ConfidenceTrigger`: turns the per-step class/confidence stream into a single "flag at time *t*" decision (threshold τ + K-consecutive-window confirmation; "committal" classes only). |
| `detector.py` | `StreamingDetector`: wires Source → Classifier → Trigger, runs the loop, returns a `DetectionResult` (fired?, fire time, fired label, full trajectory). |
| `harness.py` | `Stopwatch`: run the detector on one file vs its sealed label → per-call record; batch over many files → aggregate. |
| `report.py` | Aggregate + pretty-print + CSV/JSON. |

## 5. ASR strategy — oracle vs real (the honest caveat)

Re-running Whisper on the growing prefix every second for 2000 calls is ~tens of
thousands of ASR passes. For Phase 1 we **do not** do that. We already have full-call,
**word-timestamped** transcripts, so the `OracleTranscriptSource` reconstructs "what text
was available by time *t*" by revealing all words with end-time ≤ *t*.

**This measures the latency of the DETECTION LOGIC, holding ASR perfect.** It is an
*optimistic* proxy: a real streaming ASR on partial audio can produce different / less
stable text than the full-call transcript truncated to *t*. We separate the two concerns
on purpose:
- **Phase 1:** oracle ASR → "given good text, how early can the rules decide?" (cheap, all 2000 calls).
- **Validation:** `RealtimeAudioSource` with true incremental Whisper on a *sample* → measures the extra delay/error the real ASR adds.

Both Sources implement the same interface, so the brain and the metrics are identical.

## 6. Confidence → flag policy (flag-only)

At each step the classifier emits a label + confidence on the prefix. The
`ConfidenceTrigger` converts this stream into one decision:

- **Committal classes** = `{simple_vm, smart_vm, fair}`. `no_contact` / `uncertain` mean
  "keep listening" (the customer may not have spoken yet).
- **Fire** the first time a committal class is held for **K consecutive steps** each with
  confidence **≥ τ** (hysteresis → one noisy window can't trigger a flag). Fire time = the
  *t* at which confirmation completes (the moment the bot could act).
- We separately record **Fair false-flags**: a truly-Fair call flagged as VM/bot, and how
  early — this is the safety signal that *would* become a false hang-up once we add the
  action layer. Target: ~0.

τ and K are tunable; Phase 1 uses sensible defaults and Phase 2 sweeps them to draw the
**false-flag vs seconds-saved** trade-off curve.

## 7. Metrics (the stopwatch)

Per call: `{file, true_class (sealed), fired, fired_class, fire_time_s, call_len_s,
correct, lead_time_s = call_len − fire_time}`. Aggregated:

- **Detection latency** per class: median / p90 fire-time on correctly-flagged calls.
- **Fair false-flag rate** (+ how early) — the safety metric.
- **Catch rate** per VM/bot class (% flagged before call end) and **seconds saved** (lead time).
- **Accuracy at flag**, and the distribution of fire-time vs call length.
- **Compute latency** (wall-clock per step) — logged to confirm we stay under real-time.

Two latencies, never conflated: **(A) audio-time to decision** (what saves money / what
the stopwatch reports) vs **(B) wall-clock compute** (capacity; logged, not the headline).

## 8. Phase 1 evaluation plan

Run the stopwatch on the 2000 calls from 2026-05-14, sealed answer key = platform
category mapped to our 4 classes (`weak_label`). Report the section-7 metrics, per class.
Caveats stated up front: weak labels are noisy and follow the platform's taxonomy, and
the oracle ASR is optimistic — so the numbers are a **feasibility read**, not a final SLA.

## 9. Roadmap

- **Phase 1** — sim harness + `StreamingDetector(v2)` + stopwatch (this doc).
- **Phase 2** — cost-sensitive policy: τ/K sweep, false-flag vs seconds-saved curve.
- **Phase 3** — swap v2→v3 in the stream; add an **acoustic branch** so silent/music/beep
  VMs (no text) are caught early (closes the "silent-AM gap", which is mandatory live).
- **Phase 4** — real transport adapter (RTP/websocket from the dialer) + concurrency
  benchmark on the T4; then the action layer (hang-up), gated on Phase 2's safety numbers.

## 10. Production cautions (this feeds the live BOT)

- **Fail safe = keep talking.** Any error/uncertainty must default to NOT hanging up
  (never drop a possible Fair call). The detector returning "no flag" is always safe.
- **Deterministic & testable.** Pure functions where possible; the oracle path is fully
  deterministic and unit-tested. No hidden global state in the brain.
- **No label leakage.** Enforced by construction + tests (section 2).
- **Bounded latency.** Compute per step is measured; the design degrades gracefully (skip
  a step rather than block the call) — to be enforced in the realtime path.
- **Reproducible.** Fixed seeds; config (STEP, τ, K) explicit and logged with every run.

## 11. Interfaces (stable contract)

```python
@dataclass
class PrefixObservation:
    t: float                  # audio seconds elapsed
    text: str                 # full ASR text up to t (bot + customer)
    customer_text: str        # after bot-script subtraction
    n_segments: int           # speech segments started by t
    duration: float           # = t
    acoustic: dict | None     # optional prefix acoustic features

class PrefixClassifier(Protocol):
    def classify(self, obs: PrefixObservation) -> tuple[str, float, dict]: ...
        # returns (label, confidence, extra)   — sees ONLY obs, never the label

class Source(Protocol):
    def observations(self) -> Iterator[PrefixObservation]: ...
```
