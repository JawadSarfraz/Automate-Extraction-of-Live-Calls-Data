# Classifier v1 — findings (14 May 2026, 2000 recordings)

## What we built
A full, fast pipeline (`ml/classify_v1.py`, ~6s for 2000 files on M1):
recordings ↔ calls join → acoustic feature extraction (DSP, no neural nets) →
platform-category weak labels → audio-only heuristic classifier → validation report
→ review/labeling tool (`ml/build_review.py` → `data/audio/2026-05-14/review.html`).

## Class distribution (platform-derived, mapped to our taxonomy)
| Class | Count | % |
|---|---|---|
| no_contact (Silent/Hangup/DAIR/Inaudible) | 1114 | 55.7% |
| fair (Qualified/Not Interested/DNC/DNQ/…) | 524 | 26.2% |
| simple_vm (Answering Machine) | 277 | 13.8% |
| smart_vm (Honeypot) | 41 | 2.0% |
| unmatched (no joined call) | 44 | 2.2% |

## Key findings (why acoustic-only is not enough)
1. **Our bot's voice is in every recording** (mono mix). Naive turn-taking/VAD
   therefore fires on almost all calls → audio model over-predicts `fair`
   (98% recall on fair, but useless precision). Must separate bot vs customer.
2. **Answering machines here are IVR/voice greetings, not beeps.** Beep detector
   fired on only 5% of them ("press 3 to speak to…", "EARLY_TRIGGER"). Beep is a
   weak standalone signal for this campaign.
3. **Smart-VM (Honeypot) is NOT simply "long calls."** Median 6s, max 52s vs
   24s overall. The "prolonging" pattern shows up as *many short turns*
   (e.g. "don't disappear" → 17 segments in 38s), not raw duration.
4. The discriminating content ("where are you from", "I didn't hear you", abuse,
   looping) lives in **what is said and by whom** — i.e. transcript + speaker.

## Conclusion
Acoustic DSP gives useful *features* but cannot reproduce the labels on its own.
The next investment is **our own ASR + speaker attribution**, not more DSP rules.

## v2 plan
1. **ASR** every recording with on-device Whisper (M1, faster-whisper/mlx) →
   real transcript + word timestamps.
2. **Speaker attribution**: our bot's lines are scripted/known → tag bot vs
   customer turns; measure customer interruptions (overlap), response latency,
   turn counts, talk-ratio.
3. **Content signals**: abuse/profanity, voicemail phrases, looping/stalling
   ("hello? are you there?"), coherence of customer replies.
4. **Model**: start with transparent rules + gradient boosting on
   (acoustic ⊕ ASR ⊕ dialogue) features, weak-labeled by platform category,
   corrected via the review tool. Optionally a Claude-API zero-shot classifier
   on transcripts for a strong baseline.
5. **Stream-ify** the winning features for real-time early hang-up decisions.
