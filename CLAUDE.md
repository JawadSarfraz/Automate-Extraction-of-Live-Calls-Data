# CLAUDE.md — project context & handoff

> Read this first. It tells you what this project is, what's been built, what we
> learned, and what to do next. Claude Code loads this automatically.
> **Last updated:** 2026-05-25 (added v3 classifier, streaming detector, UDP packet transport).

## 1. Goal

Outbound dialer (campaign 270, "Final Expense") runs an **AI voice bot**
("Kate"/"Hannah") that calls leads. We want to classify each call so the bot can
make a **real-time decision: keep talking or hang up**.

Three target classes (+ a practical fourth):
- **Fair** — genuine human conversation (not interrupted, no abuse, no background noise).
- **Simple VM** — an automated voicemail/IVR message, **or** abusive bashing.
- **Smart VM** — *another bot* posing as a customer, stalling / prolonging the call
  (e.g. anti-telemarketer "honeypot" bots).
- **no_contact** — silence / instant hangup / no real conversation (≈50% of calls).

The end game: a fast, **streaming** detector that protects Fair-call recall (never
hang up on a real customer) while cutting wasted time on VMs and bots. The detector
is meant to run live inside the bot, consuming the call's audio as it arrives.

## 2. Two data sources (both from one dashboard, different APIs)

Dashboard: `https://dashboard.xlitexcore.xdialnetworks.com` (login via `.env` credentials).
API host: `https://api.xlitecore.xdialnetworks.com` (note: `xlitecore`, NOT `xlitexcore`).
Auth: log in once with Playwright → grab the **Bearer JWT** from `localStorage.access_token`
(valid ~24h, cached in `.auth.json`, auto-refreshed). Both APIs use that token.

1. **Recordings** (`/api/v1/recordings/campaign/270?date=…&page_size=500&sort_dir=desc`)
   → audio metadata + **public `file_url`** for each `.wav`. page_size caps at 500.
2. **Calls / dispositions** (`/api/v1/campaigns/270/dashboard?start_date=D&end_date=D&page_size=500`)
   → per-call `category` (Answering Machine, Honeypot, User Silent, Qualified, DNC,
   DNQ, Not Interested, …), `transferred`, and a **short `transcription` snippet**.

The two sets differ in size and aren't 1:1; join by **phone number + nearest
timestamp** (recording start is ~25s before the call's disposition timestamp;
join within a 90s window matches ~98%).

## 3. Pipeline (commands, in order, for one day)

```bash
# --- Node extraction (needs .env with credentials — copy from .env.example) ---
node src/extract.mjs        --date 2026-05-14 --limit 2000   # recordings + file_urls
node src/extract-calls.mjs  --date 2026-05-14 --all          # platform labels (weak supervision)
node src/download.mjs       --date 2026-05-14                # the .wav files (resumable)

# --- Python ML (.venv/bin/python). GPU needs the CUDA libs on the path, see §8. ---
LD_LIBRARY_PATH="$LDLP" .venv/bin/python ml/transcribe.py --date 2026-05-14 --device cuda --workers 1
.venv/bin/python ml/classify_v1.py     --date 2026-05-14    # acoustic baseline + weak labels (RUN BEFORE v2/v3)
.venv/bin/python ml/classify_v2.py     --date 2026-05-14    # rule-based content classifier
.venv/bin/python ml/build_review_v2.py --date 2026-05-14 [--audio remote]   # listening/labeling page

# --- v3 (supervised; needs requirements-v3.txt, see §10) ---
.venv/bin/python ml/embed.py           --date 2026-05-14    # MiniLM customer-residue embeddings (GPU)
.venv/bin/python ml/classify_v3.py     --date 2026-05-14    # HGB over emb+acoustic+cues; OOF eval vs v2
.venv/bin/python ml/build_review_disagree.py --date 2026-05-14 [--audio remote]  # v2-vs-v3 label queue

# --- streaming early-detection (see §11) ---
.venv/bin/python -m streaming.run_stopwatch  --date 2026-05-14    # time-to-detection report
.venv/bin/python -m streaming.build_timeline --date 2026-05-14 [--audio remote]  # per-second viewer
.venv/bin/python -m streaming.packet_demo --selftest <file>.wav --device cpu     # UDP packet demo
.venv/bin/python -m pytest tests/ -q                              # 31 tests
```

Setup: `npm install && npx playwright install --with-deps chromium`;
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## 4. How the v2 classifier works (the key idea) — full spec in DETECTION_ALGORITHM.md

The bot's script is identical on every call, so it **dominates every transcript**.
The trick (`ml/separate.py`): learn a "bot phrase bank" of recurring 5-grams
(+ a curated list of qualifying lines) and **strip them out**, leaving only the
**customer's words**. We then run transparent content detectors on that residue
(first match wins):
- voicemail/IVR phrases → Simple VM; abuse/profanity → Simple VM
- stalling/"honeypot" phrases or looping, *without* engagement → Smart VM
- age stated / "have a plan" / "not interested" → Fair (BEATS a lone stall phrase)
- little/no customer speech → no_contact

`classify_v1.py` is the earlier **acoustic-only** baseline (DSP features in
`ml/features.py`); kept for validation + it produces the `weak_label` v2/v3 consume.
**The exact algorithm, thresholds, and phrase lists are in `DETECTION_ALGORITHM.md`.**

## 5. What we learned (don't relearn the hard way)

- **The recording is mono and contains OUR bot's voice.** Naive VAD over-predicts
  "fair". Must isolate the customer (the bot-strip in `separate.py`).
- **The platform `transcription` field is useless** — short status tokens. We run our **own** Whisper ASR.
- **Answering machines here are IVR/voice greetings, not beeps**. Silent/music-only AMs
  have no transcribable text → fall into no_contact. Main **open gap** (needs acoustic assist).
- **Smart VMs are NOT long calls** (median ~6s); "prolonging" = many rapid short turns.
- Platform `category` is **weak supervision, not ground truth** — taxonomy differs.
- **v3 vs v2 caveat:** v3 is trained AND scored on the platform weak labels, so its big
  lead over v2 mostly means "it learned the platform taxonomy" — not that it's better on
  OUR taxonomy. That needs human labels. Don't over-read it.

## 6. Current state (as of 2026-05-25)

| Date | recordings | calls | audio | transcripts | v1 | v2 | v3 | stopwatch |
|------|-----------|-------|-------|-------------|----|----|----|-----------|
| 2026-05-14 | 2000 | 5427 | 2000 | 2000 | ✓ | ✓ | ✓ | ✓ |
| 2026-05-15 | 2000 | — | 2000 | — | — | — | — | — |

- **v2** (14 May): no_contact 49.5%, fair 34.0%, smart_vm 5.7%, simple_vm 4.2%, uncertain 6.5%.
- **v3** (14 May, 5-fold OOF on 1956 weak labels): **HGB macro-F1 0.819** (LogReg 0.701).
  Per-class F1 fair .92 / no_contact .92 / simple_vm .78 / smart_vm .66. Outputs in
  `data/analysis/classified_v3_*.{csv,json}`, model `v3_model_2026-05-14.joblib`.
- **Streaming stopwatch** (v2 brain, oracle ASR): VMs flagged at **median 4s**, bots at
  **5s** (early detection works); BUT **Fair false-flag rate 15.3%** (the safety blocker)
  and v2 misses ~79% of Simple VMs / ~66% of Smart VMs. See `data/analysis/stopwatch_*`.
- **Packet path proven**: Simple/Smart VM correctly detected on a live **UDP** audio
  stream; a Fair call correctly NOT flagged. (`streaming/packet_demo.py --selftest`).
- **Tests:** 31 passing (`tests/`).

**Git:** all this session's work is committed **locally on `main` but NOT pushed** — the
box has no GitHub credentials. To push, configure a Personal Access Token (repo scope).
**Note:** all of `data/` (audio, transcripts, analysis, HTML) is git-ignored — regenerate
by running the pipeline. The repo is **code only**.

## 7. Next steps (roadmap)

1. **Tighten real-time cadence** — the packet demo flags later than the oracle stopwatch
   because re-transcribing the growing buffer on CPU is slow. Run ASR on the **GPU** and/or
   transcribe a **sliding window** so live fire-times match the oracle (~2–5s).
2. **Phase 2 — cut the Fair false-flag rate** (currently 15.3%): tune the policy
   (raise τ/K, suppress smart_vm flags in the first few seconds, lock to "keep talking"
   the instant an engagement/age cue appears); sweep τ/K → false-flag-vs-seconds-saved curve.
3. **Human labels** — `build_review_disagree.py` queue (the ~35% of calls where v2 & v3
   disagree is the highest-information set). Export `my_labels_*.json`; retarget v3 to OUR taxonomy.
4. **Close the silent-AM gap** — acoustic branch (beep/music/long-monologue) for calls
   with no transcribable customer speech; mandatory for streaming.
5. **The cut/action layer** — DecisionPolicy + safety gates → SimSink → ShadowSink (log-only
   on live traffic) → LiveSink (real hangup). Never skip the shadow stage. See STREAMING_DESIGN.md §9.
6. **Telephony integration** — get a live customer-audio feed (RTP / websocket / media-server
   fork) + a hangup/transfer hook from the dialer; then write one `RtpSource`/`WsSource` adapter.
7. **Scale** — run the pipeline on 15 May + more dates (`--all`); GPU Whisper makes it cheap.

## 8. Environment & gotchas

- **GPU box:** Tesla T4 (15GB), driver 580 / CUDA 13, Ubuntu 24.04. SSH key needed for
  manual tunnels (we don't have one — use VSCode port-forward or the `--audio remote` files).
- **CUDA libs for faster-whisper/CTranslate2** (the real blocker): model loads on cuda but
  inference fails with `libcublas.so.12 not found`. Fix:
  ```bash
  .venv/bin/pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
  SITE=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
  export LDLP="$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib"   # prepend to LD_LIBRARY_PATH
  ```
  T4 base.en float16 ≈ **12.7 min for 2000 calls** (~5× the M1 CPU's ~64 min).
- **torch** is the GPU build (`torch 2.12.0+cu130`, self-contained nvidia cu13 libs — separate
  from the cu12 libs above; no conflict). Used by v3 embeddings (`ml/embed.py`) and as a
  faster-whisper dep. Streaming ASR defaults to CPU to avoid the LD_LIBRARY_PATH dance.
- **Secrets:** `.env` + `.auth.json` git-ignored — never commit. `cp .env.example .env` per box.
- **Public audio:** `.wav` `file_url`s need no auth (and support HTTP range → seekable),
  which is why `--audio remote` HTML files play anywhere.
- **Viewing HTML on the box:** `--audio remote` builds a standalone file → download via VSCode
  Explorer (right-click → Download) and open locally; OR forward port 8765 in VSCode and serve
  `data/audio/<date>/` with `python -m http.server 8765 --bind 127.0.0.1`.
- Stack split is intentional: **Node** for extraction/Playwright, **Python** for ML/audio/streaming.

## 9. Key docs

- `DETECTION_ALGORITHM.md` — exact detection logic (the spec to share with the team).
- `STREAMING_DESIGN.md` — streaming architecture: blind detector + sealed-label grader,
  swappable transport, oracle-vs-real ASR, the cut-layer rollout (sim → shadow → live).
- `ml/FINDINGS_v1.md` — acoustic-baseline findings.

## 10. v3 supervised classifier

Features per call: **384-d MiniLM embedding of the customer residue ⊕ 11 acoustic DSP
(from v1) ⊕ 7 v2 lexical-cue counts**. Supervised on platform weak labels (bootstrap;
built to fold in human labels later). HGB beats LogReg. Heavy deps are **optional**:
```bash
.venv/bin/pip install torch                       # GPU build (or --index-url .../cpu)
.venv/bin/pip install -r requirements-v3.txt      # sentence-transformers, scikit-learn, joblib
```

## 11. Streaming detector & packet transport (`streaming/`)

Decoupled **transport (ears) ↔ detector (brain)**. The detector is fed `PrefixObservation`s
and never sees the label; a separate `Stopwatch` grades after the call.
- `observation.py` `PrefixObservation`; `classifiers.py` `V2RulesClassifier`;
  `trigger.py` `ConfidenceTrigger` (fire when a committal class holds ≥τ for K steps);
  `detector.py` `StreamingDetector`; `harness.py`/`report.py` stopwatch + metrics.
- **Sources:** `OracleTranscriptSource` (reveal word-timestamps by t — fast eval over all
  calls), `RealtimeAudioSource` (chunk a wav + real ASR), `PacketAudioSource` (receive UDP).
- **Packet transport:** `packet_io.py` (RTP-like codec), `packet_sender.py` (stream a wav as
  ~20ms UDP frames), `packet_demo.py` (`--selftest` runs sender+receiver+detector live;
  `--listen` for a real two-process run).
- `run_stopwatch.py`, `build_timeline.py` (per-second viewer, `--audio remote` for standalone).

## 12. File map

```
src/        extract.mjs · extract-calls.mjs · download.mjs · auth.mjs · config.mjs · explore.mjs · probe-calls.mjs
ml/         transcribe.py · separate.py · features.py · classify_v1.py · classify_v2.py · classify_v3.py ·
            embed.py · build_review_v2.py · build_review_disagree.py · build_review.py · FINDINGS_v1.md
streaming/  observation · classifiers · trigger · detector · sources · harness · report ·
            run_stopwatch · build_timeline · packet_io · packet_sender · packet_source · packet_demo · _paths
tests/      test_trigger · test_oracle_source · test_v2_classifier · test_detector_integration · test_packet_io
docs        CLAUDE.md · README.md · STREAMING_DESIGN.md · DETECTION_ALGORITHM.md
deps        requirements.txt · requirements-v3.txt · .env.example · conftest.py
```
