# CLAUDE.md — project context & handoff

> Read this first. It tells you what this project is, what's been built, what we
> learned, and what to do next. Claude Code loads this automatically.

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

The end game: a fast, streaming detector that protects Fair-call recall (never
hang up on a real customer) while cutting wasted time on VMs and bots.

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
# Node extraction (needs .env with credentials — copy from .env.example)
node src/extract.mjs        --date 2026-05-14 --limit 2000   # recordings + file_urls
node src/extract-calls.mjs  --date 2026-05-14 --all          # platform labels (weak supervision)
node src/download.mjs       --date 2026-05-14                # the .wav files (resumable)

# Python ML (.venv/bin/python; on AWS GPU add --device cuda)
.venv/bin/python ml/transcribe.py      --date 2026-05-14 --workers 4   # Whisper ASR
.venv/bin/python ml/classify_v1.py     --date 2026-05-14               # acoustic baseline + validation
.venv/bin/python ml/classify_v2.py     --date 2026-05-14               # content-based classifier (the real one)
.venv/bin/python ml/build_review_v2.py --date 2026-05-14 [--audio remote]  # listening/labeling page
```

Setup: `npm install && npx playwright install chromium`;
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## 4. How the v2 classifier works (the key idea)

The bot's script is identical on every call, so it **dominates every transcript**.
The trick (`ml/separate.py`): learn a "bot phrase bank" of recurring n-grams
(+ a curated list of qualifying lines) and **strip them out**, leaving only the
**customer's words**. We then run transparent content detectors on that residue:
- voicemail/IVR phrases → Simple VM
- abuse/profanity → Simple VM (per user's definition)
- stalling/"honeypot" phrases ("where are you from", "sorry can you repeat that")
  or looping → Smart VM
- age stated / "have a plan" / "not interested" → Fair (this BEATS a lone stall phrase)
- little/no customer speech → no_contact

`classify_v1.py` is the earlier **acoustic-only** baseline (DSP features in
`ml/features.py`); kept for validation. It's weak on its own (see findings).

## 5. What we learned (don't relearn the hard way)

- **The recording is mono and contains OUR bot's voice.** Naive VAD/turn-taking
  fires on every call → acoustic-only over-predicts "fair". Must isolate the customer.
- **The platform `transcription` field is useless for us** — short status tokens
  (median 11 chars, e.g. "User-Silent", "INAUDIBLE"). We run our **own** Whisper ASR.
- **Answering machines here are IVR/voice greetings, not beeps** (beep detector hit ~5%).
  Silent/music-only AMs have no transcribable text → they fall into no_contact. This
  is the main **open gap** (needs an acoustic assist or relabeling).
- **Smart VMs are NOT long calls** (median ~6s); "prolonging" shows as many rapid
  short turns, not duration.
- Platform `category` is **weak supervision, not ground truth** — taxonomy differs
  (e.g. abuse = "DNC" there, but "Simple VM" in our scheme).

## 6. Current state (as of last session)

| Date | recordings | calls | audio | transcripts | v1 | v2 |
|------|-----------|-------|-------|-------------|----|----|
| 2026-05-14 | 2000 | 5427 | 2000 | 2000 | ✓ | ✓ |
| 2026-05-15 | 2000 | — | 2000 | — | — | — |

v2 result on 14 May (2000 calls): no_contact 49.8%, fair 34.1%, smart_vm 5.6% (112),
simple_vm 4.3% (87), uncertain 6.2%. Outputs in `data/analysis/classified_v2_2026-05-14.{csv,json}`.
Review/labeling page: `data/audio/2026-05-14/review_v2.html` (open in browser; relabel
F/Si/Sm/N → Export → `my_labels_v2_*.json`). `--audio remote` makes a shareable copy.

**Note:** all of `data/` (audio, transcripts, analysis) is git-ignored — regenerate
it on the box by running the pipeline. The repo is **code only**.

## 7. Next steps (roadmap)

1. **Human labels** — review `review_v2.html`, correct mislabels, export
   `my_labels_v2_*.json`. ~100–200 verified labels/class is the bottleneck unlock.
2. **Train v3 (GPU)** — supervised model on (customer-text embeddings ⊕ acoustic
   features), weak-labeled by category + corrected by human labels. Beats rule-based v2.
3. **Close the silent-AM gap** — add an acoustic assist (beep/music/long-monologue)
   for calls with no transcribable customer speech.
4. **Scale** — run the pipeline on 15 May and more dates (`--all`); GPU Whisper makes
   tens of thousands of calls practical.
5. **Stream-ify** — make the winning features computable on a call prefix for
   real-time early hang-up decisions; tune a cost-sensitive policy (protect Fair recall).

## 8. Environment & gotchas

- macOS dev box was Apple **M1** (Whisper ran on CPU, base.en, ~64 min for 2000).
  On AWS GPU: `ml/transcribe.py --device cuda --workers 1` (auto float16); may need
  `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` for CTranslate2.
- **Secrets**: `.env` (credentials) and `.auth.json` (JWT) are git-ignored — never
  commit them. `cp .env.example .env` and fill in on each new box.
- **Public audio**: the `.wav` `file_url`s need no auth; only the metadata/label
  fetch needs login. That's why `--audio remote` review pages work for anyone.
- Stack split is intentional: **Node** for extraction/Playwright, **Python** for ML/audio.

## 9. File map

```
src/  extract.mjs · extract-calls.mjs · download.mjs · auth.mjs · config.mjs · explore.mjs · probe-calls.mjs
ml/   transcribe.py · separate.py · features.py · classify_v1.py · classify_v2.py · build_review_v2.py · build_review.py · FINDINGS_v1.md
README.md (setup + usage)   requirements.txt   .env.example
```
