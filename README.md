# Automate Extraction of Live Calls Data

Extracts call-recording data from the xDial dashboard
(`dashboard.xlitexcore.xdialnetworks.com`) for a given campaign and date, and
saves it as **JSON** and **CSV**.

## How it works

The dashboard's "Recordings" view is backed by a JSON API:

```
GET https://api.xlitecore.xdialnetworks.com/api/v1/recordings/campaign/<id>?date=YYYY-MM-DD&page=N&page_size=500&sort_by=time&sort_dir=desc
```

The tool logs in once with Playwright (Chromium), captures the Bearer JWT the
app stores in `localStorage`, then calls this API directly with pagination
(500 records/request — the API's max). This is far faster and more reliable
than scraping the rendered table page-by-page.

## Setup

```bash
npm install
npx playwright install chromium      # one-time browser download

# Python ML/audio pipeline
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Credentials live in `.env` (git-ignored). Copy the template and fill it in:

```bash
cp .env.example .env      # then edit with your real username/password
```

```
XDIAL_BASE_URL=https://dashboard.xlitexcore.xdialnetworks.com
XDIAL_USER=your_username
XDIAL_PASS=your_password
XDIAL_CAMPAIGN_ID=270
```

## Usage

```bash
# First 2000 calls for 14 May 2026 (newest-first, matches the dashboard view)
node src/extract.mjs --date 2026-05-14 --limit 2000

# Oldest-first instead (earliest calls of the day)
node src/extract.mjs --date 2026-05-14 --limit 2000 --sort asc

# Every recording for a date
node src/extract.mjs --date 2026-05-15 --all

# Different campaign
node src/extract.mjs --date 2026-05-15 --limit 2000 --campaign 270
```

### Options

| Flag         | Default      | Meaning                                            |
|--------------|--------------|----------------------------------------------------|
| `--date`     | `2026-05-14` | Day to extract, `YYYY-MM-DD`                        |
| `--limit`    | `2000`       | Max records to save                                |
| `--all`      | off          | Fetch every record for the date (ignores `--limit`)|
| `--sort`     | `desc`       | `desc` = newest call first, `asc` = oldest first   |
| `--campaign` | `270`        | Campaign id                                        |

## Output

Files are written to `data/` (git-ignored):

- `data/recordings_<date>.json` — records plus a `meta` block (total available,
  extracted count, sort order, timestamp, source).
- `data/recordings_<date>.csv` — columns:
  `date, time, datetime, phone_number, duration, size, extension, server_name, server_id, file_url`

## Downloading the audio (.wav) recordings

`extract.mjs` only fetches the metadata + the `file_url` of each recording. To
download the actual audio so you can listen to it:

```bash
# Download the .wav files referenced by every recordings_*.json in data/
node src/download.mjs

# Just one date, with more parallelism
node src/download.mjs --date 2026-05-14 --concurrency 12
```

- Audio is saved to `data/audio/<date>/<original-filename>.wav`.
- **Resumable**: files already on disk are skipped, so you can re-run anytime.
- Downloads are validated as real WAV files (RIFF header) and written
  atomically (`.part` → rename), so a partial file never looks complete.
- A **player page** is generated at `data/audio/<date>/index.html` — open it in
  a browser to play/search/download every recording for that day. To listen:

  ```bash
  open data/audio/2026-05-14/index.html      # macOS
  ```

## Call classification — Fair / Simple-VM / Smart-VM (ML pipeline)

Beyond extraction, the `ml/` pipeline classifies each call into:
- **Fair** — genuine human conversation
- **Simple VM** — automated voicemail/IVR message, or abusive bashing
- **Smart VM** — another bot acting like a customer / stalling / prolonging
- **no_contact** — silent / instant hangup / no real conversation

It works on the **customer's words**: we transcribe each recording with Whisper,
fingerprint and strip our own bot's script (which is identical on every call),
and run transparent content detectors on what the customer actually said.

End-to-end for one day (after `.env` is set):

```bash
# 1. metadata + audio
node src/extract.mjs        --date 2026-05-14 --limit 2000   # recordings + file_urls
node src/extract-calls.mjs  --date 2026-05-14 --all          # platform labels (weak supervision)
node src/download.mjs       --date 2026-05-14                # the .wav files

# 2. transcribe (CPU here; on a GPU box add --device cuda)
.venv/bin/python ml/transcribe.py --date 2026-05-14 --workers 4

# 3. classify + build the listening/labeling page
.venv/bin/python ml/classify_v1.py     --date 2026-05-14     # acoustic baseline + validation
.venv/bin/python ml/classify_v2.py     --date 2026-05-14     # content-based classifier
.venv/bin/python ml/build_review_v2.py --date 2026-05-14     # -> data/audio/<date>/review_v2.html
open data/audio/2026-05-14/review_v2.html                    # listen, verify, relabel, export
```

Outputs land in `data/transcripts/<date>/` and `data/analysis/classified_v2_<date>.{csv,json}`.
See `ml/FINDINGS_v1.md` for what we learned (why acoustic-only wasn't enough → ASR).

## Running on AWS (GPU)

The repo holds **code only** — the data (audio, transcripts) is git-ignored and
regenerated on the box. On a CUDA GPU instance, Whisper runs an order of
magnitude faster.

```bash
git clone https://github.com/JawadSarfraz/Automate-Extraction-of-Live-Calls-Data.git
cd Automate-Extraction-of-Live-Calls-Data
npm install && npx playwright install --with-deps chromium
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # fill in credentials

# regenerate data, then transcribe ON THE GPU:
node src/extract.mjs --date 2026-05-14 --limit 2000
node src/extract-calls.mjs --date 2026-05-14 --all
node src/download.mjs --date 2026-05-14
.venv/bin/python ml/transcribe.py --date 2026-05-14 --device cuda --workers 1
```

> The `.wav` URLs are publicly downloadable; only the metadata/label fetch needs
> login. For GPU Whisper, also `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
> if CTranslate2 doesn't find CUDA libs.

## Auth notes

- The JWT is valid ~24h and is cached in `.auth.json` (git-ignored, **never commit**).
  The tool reuses it while valid and silently re-logs-in when it expires.
- `src/explore.mjs` is a diagnostic that dumps the login + recordings page
  structure and screenshots (useful if the site layout changes).

## Files

```
src/
  config.mjs        loads .env, shared paths/config
  auth.mjs          login + JWT reuse/refresh
  extract.mjs       recordings extractor (API pagination → JSON/CSV)
  extract-calls.mjs calls/disposition extractor (weak labels)
  download.mjs      downloads .wav audio + builds a player page
  explore.mjs       page-structure diagnostic
ml/
  transcribe.py     Whisper ASR over a day's audio (CPU/GPU, resumable)
  separate.py       bot-script fingerprint → isolate customer words
  features.py       acoustic DSP features (VAD, beep, music, monologue)
  classify_v1.py    acoustic baseline + validation report
  classify_v2.py    content-based Fair/Simple/Smart classifier
  build_review*.py  generate the listening/labeling HTML pages
  FINDINGS_v1.md    what we learned from the acoustic baseline
```
