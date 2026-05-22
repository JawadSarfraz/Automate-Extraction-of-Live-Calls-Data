// Extract call-recording data for a given date from the xDial dashboard API
// and save it as JSON + CSV.
//
// Usage:
//   node src/extract.mjs --date 2026-05-14 --limit 2000 [--sort desc|asc]
//   node src/extract.mjs --date 2026-05-14 --all          (fetch every record)
//
// Defaults: date=2026-05-14, limit=2000, sort=desc (newest first, matches the
// dashboard's default view), campaign from XDIAL_CAMPAIGN_ID (270).
import fs from 'node:fs';
import path from 'node:path';
import { CONFIG } from './config.mjs';
import { getToken, login } from './auth.mjs';

const API_HOST = 'https://api.xlitecore.xdialnetworks.com';
const PAGE_SIZE = 500;          // API hard-caps page_size at 500
const MAX_RETRIES = 4;
const log = (...a) => console.log('[extract]', ...a);

function parseArgs(argv) {
  const args = { date: '2026-05-14', limit: 2000, sort: 'desc', all: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--date') args.date = argv[++i];
    else if (a === '--limit') args.limit = parseInt(argv[++i], 10);
    else if (a === '--sort') args.sort = argv[++i];
    else if (a === '--all') args.all = true;
    else if (a === '--campaign') args.campaign = argv[++i];
  }
  args.campaign = args.campaign || CONFIG.campaignId;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(args.date)) throw new Error(`--date must be YYYY-MM-DD, got "${args.date}"`);
  if (!['asc', 'desc'].includes(args.sort)) throw new Error('--sort must be asc or desc');
  return args;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Fetch a single page, retrying transient failures and re-authenticating on 401.
async function fetchPage({ campaign, date, page, sort }, tokenRef) {
  const url = `${API_HOST}/api/v1/recordings/campaign/${campaign}`
    + `?date=${date}&page=${page}&page_size=${PAGE_SIZE}&sort_by=time&sort_dir=${sort}`;
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(url, {
        headers: {
          authorization: `Bearer ${tokenRef.token}`,
          accept: 'application/json',
          referer: `${CONFIG.baseUrl}/`,
        },
      });
      if (res.status === 401 || res.status === 403) {
        log('auth expired/denied — re-logging in...');
        tokenRef.token = await login();
        continue; // retry without counting against backoff
      }
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text().catch(() => '')}`.slice(0, 200));
      return await res.json();
    } catch (err) {
      if (attempt === MAX_RETRIES) throw err;
      const wait = 500 * 2 ** (attempt - 1);
      log(`page ${page} attempt ${attempt} failed (${err.message}); retrying in ${wait}ms`);
      await sleep(wait);
    }
  }
}

// "200010" -> "20:00:10"
function formatTime(t) {
  const s = String(t).padStart(6, '0');
  return `${s.slice(0, 2)}:${s.slice(2, 4)}:${s.slice(4, 6)}`;
}

function normalize(rec, date) {
  const time = formatTime(rec.time);
  return {
    date,
    time,
    datetime: `${date} ${time}`,
    phone_number: rec.phone_number ?? '',
    duration: rec.duration ?? '',
    size: rec.size ?? '',
    extension: rec.extension ?? '',
    server_name: rec.server_name ?? '',
    server_id: rec.server_id ?? '',
    file_url: rec.file_url ?? '',
  };
}

function toCsv(rows) {
  if (rows.length === 0) return '';
  const headers = Object.keys(rows[0]);
  const esc = (v) => {
    const s = String(v ?? '');
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [headers.join(',')];
  for (const r of rows) lines.push(headers.map((h) => esc(r[h])).join(','));
  return lines.join('\n') + '\n';
}

async function main() {
  const args = parseArgs(process.argv);
  fs.mkdirSync(CONFIG.dataDir, { recursive: true });
  log(`campaign=${args.campaign} date=${args.date} sort=${args.sort} target=${args.all ? 'ALL' : args.limit}`);

  const tokenRef = { token: await getToken() };

  const collected = [];
  let pagination = null;
  let page = 1;
  while (true) {
    const data = await fetchPage({ campaign: args.campaign, date: args.date, page, sort: args.sort }, tokenRef);
    pagination = data.pagination;
    const recs = data.recordings || [];
    collected.push(...recs);
    log(`page ${page}/${pagination.total_pages} → +${recs.length} (have ${collected.length}/${pagination.total_records})`);

    const reachedLimit = !args.all && collected.length >= args.limit;
    if (reachedLimit || !pagination.has_next || recs.length === 0) break;
    page++;
    await sleep(150); // be polite to the API
  }

  const target = args.all ? collected.length : Math.min(args.limit, collected.length);
  const sliced = collected.slice(0, target).map((r) => normalize(r, args.date));

  const base = path.join(CONFIG.dataDir, `recordings_${args.date}`);
  const jsonOut = {
    meta: {
      campaign_id: args.campaign,
      date: args.date,
      sort_by: 'time',
      sort_dir: args.sort,
      order_note: args.sort === 'desc' ? 'newest call first (matches dashboard default)' : 'oldest call first',
      total_records_for_date: pagination?.total_records ?? null,
      extracted_count: sliced.length,
      requested_limit: args.all ? 'all' : args.limit,
      extracted_at: new Date().toISOString(),
      source: `${API_HOST}/api/v1/recordings/campaign/${args.campaign}`,
    },
    recordings: sliced,
  };
  fs.writeFileSync(`${base}.json`, JSON.stringify(jsonOut, null, 2));
  fs.writeFileSync(`${base}.csv`, toCsv(sliced));

  log('─'.repeat(60));
  log(`✓ Date ${args.date}: ${pagination?.total_records ?? '?'} total recordings on server`);
  log(`✓ Extracted ${sliced.length} records (sort=${args.sort}, ${jsonOut.meta.order_note})`);
  log(`✓ Saved JSON → ${path.relative(CONFIG.dataDir + '/..', base)}.json`);
  log(`✓ Saved CSV  → ${path.relative(CONFIG.dataDir + '/..', base)}.csv`);
}

main().catch((e) => { console.error('[extract] FAILED:', e); process.exit(1); });
