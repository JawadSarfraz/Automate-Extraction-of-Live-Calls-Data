// Extract the full "calls" dataset (category/disposition + transcript snippet +
// transferred flag) for a date. This is the label/metadata source for the
// Fair/Simple/Smart classifier. Saves JSON + CSV to data/.
//
// Usage: node src/extract-calls.mjs --date 2026-05-14 [--all|--limit N]
import fs from 'node:fs';
import path from 'node:path';
import { CONFIG } from './config.mjs';
import { getToken, login } from './auth.mjs';

const API = 'https://api.xlitecore.xdialnetworks.com/api/v1';
const PAGE_SIZE = 500;
const log = (...a) => console.log('[calls]', ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function parseArgs(argv) {
  const a = { date: '2026-05-14', all: true, limit: Infinity, campaign: CONFIG.campaignId };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--date') a.date = argv[++i];
    else if (argv[i] === '--limit') { a.limit = parseInt(argv[++i], 10); a.all = false; }
    else if (argv[i] === '--all') a.all = true;
    else if (argv[i] === '--campaign') a.campaign = argv[++i];
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(a.date)) throw new Error('--date must be YYYY-MM-DD');
  return a;
}

async function fetchPage({ campaign, date, page }, tokenRef) {
  const url = `${API}/campaigns/${campaign}/dashboard?start_date=${date}&end_date=${date}&page=${page}&page_size=${PAGE_SIZE}&sort_order=desc`;
  for (let attempt = 1; attempt <= 4; attempt++) {
    try {
      const res = await fetch(url, { headers: { authorization: `Bearer ${tokenRef.token}`, accept: 'application/json', referer: `${CONFIG.baseUrl}/` } });
      if (res.status === 401 || res.status === 403) { tokenRef.token = await login(); continue; }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      if (attempt === 4) throw e;
      await sleep(500 * 2 ** (attempt - 1));
    }
  }
}

function toCsv(rows) {
  if (!rows.length) return '';
  const headers = Object.keys(rows[0]);
  const esc = (v) => { const s = String(v ?? ''); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [headers.join(','), ...rows.map((r) => headers.map((h) => esc(r[h])).join(','))].join('\n') + '\n';
}

async function main() {
  const args = parseArgs(process.argv);
  fs.mkdirSync(CONFIG.dataDir, { recursive: true });
  const tokenRef = { token: await getToken() };
  log(`campaign=${args.campaign} date=${args.date} target=${args.all ? 'ALL' : args.limit}`);

  const calls = [];
  let pagination = null, page = 1, categories = null;
  while (true) {
    const data = await fetchPage({ campaign: args.campaign, date: args.date, page }, tokenRef);
    pagination = data.pagination;
    categories = categories || data.all_categories;
    calls.push(...(data.calls || []));
    log(`page ${page}/${pagination.total_pages} → have ${calls.length}/${pagination.total_records}`);
    if ((!args.all && calls.length >= args.limit) || !pagination.has_next || (data.calls || []).length === 0) break;
    page++;
    await sleep(120);
  }
  const sliced = args.all ? calls : calls.slice(0, args.limit);

  const base = path.join(CONFIG.dataDir, `calls_${args.date}`);
  fs.writeFileSync(`${base}.json`, JSON.stringify({
    meta: {
      campaign_id: args.campaign, date: args.date,
      total_records_for_date: pagination?.total_records ?? null,
      extracted_count: sliced.length, extracted_at: new Date().toISOString(),
      category_distribution: (categories || []).map((c) => ({ name: c.name, count: c.count, transferred: c.transferred_count })),
      source: `${API}/campaigns/${args.campaign}/dashboard`,
    },
    calls: sliced,
  }, null, 2));
  fs.writeFileSync(`${base}.csv`, toCsv(sliced));
  log(`✓ saved ${sliced.length} calls → ${path.relative(CONFIG.dataDir + '/..', base)}.json/.csv`);
}

main().catch((e) => { console.error('[calls] FAILED:', e); process.exit(1); });
