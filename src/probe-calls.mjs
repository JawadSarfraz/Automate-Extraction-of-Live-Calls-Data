// Signal probe: fetch a sample of the 14 May "calls" dataset and inspect
// transcripts grouped by the platform's existing categories, to gauge how
// separable Fair / Simple VM / Smart VM are from text + metadata.
import { getToken } from './auth.mjs';
import { CONFIG } from './config.mjs';

const API = 'https://api.xlitecore.xdialnetworks.com/api/v1';
const DATE = process.argv[2] || '2026-05-14';
const PAGES = parseInt(process.argv[3] || '3', 10); // 3 x 500 = 1500 calls sampled

(async () => {
  const token = await getToken();
  const h = { authorization: `Bearer ${token}`, accept: 'application/json', referer: `${CONFIG.baseUrl}/` };

  const calls = [];
  let allCategories = null;
  for (let p = 1; p <= PAGES; p++) {
    const url = `${API}/campaigns/${CONFIG.campaignId}/dashboard?start_date=${DATE}&end_date=${DATE}&page=${p}&page_size=500&sort_order=desc`;
    const j = await (await fetch(url, { headers: h })).json();
    if (!allCategories) { allCategories = j.all_categories; console.log('total_records for', DATE, '=', j.pagination.total_records); }
    calls.push(...(j.calls || []));
  }
  console.log(`sampled ${calls.length} calls\n`);

  console.log('=== CATEGORY DISTRIBUTION (whole day) ===');
  for (const c of allCategories.sort((a, b) => b.count - a.count)) {
    console.log(`  ${String(c.count).padStart(6)}  ${c.name}  (transferred=${c.transferred_count})`);
  }

  // transcript length stats overall
  const tlen = calls.map((c) => (c.transcription || '').length).sort((a, b) => a - b);
  console.log(`\ntranscript chars: min ${tlen[0]} median ${tlen[Math.floor(tlen.length / 2)]} max ${tlen[tlen.length - 1]}; empty=${calls.filter((c) => !c.transcription).length}`);

  // group by category and print examples
  const byCat = {};
  for (const c of calls) (byCat[c.category] ||= []).push(c);
  console.log('\n=== TRANSCRIPT EXAMPLES PER CATEGORY (sampled) ===');
  for (const [cat, arr] of Object.entries(byCat).sort((a, b) => b[1].length - a[1].length)) {
    const lens = arr.map((c) => (c.transcription || '').length);
    const avg = Math.round(lens.reduce((a, b) => a + b, 0) / lens.length);
    console.log(`\n### ${cat}  (n=${arr.length} in sample, avg transcript ${avg} chars, transferred=${arr.filter((c) => c.transferred).length}) ###`);
    for (const c of arr.slice(0, 3)) {
      console.log(`  • [${c.timestamp}] tx="${(c.transcription || '(empty)').slice(0, 280)}"`);
    }
  }
})().catch((e) => { console.error('PROBE ERROR:', e); process.exit(1); });
