// Download the actual .wav recordings referenced in an extracted JSON file,
// then generate an HTML player so they can be listened to locally.
//
// Usage:
//   node src/download.mjs                       (all data/recordings_*.json found)
//   node src/download.mjs --date 2026-05-14
//   node src/download.mjs --date 2026-05-14 --date 2026-05-15 --concurrency 10
//
// - Saves to data/audio/<date>/<original-filename>.wav
// - Resumable: skips files already on disk with the right size
// - Retries transient failures; writes a player index.html per date
import fs from 'node:fs';
import path from 'node:path';
import { CONFIG } from './config.mjs';

const log = (...a) => console.log('[download]', ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function parseArgs(argv) {
  const args = { dates: [], concurrency: 8, retries: 4 };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--date') args.dates.push(argv[++i]);
    else if (a === '--concurrency') args.concurrency = parseInt(argv[++i], 10);
    else if (a === '--retries') args.retries = parseInt(argv[++i], 10);
  }
  return args;
}

function discoverDates(args) {
  if (args.dates.length) return args.dates;
  return fs.readdirSync(CONFIG.dataDir)
    .map((f) => f.match(/^recordings_(\d{4}-\d{2}-\d{2})\.json$/))
    .filter(Boolean)
    .map((m) => m[1]);
}

function fmtBytes(n) {
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(2) + ' GB';
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(1) + ' MB';
  return (n / 1024).toFixed(0) + ' KB';
}

async function downloadOne(rec, destDir, retries) {
  const url = rec.file_url;
  if (!url) return { status: 'skip', reason: 'no url' };
  const name = path.basename(new URL(url).pathname);
  const dest = path.join(destDir, name);

  // Resume: a completed file already exists on disk (we write atomically below,
  // so a present file is always a fully-downloaded one).
  if (fs.existsSync(dest)) return { status: 'skip', name, bytes: fs.statSync(dest).size };

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      // Ask for the raw bytes (no gzip) so Content-Length matches what we write.
      const res = await fetch(url, { headers: { 'accept-encoding': 'identity' } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length < 44 || buf.toString('latin1', 0, 4) !== 'RIFF') {
        throw new Error(`not a WAV (got ${buf.length}B, head="${buf.toString('latin1', 0, 4)}")`);
      }
      const expected = parseInt(res.headers.get('content-length') || '0', 10);
      if (expected && buf.length !== expected) throw new Error(`size mismatch ${buf.length}/${expected}`);
      // Atomic write: temp file then rename, so partial downloads never look complete.
      const tmp = dest + '.part';
      fs.writeFileSync(tmp, buf);
      fs.renameSync(tmp, dest);
      return { status: 'ok', name, bytes: buf.length };
    } catch (err) {
      if (attempt === retries) return { status: 'fail', name, reason: err.message };
      await sleep(400 * 2 ** (attempt - 1));
    }
  }
}

// Simple bounded-concurrency runner.
async function runPool(items, concurrency, worker, onProgress) {
  const results = [];
  let idx = 0, done = 0;
  async function next() {
    while (idx < items.length) {
      const myIdx = idx++;
      const r = await worker(items[myIdx], myIdx);
      results[myIdx] = r;
      onProgress(++done, items.length, r);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, next));
  return results;
}

function writePlayerHtml(date, recs, destDir, stats) {
  const rows = recs.map((r, i) => {
    const file = r.file_url ? path.basename(new URL(r.file_url).pathname) : '';
    return `<tr>
      <td>${i + 1}</td><td>${r.time}</td><td>${r.phone_number}</td>
      <td>${r.duration}</td><td>${r.size}</td><td>${r.extension}</td>
      <td><audio controls preload="none" src="${file}"></audio></td>
      <td><a href="${file}" download>⬇</a></td>
    </tr>`;
  }).join('\n');
  const html = `<!doctype html><html><head><meta charset="utf-8">
<title>Recordings ${date} — Campaign ${CONFIG.campaignId}</title>
<style>
  body{font:14px system-ui,Segoe UI,Arial;margin:0;background:#f5f6f8;color:#222}
  header{position:sticky;top:0;background:#1f6feb;color:#fff;padding:12px 18px;box-shadow:0 2px 6px rgba(0,0,0,.15)}
  header b{font-size:16px}
  .meta{font-size:12px;opacity:.9;margin-top:2px}
  #q{margin:10px 18px;padding:8px 10px;width:280px;border:1px solid #ccc;border-radius:6px}
  table{border-collapse:collapse;width:calc(100% - 36px);margin:0 18px 40px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.1)}
  th,td{padding:7px 10px;border-bottom:1px solid #eee;text-align:left;white-space:nowrap}
  th{position:sticky;top:64px;background:#fafbfc;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#555}
  tr:hover{background:#f0f6ff}
  audio{height:32px;vertical-align:middle}
</style></head><body>
<header><b>📞 Call Recordings — ${date}</b><div class="meta">Campaign ${CONFIG.campaignId} · ${recs.length} recordings · downloaded ${stats.ok + stats.skip}/${recs.length} (${fmtBytes(stats.bytes)}) · newest first</div></header>
<input id="q" placeholder="Filter by phone number…" oninput="filt(this.value)">
<table id="t"><thead><tr><th>#</th><th>Time</th><th>Phone</th><th>Duration</th><th>Size</th><th>Ext</th><th>Listen</th><th>Save</th></tr></thead>
<tbody>${rows}</tbody></table>
<script>
function filt(v){v=v.trim();for(const tr of document.querySelectorAll('#t tbody tr')){tr.style.display=tr.children[2].textContent.includes(v)?'':'none';}}
</script></body></html>`;
  fs.writeFileSync(path.join(destDir, 'index.html'), html);
}

async function main() {
  const args = parseArgs(process.argv);
  const dates = discoverDates(args);
  if (!dates.length) { log('No recordings_*.json found in data/. Run extract first.'); return; }
  log('dates:', dates.join(', '), '| concurrency:', args.concurrency);

  for (const date of dates) {
    const jsonPath = path.join(CONFIG.dataDir, `recordings_${date}.json`);
    if (!fs.existsSync(jsonPath)) { log(`! missing ${jsonPath}, skipping`); continue; }
    const recs = JSON.parse(fs.readFileSync(jsonPath, 'utf8')).recordings;
    const destDir = path.join(CONFIG.dataDir, 'audio', date);
    fs.mkdirSync(destDir, { recursive: true });
    log(`\n=== ${date}: ${recs.length} files → ${path.relative(CONFIG.dataDir + '/..', destDir)} ===`);

    const stats = { ok: 0, skip: 0, fail: 0, bytes: 0 };
    const failures = [];
    await runPool(recs, args.concurrency,
      (rec) => downloadOne(rec, destDir, args.retries),
      (done, total, r) => {
        if (r.status === 'ok') { stats.ok++; stats.bytes += r.bytes; }
        else if (r.status === 'skip') { stats.skip++; stats.bytes += r.bytes || 0; }
        else { stats.fail++; failures.push(r); }
        if (done % 100 === 0 || done === total) {
          process.stdout.write(`\r  ${done}/${total}  ok=${stats.ok} skip=${stats.skip} fail=${stats.fail}  ${fmtBytes(stats.bytes)}   `);
        }
      });
    process.stdout.write('\n');
    if (failures.length) {
      log(`  ${failures.length} failures (first 5):`, failures.slice(0, 5).map((f) => `${f.name}: ${f.reason}`));
      fs.writeFileSync(path.join(destDir, '_failures.json'), JSON.stringify(failures, null, 2));
    }
    writePlayerHtml(date, recs, destDir, stats);
    log(`  ✓ ${date}: downloaded ${stats.ok}, skipped ${stats.skip}, failed ${stats.fail}, total ${fmtBytes(stats.bytes)}`);
    log(`  ▶ Listen: open ${path.relative(CONFIG.dataDir + '/..', path.join(destDir, 'index.html'))}`);
  }
}

main().catch((e) => { console.error('[download] FAILED:', e); process.exit(1); });
