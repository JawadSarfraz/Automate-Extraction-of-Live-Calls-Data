// Exploration script: log in, then dump the structure of the login page and
// the recordings view so we can build a precise extractor.
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';
import { CONFIG } from './config.mjs';

const log = (...a) => console.log('[explore]', ...a);

async function dumpInteractive(page, label) {
  const info = await page.evaluate(() => {
    const sel = (el) => {
      const parts = [];
      if (el.id) parts.push('#' + el.id);
      if (el.name) parts.push(`[name="${el.name}"]`);
      return parts.join('') || el.tagName.toLowerCase();
    };
    const inputs = [...document.querySelectorAll('input,select,textarea')].map((el) => ({
      tag: el.tagName.toLowerCase(),
      type: el.type || null,
      name: el.name || null,
      id: el.id || null,
      placeholder: el.placeholder || null,
      sel: sel(el),
    }));
    const buttons = [...document.querySelectorAll('button,a[role="button"],[type="submit"]')]
      .map((el) => ({ tag: el.tagName.toLowerCase(), text: (el.innerText || el.value || '').trim().slice(0, 40), id: el.id || null }))
      .filter((b) => b.text);
    return { url: location.href, title: document.title, inputs, buttons };
  });
  log(`--- ${label} ---`);
  console.log(JSON.stringify(info, null, 2));
  return info;
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await ctx.newPage();

  // 1. Login page
  log('opening login page:', CONFIG.baseUrl);
  await page.goto(CONFIG.baseUrl, { waitUntil: 'networkidle', timeout: 60000 });
  await page.screenshot({ path: path.join(CONFIG.shotDir, '01-login.png'), fullPage: true });
  await dumpInteractive(page, 'LOGIN PAGE');

  // 2. Attempt login: find username + password inputs heuristically
  const userInput = page.locator('input[type="text"], input[name*="user" i], input[name*="login" i], input[type="email"]').first();
  const passInput = page.locator('input[type="password"]').first();
  await userInput.fill(CONFIG.user);
  await passInput.fill(CONFIG.pass);
  log('filled credentials, submitting...');
  const submit = page.locator('button[type="submit"], button:has-text("Login"), button:has-text("Sign in"), input[type="submit"]').first();
  await Promise.all([
    page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => {}),
    submit.click().catch(async () => { await passInput.press('Enter'); }),
  ]);
  await page.waitForTimeout(3000);
  await page.screenshot({ path: path.join(CONFIG.shotDir, '02-after-login.png'), fullPage: true });
  log('after login url:', page.url());

  // Save auth so the extractor can reuse the session
  await ctx.storageState({ path: CONFIG.storageState });
  log('saved storage state to', CONFIG.storageState);

  // 3. Navigate to the recordings view
  const recUrl = `${CONFIG.baseUrl}/dashboard?campaign_id=${CONFIG.campaignId}&view=recordings`;
  log('opening recordings view:', recUrl);
  await page.goto(recUrl, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(4000);
  await page.screenshot({ path: path.join(CONFIG.shotDir, '03-recordings.png'), fullPage: true });
  await dumpInteractive(page, 'RECORDINGS VIEW');

  // 4. Dump table-ish structure
  const tables = await page.evaluate(() => {
    const out = [];
    document.querySelectorAll('table').forEach((t, i) => {
      const headers = [...t.querySelectorAll('thead th, thead td')].map((h) => h.innerText.trim());
      const firstRow = [...(t.querySelector('tbody tr')?.children || [])].map((c) => c.innerText.trim().slice(0, 30));
      out.push({ index: i, rows: t.querySelectorAll('tbody tr').length, headers, firstRow });
    });
    // also look for grid/role-based tables
    const grids = document.querySelectorAll('[role="grid"], [role="table"], .ag-root, .MuiDataGrid-root, .rt-table');
    return { tables: out, gridSelectors: [...grids].map((g) => g.className) };
  });
  log('--- TABLES / GRIDS ---');
  console.log(JSON.stringify(tables, null, 2));

  // Dump the rendered HTML of the main content for offline inspection
  const html = await page.content();
  fs.writeFileSync(path.join(CONFIG.shotDir, 'recordings.html'), html);
  log('wrote recordings.html (length', html.length, ')');

  await browser.close();
  log('done.');
})().catch((e) => {
  console.error('[explore] ERROR:', e);
  process.exit(1);
});
