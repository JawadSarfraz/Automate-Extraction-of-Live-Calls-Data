// Session/auth management for the xDial dashboard.
// Reuses a saved JWT when still valid; otherwise logs in via Playwright to
// refresh it. Returns a Bearer token usable against the API directly.
import fs from 'node:fs';
import { chromium } from 'playwright';
import { CONFIG } from './config.mjs';

const log = (...a) => console.log('[auth]', ...a);

function decodeJwt(token) {
  try {
    const payload = JSON.parse(Buffer.from(token.split('.')[1], 'base64').toString());
    return payload;
  } catch {
    return null;
  }
}

function tokenFromStorageState() {
  if (!fs.existsSync(CONFIG.storageState)) return null;
  try {
    const state = JSON.parse(fs.readFileSync(CONFIG.storageState, 'utf8'));
    for (const origin of state.origins || []) {
      const entry = (origin.localStorage || []).find((x) => x.name === 'access_token');
      if (entry) return entry.value;
    }
  } catch {}
  return null;
}

function isValid(token, skewSeconds = 120) {
  if (!token) return false;
  const payload = decodeJwt(token);
  if (!payload?.exp) return false;
  return payload.exp * 1000 > Date.now() + skewSeconds * 1000;
}

// Log in with Playwright and persist storage state (cookies + localStorage token).
export async function login() {
  if (!CONFIG.user || !CONFIG.pass) {
    throw new Error('Missing credentials. Set XDIAL_USER and XDIAL_PASS in .env');
  }
  log('logging in as', CONFIG.user);
  const browser = await chromium.launch({ headless: true });
  try {
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    await page.goto(CONFIG.baseUrl, { waitUntil: 'networkidle', timeout: 60000 });
    await page.locator('input[name="username"], input[type="text"]').first().fill(CONFIG.user);
    await page.locator('input[name="password"], input[type="password"]').first().fill(CONFIG.pass);
    await Promise.all([
      page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => {}),
      page.locator('button[type="submit"], button:has-text("Sign In"), button:has-text("Login")').first().click(),
    ]);
    await page.waitForTimeout(2500);

    const token = await page.evaluate(() => localStorage.getItem('access_token'));
    if (!token) throw new Error('Login appears to have failed: no access_token in localStorage.');
    await ctx.storageState({ path: CONFIG.storageState });
    log('login OK, token saved. landed on', page.url());
    return token;
  } finally {
    await browser.close();
  }
}

// Return a valid Bearer token, refreshing via login() only when needed.
export async function getToken() {
  let token = tokenFromStorageState();
  if (isValid(token)) {
    const exp = new Date(decodeJwt(token).exp * 1000).toISOString();
    log('reusing saved token (valid until', exp + ')');
    return token;
  }
  log('no valid saved token, logging in...');
  return login();
}
