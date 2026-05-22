// Tiny .env loader (no external deps) + shared config.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(__dirname, '..');

function loadEnv() {
  const envPath = path.join(ROOT, '.env');
  if (!fs.existsSync(envPath)) return;
  const text = fs.readFileSync(envPath, 'utf8');
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const idx = trimmed.indexOf('=');
    if (idx === -1) continue;
    const key = trimmed.slice(0, idx).trim();
    const val = trimmed.slice(idx + 1).trim();
    if (!(key in process.env)) process.env[key] = val;
  }
}
loadEnv();

export const CONFIG = {
  baseUrl: process.env.XDIAL_BASE_URL || 'https://dashboard.xlitexcore.xdialnetworks.com',
  user: process.env.XDIAL_USER,
  pass: process.env.XDIAL_PASS,
  campaignId: process.env.XDIAL_CAMPAIGN_ID || '270',
  dataDir: path.join(ROOT, 'data'),
  shotDir: path.join(ROOT, 'screenshots'),
  storageState: path.join(ROOT, '.auth.json'),
};
