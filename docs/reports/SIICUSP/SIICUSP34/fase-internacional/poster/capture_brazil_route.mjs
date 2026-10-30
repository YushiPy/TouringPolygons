/** Capture the Brazil tour of the English SIICUSP page; never redraw or restyle its geometry.
 * Playwright serves the requests from apps/siicusp34, so no server is needed.
 * From the repository root, with the dashboard's Playwright installed:
 *
 *   node docs/reports/SIICUSP/SIICUSP34/fase-internacional/poster/capture_brazil_route.mjs
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const directory = dirname(fileURLToPath(import.meta.url));
const root = resolve(directory, '../../../../../..');
const require = createRequire(resolve(root, 'apps/benchmark-dashboard/package.json'));
const { chromium } = require('playwright');
const url = 'http://127.0.0.1:8773/en/';
const output = resolve(directory, 'figures/brazil-route-square.png');
const appRoot = resolve(root, 'apps/siicusp34');
const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};
await mkdir(dirname(output), { recursive: true });
// Without Playwright's own Chromium, fall back to the installed Google Chrome.
const browser = await chromium.launch({ headless: true }).catch(() => chromium.launch({ headless: true, channel: 'chrome' }));
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 6, reducedMotion: 'reduce' });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route(`${new URL(url).origin}/**`, async route => {
    const pathname = decodeURIComponent(new URL(route.request().url()).pathname);
    const file = resolve(appRoot, pathname.endsWith('/') ? `.${pathname}index.html` : `.${pathname}`);
    assert(file.startsWith(`${appRoot}${sep}`), 'App request must stay inside apps/siicusp34.');
    try {
      await route.fulfill({ status: 200, body: await readFile(file), contentType: contentTypes[extname(file)] || 'application/octet-stream' });
    } catch {
      await route.fulfill({ status: 404, body: 'Not found' });
    }
  });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.locator('#animated-route').waitFor();
  await page.evaluate(() => document.fonts.ready);
  assert.equal(await page.locator('#route-progress').inputValue(), '1000');
  // Hide the floating controls and the legend; the poster text explains the scene.
  await page.addStyleTag({ content: '.map-camera-tools, .map-legend { visibility: hidden !important; }' });
  const scene = await page.locator('#route-map').evaluate(map => {
    const content = map.querySelector('#map-content').getBoundingClientRect();
    const box = map.getBoundingClientRect();
    const demo = window.TPPBrEstadosDemo;
    return {
      content: { x: content.x - box.x, y: content.y - box.y, width: content.width, height: content.height },
      map: { width: box.width, height: box.height },
      regionCount: map.querySelectorAll('polygon.region').length,
      lengthMetres: demo.length,
      lowerBound: demo.lower_bound,
      upperBound: demo.upper_bound,
      termination: demo.termination,
      seconds: demo.seconds,
    };
  });
  assert.equal(scene.regionCount, 27);
  assert.equal(scene.termination, 'optimal');
  assert.deepEqual(errors, []);
  // A square around every region and the whole tour, with a small margin, inside the map.
  const clip = await page.locator('#route-map').boundingBox();
  const side = Math.min(scene.map.height, Math.max(scene.content.width, scene.content.height) + 24);
  const centerX = scene.content.x + scene.content.width / 2, centerY = scene.content.y + scene.content.height / 2;
  const square = {
    x: clip.x + Math.max(0, Math.min(scene.map.width - side, centerX - side / 2)),
    y: clip.y + Math.max(0, Math.min(scene.map.height - side, centerY - side / 2)),
    width: side,
    height: side,
  };
  await page.screenshot({ path: output, clip: square, animations: 'disabled' });
  const png = await readFile(output);
  const sha256 = buffer => createHash('sha256').update(buffer).digest('hex');
  const sources = {};
  for (const file of ['en/index.html', 'en/app.js', 'styles.css', 'en/en.css', 'data/br-estados-demo.js']) {
    sources[`apps/siicusp34/${file}`] = sha256(await readFile(resolve(appRoot, file)));
  }
  const metadata = {
    renderer: 'Chromium screenshot of #route-map of apps/siicusp34/en/ after app.js executes',
    browserVersion: browser.version(),
    viewport: { width: 1440, height: 1100, deviceScaleFactor: 6 },
    squareCropCSS: { x: square.x - clip.x, y: square.y - clip.y, side },
    pixels: { width: png.readUInt32BE(16), height: png.readUInt32BE(20) },
    scene, sources, imageSha256: sha256(png),
  };
  await writeFile(resolve(directory, 'figures/brazil-route-capture.json'), JSON.stringify(metadata, null, 2) + '\n');
  console.log(JSON.stringify({ output, ...metadata.pixels, regions: scene.regionCount }));
} finally {
  await browser.close();
}
