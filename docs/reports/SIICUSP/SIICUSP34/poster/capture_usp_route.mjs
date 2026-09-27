/** Capture the real SIICUSP app; never redraw or restyle its geometry.
 * Playwright serves requests from the local app files, so no server is needed.
 * Optional argument: local app URL (default http://127.0.0.1:8773/).
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const directory = dirname(fileURLToPath(import.meta.url));
const root = resolve(directory, '../../../../..');
const require = createRequire(resolve(root, 'apps/benchmark-dashboard/package.json'));
const { chromium } = require('playwright');
const url = process.argv[2] || 'http://127.0.0.1:8773/';
assert(['127.0.0.1', 'localhost'].includes(new URL(url).hostname), 'Use the local frozen app.');
const output = resolve(directory, 'figures/usp-route-print.png');
const squareOutput = resolve(directory, 'figures/usp-route-square.png');
const appRoot = resolve(root, 'apps/siicusp34');
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
    deviceScaleFactor: 6,
    reducedMotion: 'reduce',
  });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const contentTypes = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.woff2': 'font/woff2',
  };
  await page.route(`${new URL(url).origin}/**`, async route => {
    const pathname = decodeURIComponent(new URL(route.request().url()).pathname);
    const file = resolve(appRoot, pathname === '/' ? 'index.html' : `.${pathname}`);
    assert(file.startsWith(`${appRoot}${sep}`), 'App request must stay inside apps/siicusp34.');
    try {
      await route.fulfill({
        status: 200,
        body: await readFile(file),
        contentType: contentTypes[extname(file)] || 'application/octet-stream',
      });
    } catch {
      await route.fulfill({ status: 404, body: 'Not found' });
    }
  });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.locator('#animated-route').waitFor();
  await page.evaluate(() => document.fonts.ready);
  assert.equal(await page.locator('#route-progress').inputValue(), '1000');
  assert.equal(await page.locator('#show-labels').getAttribute('aria-pressed'), 'false');
  assert.equal(
    await page.locator('.map-legend').innerText(),
    'Caminho encontrado\ns partida · t chegada\nCor: primeiras → últimas visitas',
  );
  assert.equal(await page.locator('.legend-visited').count(), 0);
  for (const id of ['show-contacts', 'show-hulls', 'show-decomposition']) {
    assert.equal(await page.locator(`#${id}`).getAttribute('aria-pressed'), 'false');
  }
  // The screenshot crops to the map. Hide the floating camera control only.
  // No changes to SVG elements, projection, geometry, route, strokes or colors.
  await page.addStyleTag({ content: '.map-camera-tools { visibility: hidden !important; }' });
  const scene = await page.locator('#route-map').evaluate(map => {
    const path = map.querySelector('#animated-route');
    const style = getComputedStyle(path);
    const demo = window.TPPUspDemo;
    return {
      viewBox: map.getAttribute('viewBox'),
      width: map.getBoundingClientRect().width,
      height: map.getBoundingClientRect().height,
      regionCount: map.querySelectorAll('polygon.region').length,
      routePoints: path.getAttribute('points'),
      routeStroke: style.stroke,
      routeStrokeWidth: style.strokeWidth,
      routeFilter: style.filter,
      background: getComputedStyle(map.closest('.drawing-panel')).backgroundColor,
      lengthMeters: demo.length,
      lowerBound: demo.lower_bound,
      upperBound: demo.upper_bound,
      termination: demo.termination,
    };
  });
  assert.equal(scene.regionCount, 51);
  assert.equal(scene.viewBox, '0 0 840 480');
  assert.equal(scene.width, 1008);
  assert.equal(scene.height, 576);
  assert.equal(scene.routeStroke, 'rgb(255, 173, 102)');
  assert.equal(scene.routeStrokeWidth, '3px');
  assert.equal(scene.routeFilter, 'none');
  assert.equal(scene.termination, 'optimal');
  const clip = await page.locator('#route-map').boundingBox();
  const legend = await page.locator('.map-legend').boundingBox();
  assert(legend.x >= clip.x + 216 && legend.x + legend.width <= clip.x + 792,
    'Centered legend must fit completely inside the square crop.');
  assert(legend.y + legend.height <= clip.y + clip.height,
    'Legend must fit inside the map image.');
  assert.deepEqual(errors, []);
  await page.screenshot({ path: output, clip, animations: 'disabled' });
  // Crop the real scene to a square around every region and the complete route.
  // At 6 device pixels per CSS pixel, this is x=216..792 and y=0..576.
  const squareClip = { x: clip.x + 216, y: clip.y, width: 576, height: 576 };
  await page.screenshot({ path: squareOutput, clip: squareClip, animations: 'disabled' });
  const png = await readFile(output);
  const pixels = { width: png.readUInt32BE(16), height: png.readUInt32BE(20) };
  assert.equal(pixels.width, 6048);
  assert.equal(pixels.height, 3456);
  const squarePng = await readFile(squareOutput);
  const squarePixels = { width: squarePng.readUInt32BE(16), height: squarePng.readUInt32BE(20) };
  assert.deepEqual(squarePixels, { width: 3456, height: 3456 });
  const sha256 = buffer => createHash('sha256').update(buffer).digest('hex');
  const sources = {};
  for (const file of ['index.html', 'app.js', 'styles.css', 'data/usp-demo.js']) {
    sources[`apps/siicusp34/${file}`] = sha256(await readFile(resolve(root, 'apps/siicusp34', file)));
  }
  const metadata = {
    renderer: 'Chromium screenshot of #route-map with centered .map-legend after apps/siicusp34/app.js executes',
    browserVersion: browser.version(),
    viewport: { width: 1440, height: 1100, deviceScaleFactor: 6 },
    pixels,
    squareCropCSS: { x: 216, y: 0, width: 576, height: 576 },
    squarePixels,
    scene, sources, imageSha256: sha256(png),
    squareImageSha256: sha256(squarePng),
  };
  await writeFile(resolve(directory, 'figures/usp-route-capture.json'), JSON.stringify(metadata, null, 2) + '\n');
  console.log(JSON.stringify({ output, ...metadata.pixels, regions: scene.regionCount, errors }));
} finally {
  await browser.close();
}
