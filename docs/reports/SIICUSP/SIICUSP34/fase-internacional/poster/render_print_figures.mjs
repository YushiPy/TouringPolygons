/** The Brazil figure of the poster, for printing on paper: a light background, no web styling.
 *
 * figures/brazil-route.pdf is the Brazil TSPN tour of apps/siicusp34/en/data/br-estados-tspn.js,
 * drawn from its geometry, tour and visiting order (nothing is recomputed). Chrome prints the SVG
 * to PDF and pdfcrop trims the margins. The method cards come from
 * apps/siicusp34/scripts/build_tspn_cards.py. From the repository root, with the dashboard's
 * Playwright installed:
 *
 *   node docs/reports/SIICUSP/SIICUSP34/fase-internacional/poster/render_print_figures.mjs
 */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const directory = dirname(fileURLToPath(import.meta.url));
const root = resolve(directory, '../../../../../..');
const require = createRequire(resolve(root, 'apps/benchmark-dashboard/package.json'));
const { chromium } = require('playwright');
const figures = resolve(directory, 'figures');

const ink = '#142d38', accent = '#cf5a20', teal = '#1f7a73', pale = '#eef4f1';

function brazilSvg(demo) {
  const size = 1000, margin = 30;
  const points = [...demo.geometry.polygons.flat(), ...demo.path];
  const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = (size - 2 * margin) / Math.max(maxX - minX, maxY - minY);
  const offsetX = (size - (maxX - minX) * scale) / 2, offsetY = (size - (maxY - minY) * scale) / 2;
  const project = ([x, y]) => [offsetX + (x - minX) * scale, size - offsetY - (y - minY) * scale];
  const coordinates = (list) => list.map((point) => project(point).map((value) => value.toFixed(2)).join(',')).join(' ');
  // A TSPN tour has no first visit; the data starts it at the Federal District (Brasília), lighter first.
  const shade = (rank) => {
    const t = rank / (demo.polygons - 1);
    const mix = (a, b) => Math.round(a + (b - a) * t);
    return `rgb(${mix(146, 38)},${mix(204, 122)},${mix(188, 108)})`;
  };
  const regions = demo.geometry.polygons.map((polygon, index) =>
    `<polygon points="${coordinates(polygon)}" fill="${shade(demo.order.indexOf(index))}" stroke="#ffffff" stroke-width="2" stroke-linejoin="round"/>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">${regions}`
    + `<polyline points="${coordinates(demo.path)}" fill="none" stroke="#a63f0d" stroke-width="8" stroke-linejoin="round" stroke-linecap="round"/>`
    + '</svg>';
}

async function printSvg(page, svg, output) {
  const [, , width, height] = svg.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number);
  await page.setContent(`<!doctype html><style>@page{margin:0}html,body{margin:0}svg{display:block;width:${width}px;height:${height}px}</style>${svg}`);
  await page.pdf({ path: output, width: `${width}px`, height: `${height}px`, printBackground: true, pageRanges: '1' });
  // The card SVGs leave wide empty margins; crop every figure to its drawing (pdfcrop, from TeX Live).
  execFileSync('pdfcrop', ['--margins', '4', output, output], { stdio: 'ignore' });
}

await mkdir(figures, { recursive: true });
// Without Playwright's own Chromium, fall back to the installed Google Chrome.
const browser = await chromium.launch({ headless: true }).catch(() => chromium.launch({ headless: true, channel: 'chrome' }));
try {
  const page = await browser.newPage();
  const source = await readFile(resolve(root, 'apps/siicusp34/en/data/br-estados-tspn.js'), 'utf8');
  const demo = JSON.parse(source.slice(source.indexOf('=') + 1).trim().replace(/;$/, ''));
  assert.equal(demo.polygons, 27);
  assert.equal(demo.cycle, true);
  assert.equal(demo.termination, 'optimal');
  await printSvg(page, brazilSvg(demo), resolve(figures, 'brazil-route.pdf'));
  console.log('ok');
} finally {
  await browser.close();
}
