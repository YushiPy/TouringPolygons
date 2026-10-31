/** Vector figures of the poster for printing on paper: light backgrounds, no web styling.
 *
 * - figures/brazil-route.pdf: the Brazil TSPN tour of apps/siicusp34/en/data/br-estados-tspn.js,
 *   drawn from its geometry, tour and visiting order (nothing is recomputed);
 * - figures/method-step-0N.pdf: the six method cards of the first-stage poster
 *   (../../fase-local/poster/figures/method-step-0N.svg), with a light palette and English labels.
 *
 * Chrome prints each SVG to PDF, so the cards keep their CSS, and pdfcrop trims the margins. From the repository root, with the
 * dashboard's Playwright installed:
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
const cards = resolve(directory, '../../fase-local/poster/figures');

const ink = '#142d38', accent = '#cf5a20', teal = '#1f7a73', pale = '#eef4f1';

// Overrides for the dark card palette of the web page; later rules with !important win.
const lightCards = `
svg.algorithm-flow text { fill:${ink} !important; }
.diagram-region, .diagram-nonconvex-region { fill:#bfe3d8 !important; fill-opacity:1 !important; stroke:${teal} !important; }
.diagram-region-incidental { fill:#f3d9a4 !important; stroke:#a87b22 !important; }
.diagram-region-unvisited { fill:#e6edeb !important; stroke:#7f9aa0 !important; }
.diagram-heuristic-route, .diagram-refined-route, .diagram-route, .diagram-branch { stroke:${accent} !important; }
.diagram-optimal-route { stroke:${ink} !important; }
.diagram-endpoint { fill:${ink} !important; }
.algorithm-flow .diagram-polygon-label, .diagram-region-label { fill:${ink} !important; stroke:#ffffff !important; }
.diagram-piece-fill { fill:#9fd3c5 !important; fill-opacity:.7 !important; stroke:${teal} !important; }
.diagram-piece-fill:nth-child(even) { fill:#d3ece5 !important; fill-opacity:1 !important; }
.diagram-piece-chosen { fill:#f6b38a !important; fill-opacity:1 !important; }
.diagram-piece-cut { stroke:${teal} !important; }
.diagram-region-outline { stroke:${teal} !important; }
.diagram-node { fill:${pale} !important; stroke:${teal} !important; }
.diagram-node-child { stroke:${accent} !important; }
.diagram-axis { stroke:#7f9aa0 !important; }
.diagram-upper { stroke:${teal} !important; }
.diagram-lower { stroke:${accent} !important; }
.legend-line-solid { stroke:${accent} !important; }
.legend-line-dashed { stroke:${ink} !important; }
svg.algorithm-flow text.legend-text { fill:${ink} !important; }
`;
const english = [['Heurística · 122,65 m', 'Heuristic · 122.65 m'], ['Ótimo · 100 m', 'Optimum · 100 m']];

function brazilSvg(demo) {
  const size = 1000, margin = 30;
  const points = [...demo.geometry.polygons.flat(), ...demo.path];
  const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = (size - 2 * margin) / Math.max(maxX - minX, maxY - minY);
  const offsetX = (size - (maxX - minX) * scale) / 2, offsetY = (size - (maxY - minY) * scale) / 2;
  const project = ([x, y]) => [offsetX + (x - minX) * scale, size - offsetY - (y - minY) * scale];
  const coordinates = (list) => list.map((point) => project(point).map((value) => value.toFixed(2)).join(',')).join(' ');
  // A TSPN tour has no first visit; the colours follow the visiting order from the Federal District
  // (Brasília), lighter first, so the map reads like the page's gradient.
  const federalDistrict = demo.region_codes.indexOf('DF');
  assert(federalDistrict >= 0);
  const start = demo.order.indexOf(federalDistrict);
  const rotated = [...demo.order.slice(start), ...demo.order.slice(0, start)];
  const shade = (rank) => {
    const t = rank / (demo.polygons - 1);
    const mix = (a, b) => Math.round(a + (b - a) * t);
    return `rgb(${mix(146, 38)},${mix(204, 122)},${mix(188, 108)})`;
  };
  const regions = demo.geometry.polygons.map((polygon, index) =>
    `<polygon points="${coordinates(polygon)}" fill="${shade(rotated.indexOf(index))}" stroke="#ffffff" stroke-width="2" stroke-linejoin="round"/>`).join('');
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
  for (let step = 1; step <= 6; step += 1) {
    let svg = await readFile(resolve(cards, `method-step-0${step}.svg`), 'utf8');
    for (const [from, to] of english) svg = svg.replace(from, to);
    assert(!/[ÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]/.test(svg.replace(/aria-label="[^"]*"/g, '')), `card ${step} still has Portuguese text`);
    svg = svg.replace('</style>', `${lightCards}</style>`);
    await printSvg(page, svg, resolve(figures, `method-step-0${step}.pdf`));
  }
  console.log('ok');
} finally {
  await browser.close();
}
