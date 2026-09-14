import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const webDir = path.dirname(new URL(import.meta.url).pathname);
const reportDir = path.resolve(webDir, '..');
const sourcePath = path.join(reportDir, 'main.tex');
const stylePath = path.join(webDir, 'style.css');

const figureMap = {
  order: 'figordered',
  contacts: 'figcontacts',
  unfold: 'figunfold',
  last: 'figlastmap',
  binary: 'figbinary',
  split: 'figsplit',
  limits: 'figlimits',
  rays: 'figrays',
  backwards: 'figbackwards',
  virtual: 'figvirtual',
  membership: 'figmembership',
  path: 'figpath',
};

const sourceSymbols = new Set([
  'normalized_winding', '_locate_point', 'locate_point_linear', 'locate_point',
  'build_cone', 'query', 'query_full', 'query_length', 'DirectionalMaps',
  'split_boundaries', 'cross_sign', 'dot_sign', 'inside', 'build_vertex',
  'in_cone', 'in_edge_plus', 'locate', 'virtual_source', 'reflect_query',
  'query_path', 'reflect_point', 'append', 'distance', 'prefix_length',
  'Solution::build_cone', 'Solution::solve', 'Solution::query',
  'SolutionBinarySearchDisjoint', 'point_in_cone_plus', 'point_in_edge_plus',
  'DirectionalMaps::solve', 'solve_intersecting_maps',
]);

const citationLabels = {
  dror03: 'Dror et al., 2003',
  tanjiang17: 'Tan & Jiang, 2017',
  counterexample: 'counterexample report',
  correction: 'correction report',
  audit: 'historical audit',
  code: 'source snapshot',
};

function preprocessLatex(source) {
  return source
    .replace(/\\input\{figures\.tex\}/g, '')
    .replace(/\\begin\{sourcebox\}/g, '\\begin{quote}')
    .replace(/\\end\{sourcebox\}/g, '\\end{quote}')
    .replace(/\\begin\{coverbox\}/g, '\\begin{quote}')
    .replace(/\\end\{coverbox\}/g, '\\end{quote}')
    .replace(/\\begin\{pseudo\}/g, '\\begin{verbatim}')
    .replace(/\\end\{pseudo\}/g, '\\end{verbatim}')
    .replace(/\\code\{([^{}]*)\}/g, '\\texttt{$1}')
    .replace(/\\path\{([^{}]*)\}/g, '\\texttt{$1}')
    .replace(/\\Needspace\{[^{}]*\}/g, '')
    .replace(/\\rm\s+double/g, '\\mathrm{double}');
}

function runPandoc() {
  const source = preprocessLatex(fs.readFileSync(sourcePath, 'utf8'));
  const result = spawnSync('pandoc', [
    '--from=latex',
    '--to=html5',
    '--standalone',
    '--math-method=mathml',
    '--metadata',
    'title=Fixed-order convex polygon tours',
  ], { input: source, encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(result.stderr || 'pandoc failed');
  return result.stdout;
}

function extractBody(html) {
  const match = html.match(/<body>([\s\S]*?)<\/body>/i);
  if (!match) throw new Error('Pandoc output did not contain a body');
  return match[1]
    .replace(/<header id="title-block-header">[\s\S]*?<\/header>/i, '')
    .replace(/<p>height \.5pt<\/p>/g, '')
    .replace(/READING MAP\s+14 SECTIONS\s*•\s*FIGURES/g, 'READING MAP 14 SECTIONS • 12 FIGURES');
}

function addHero(body) {
  const hero = '<div class="hero-figure"><img src="assets/figures/path.svg" alt="Static rendering of the report route illustration"></div>';
  return body.replace(/<div class="center">\s*<\/div>/, hero);
}

function addEquationAnchors(body) {
  body = body.replace(/<math display="block"[\s\S]*?<\/math>/g, (math) => {
    const label = math.match(/\\label\{([^}]+)\}/);
    const anchored = label
      ? math.replace('<math display="block"', `<math id="${label[1]}" class="equation" display="block"`)
      : math;
    return `<span class="equation-scroll">${anchored}</span>`;
  });
  return body.replace(/<math display="inline"[\s\S]*?<\/math>/g, (math) => `<span class="inline-math">${math}</span>`);
}

function addSourceIdentifiers(body) {
  body = body.replace(/<code>((?:IM|BS|SOL|VAL|GEO):[0-9]+(?:–|--)[0-9]+)<\/code>/g, (_, ref) => {
    const safe = ref.replace(/[^A-Za-z0-9]+/g, '-').replace(/-+$/, '').toLowerCase();
    return `<code class="source-ref" data-source-ref="${ref}" data-source-anchor="source-${safe}">${ref}</code>`;
  });

  body = body.replace(/<code>([^<]+)<\/code>/g, (whole, value) => {
    if (!sourceSymbols.has(value)) return whole;
    return `<code class="function-ref" data-source-symbol="${value}">${value}</code>`;
  });

  let noteNumber = 0;
  return body.replace(/<blockquote>([\s\S]*?)<\/blockquote>/g, (whole, inner) => {
    if (/Code key\./.test(inner)) return `<blockquote class="key-note">${inner}</blockquote>`;
    if (/What this guide does\./.test(inner)) return `<blockquote class="cover-note">${inner}</blockquote>`;
    if (!/Open the code:/.test(inner)) return whole;
    noteNumber += 1;
    const refs = [...inner.matchAll(/data-source-ref="([^"]+)"/g)].map((match) => match[1]);
    const data = refs.length ? ` data-source-refs="${refs.join(' ')}"` : '';
    return `<blockquote class="source-note" id="source-note-${noteNumber}"${data}>${inner}</blockquote>`;
  });
}

function addFigures(body) {
  return body.replace(/<figure id="fig:([^"]+)"[^>]*>([\s\S]*?)<\/figure>/g, (whole, key, inner) => {
    const stableId = figureMap[key];
    if (!stableId) throw new Error(`No static asset mapping for figure ${key}`);
    const caption = inner.match(/<figcaption>([\s\S]*?)<\/figcaption>/);
    if (!caption) throw new Error(`Figure ${key} has no caption`);
    const scope = /schematic|generic illustration/i.test(caption[1]) ? 'schematic scope in caption' : 'static source drawing';
    return `<figure id="fig:${key}" class="report-figure" data-figure-id="${stableId}" data-figure-scope="${scope}">
  <div class="figure-frame"><img src="assets/figures/${key}.svg" alt="Static rendering for ${stableId}; see the complete caption below"></div>
  <div class="figure-toolbar"><span class="figure-id">${stableId}</span><button class="expand-placeholder" type="button" disabled aria-disabled="true" title="Interactive expansion is scheduled for Phase 2">Expand · Phase 2</button></div>
  <figcaption>${caption[1]}</figcaption>
</figure>`;
  });
}

function highlightCodeLine(line) {
  const commentStart = line.indexOf('//');
  const codePart = commentStart >= 0 ? line.slice(0, commentStart) : line;
  const commentPart = commentStart >= 0 ? line.slice(commentStart) : '';
  let highlighted = codePart
    .replace(/\b(PUBLIC_SOLVE|SPLIT_BOUNDARIES|LEX_SIGN|INSIDE_CLOSED|BUILD_VERTEX|LOCATE|VIRTUAL_SOURCE|QUERY_PATH|QUERY_LENGTH|CONE|EDGE_PLUS|CROSS_SIGN|REFLECT_POINT|REFLECT_QUERY|REFLECT_DIRECTION)\b/g, '<span class="tok-command">$1</span>')
    .replace(/\b(if|else|for|while|return|skip|throw|require|not|and|or|in|has|each|append|mark|cache|copy|reverse|construct|replace|remove|sort|unique)\b/g, '<span class="tok-keyword">$1</span>');
  if (commentPart) highlighted += `<span class="tok-comment">${commentPart}</span>`;
  return highlighted;
}

function addCodeLineNumbers(body) {
  let blockNumber = 0;
  return body.replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/g, (_, raw) => {
    blockNumber += 1;
    const lines = raw.replace(/^\n/, '').replace(/\n$/, '').split('\n');
    const numbered = lines.map((line, index) => `<span class="code-line"><span class="line-no">${String(index + 1).padStart(2, '0')}</span><span class="line-text">${highlightCodeLine(line)}</span></span>`).join('\n');
    return `<pre class="pseudo-code" data-pseudocode-block="${blockNumber}" aria-label="Pseudocode block ${blockNumber}"><code>${numbered}</code></pre>`;
  });
}

function addCitationLinks(body) {
  return body.replace(/<span\b[^>]*class="citation"[^>]*data-cites="([^"]+)"[^>]*>\s*<\/span>/g, (_, keys) => keys.split(/\s+/).map((key) => {
    const label = citationLabels[key] || key;
    return `<a class="citation" href="#ref-${key}">[${label}]</a>`;
  }).join(' '));
}

function addBibliographyHeadingAndAnchors(body) {
  const keys = ['dror03', 'tanjiang17', 'counterexample', 'correction', 'audit', 'code'];
  const marker = '<div class="thebibliography">';
  const start = body.indexOf(marker);
  if (start < 0) throw new Error('Pandoc output did not contain the bibliography');
  const end = body.indexOf('</div>', start);
  if (end < 0) throw new Error('Bibliography was not closed');
  let bibliography = body.slice(start, end + '</div>'.length)
    .replace(/<span>9<\/span>\s*/, '');
  let index = 0;
  bibliography = bibliography.replace(/<p>/g, () => `<p id="ref-${keys[index++] || `entry-${index}`}">`);
  return `${body.slice(0, start)}<h1 id="references">References</h1>${bibliography}${body.slice(end + '</div>'.length)}`;
}

function numberHeadingsAndBuildToc(body) {
  const sections = [];
  let sectionNumber = 0;
  body = body.replace(/<h1([^>]*)>([\s\S]*?)<\/h1>/g, (whole, attrs, title) => {
    const idMatch = attrs.match(/id="([^"]+)"/);
    const id = idMatch ? idMatch[1] : '';
    if (id === 'references') {
      sections.push({ id, title: 'References', number: null, subsections: [] });
      return `<h1${attrs}>References</h1>`;
    }
    sectionNumber += 1;
    sections.push({ id, title, number: sectionNumber, subsections: [] });
    return `<h1${attrs}><span class="section-number">${sectionNumber}</span>${title}</h1>`;
  });
  let subsectionNumber = 0;
  body = body.replace(/<h2([^>]*)>([\s\S]*?)<\/h2>/g, (whole, attrs, title) => {
    subsectionNumber += 1;
    const number = `8.${subsectionNumber}`;
    const implementationSection = sections.find((section) => section.number === 8);
    if (implementationSection) implementationSection.subsections.push({ id: (attrs.match(/id="([^"]+)"/) || [])[1] || '', title, number });
    return `<h2${attrs}><span class="subsection-number">${number}</span>${title}</h2>`;
  });
  const toc = `<h2>Contents</h2><p class="toc-meta">14 sections · 4 subsections · 12 figures</p><ol>${sections.map((section) => {
    const label = section.number ? `<span class="toc-number">${section.number}</span> ${section.title}` : section.title;
    const children = section.subsections.length ? `<ol class="toc-sublist">${section.subsections.map((sub) => `<li><a href="#${sub.id}"><span class="toc-number">${sub.number}</span> ${sub.title}</a></li>`).join('')}</ol>` : '';
    return `<li><a href="#${section.id}">${label}</a>${children}</li>`;
  }).join('')}</ol>`;
  return { body, toc };
}

function buildPage() {
  for (const key of Object.keys(figureMap)) {
    const asset = path.join(webDir, 'assets', 'figures', `${key}.svg`);
    if (!fs.existsSync(asset)) throw new Error(`Missing figure asset: ${asset}`);
  }
  let body = extractBody(runPandoc());
  body = addHero(body);
  body = addEquationAnchors(body);
  body = addSourceIdentifiers(body);
  body = addFigures(body);
  body = addCodeLineNumbers(body);
  body = addCitationLinks(body);
  body = addBibliographyHeadingAndAnchors(body);
  const { body: numberedBody, toc } = numberHeadingsAndBuildToc(body);
  body = numberedBody;
  const style = fs.readFileSync(stylePath, 'utf8');
  const figureCount = (body.match(/class="report-figure"/g) || []).length;
  const codeCount = (body.match(/class="pseudo-code"/g) || []).length;
  if (figureCount !== 12) throw new Error(`Expected 12 figures, found ${figureCount}`);
  if (codeCount !== 5) throw new Error(`Expected 5 pseudocode blocks, found ${codeCount}`);
  const page = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="A static visual guide to fixed-order convex polygon tours and directional last-step maps.">
<title>Fixed-order convex polygon tours · static report</title>
<style>${style}</style>
</head>
<body>
<a class="skip-link" href="#report">Skip to report</a>
<header class="site-header">
  <div class="brand">Directional touring-polygon maps <span>· static edition</span></div>
  <nav class="header-links" aria-label="Page links"><a href="#report">Report</a><a href="#references">References</a><a href="README.md">Build notes</a></nav>
</header>
<div class="layout">
  <aside class="toc-panel" id="toc" aria-label="Table of contents">${toc}</aside>
  <main class="report" id="report">${body}</main>
</div>
</body>
</html>
`;
  fs.writeFileSync(path.join(webDir, 'index.html'), page);
}

buildPage();
