# Poster A0 — 34th SIICUSP, International Stage

English poster for the poster session of 25–26/11/2026 (edital, item 4.5:
material in English). The title is the one of the international abstract,
[`../resumo/resumo-en.tex`](../resumo/resumo-en.tex), which is also the
reference for numbers and wording. The first-stage poster stays unchanged in
[`../../fase-local/poster/`](../../fase-local/poster/).

## Build

Requires XeLaTeX, Arial, Latin Modern Math and the packages of the preamble.
The speedup figure is the abstract's (`../resumo/assets/speedup-figure.pdf`,
ignored); build it first with the abstract's generator, and the other
figures with `render_print_figures.mjs` (below); all PDFs here are ignored too (see
[`../../README.md`](../../README.md)). From this folder:

```sh
mkdir -p /tmp/siicusp34-poster-en
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/siicusp34-poster-en poster.tex
cp /tmp/siicusp34-poster-en/poster.pdf poster.pdf
```

`poster.pdf` is ignored by Git. Each block writes `POSTERBLOCK name x y w h`
to the log, to check overflow after an edit.

## Figures

The poster is printed, so its figures are drawn on white:

- `figures/brazil-route.pdf`: the Brazil TSPN tour of
  `apps/siicusp34/en/data/br-estados-tspn.js` (27 federative units), drawn from
  its geometry and tour, in one colour since a TSPN tour has no first visit.
- `figures/method-step-0N.pdf`: the six method cards of
  [`../../fase-local/poster/figures/`](../../fase-local/poster/figures/README.md),
  with a light palette and English labels.

Both come from [`render_print_figures.mjs`](render_print_figures.mjs), which
prints SVG to PDF with Playwright (or the installed Google Chrome) and trims
the margins with `pdfcrop`. From the repository root, with the dashboard's
Playwright installed:

```sh
node docs/reports/SIICUSP/SIICUSP34/fase-internacional/poster/render_print_figures.mjs
```

The speedup figure is Figure 2 of the abstract (`../resumo/assets/`); the logos
come from the first stage (`../../fase-local/poster/figures/` and
`../../fase-local/resumo/assets/`).

## Numbers

From the armijo reference measurement
([`fekete-reference-armijo-2026-10-17`](../../../../../../benchmarks/results-saved/README.md#fekete-reference-armijo-2026-10-17)):
Fekete et al. with one thread (configuration B, CHR root) against ours with one
thread (C). On the 541 instances both solve: median speedup 72.7× (73×),
arithmetic mean 196×, ours faster on all. Ours solves 557 in 300 s and case 189
in 5363 s; in the paper's configuration Fekete et al. solve 542 in 300 s and 549
in 900 s. The 257 = 1.08 × 238 split is in geometric means, as in the abstract.
The Brazil tour is a TSPN (no fixed point): 9,962 km, 0.016 s (median of three runs, one thread, Mac), from
`apps/siicusp34/en/data/br-estados-tspn.js` (`apps/siicusp34/scripts/build_br_estados_tspn.py`).
The Portuguese page's tour, which starts and ends in Brasília, is 10,039 km.

The QR code points to `https://yushipy.github.io/TouringPolygons/apps/siicusp34/en/`,
which only works once the English page is merged and published.
