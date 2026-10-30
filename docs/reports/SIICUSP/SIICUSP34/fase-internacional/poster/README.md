# Poster A0 — 34th SIICUSP, International Stage

English poster for the poster session of 25–26/11/2026 (edital, item 4.5:
material in English). The title is the one of the international abstract,
[`../resumo/resumo-en.tex`](../resumo/resumo-en.tex), which is also the
reference for numbers and wording. The first-stage poster stays unchanged in
[`../../fase-local/poster/`](../../fase-local/poster/).

## Build

Requires XeLaTeX, Arial, Latin Modern Math and the packages of the preamble.
Figures 1 and 2 are the abstract's (`../resumo/assets/*.pdf`, ignored); build
them first with the abstract's generators (see
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

- `figures/brazil-route-square.png`: screenshot of the Brazil tour as the
  English page `apps/siicusp34/en/` draws it (27 federative units), made by
  [`capture_brazil_route.mjs`](capture_brazil_route.mjs) with Playwright;
  `figures/brazil-route-capture.json` records the crop, the scene and the
  hashes of the sources. The page's legend and zoom control are hidden.
- Logos and the three method cards (`method-step-03`, `-05`, `-06`) come from
  [`../../fase-local/poster/figures/`](../../fase-local/poster/figures/README.md);
  the English SIICUSP logo comes from `../../fase-local/resumo/assets/`.

## Numbers

From the armijo reference measurement
([`fekete-reference-armijo-2026-10-17`](../../../../../../benchmarks/results-saved/README.md#fekete-reference-armijo-2026-10-17)):
Fekete et al. with one thread (configuration B, CHR root) against ours with one
thread (C). On the 541 instances both solve: median speedup 72.7× (73×),
arithmetic mean 196×, ours faster on all. Ours solves 557 in 300 s and case 189
in 5363 s; in the paper's configuration Fekete et al. solve 542 in 300 s and 549
in 900 s. The 257 = 1.08 × 238 split is in geometric means, as in the abstract.
The Brazil tour (10,039 km, 0.08 s) is the one of `apps/siicusp34/data/br-estados-demo.js`.

The QR code points to `https://yushipy.github.io/TouringPolygons/apps/siicusp34/en/`,
which only works once the English page is merged and published.
