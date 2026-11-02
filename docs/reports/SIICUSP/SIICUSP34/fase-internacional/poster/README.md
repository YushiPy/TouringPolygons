# Poster A0 — 34th SIICUSP, International Stage

English poster for the poster session of 25–26/11/2026 (edital, item 4.5:
material in English). The title is the one of the international abstract,
[`../resumo/resumo-en.tex`](../resumo/resumo-en.tex), which is also the
reference for numbers and wording. The first-stage poster stays unchanged in
[`../../fase-local/poster/`](../../fase-local/poster/).

## Build

Requires XeLaTeX, Arial, Latin Modern Math and the packages of the preamble.
The speedup figure uses the abstract's generator with a higher-contrast print
palette and a separate output in `figures/`; the abstract's PDF is unchanged. Build it and the other
figures with `render_print_figures.mjs` and `build_tspn_cards.py` (below); all PDFs here are ignored too (see
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
  its geometry and tour. A TSPN tour has no first visit, so the colours
  follow the visiting order from the Federal District, lighter first.
- `figures/method-0N.pdf`: the six method cards, as closed tours (TSPN) on the
  challenge drawings, from
  [`apps/siicusp34/scripts/build_tspn_cards.py`](../../../../../../apps/siicusp34/scripts/build_tspn_cards.py)
  with `--pdf-dir` pointing here; the same script draws the page's SVG cards.

Cards 01, 02, 04 and 06 use the same combined-challenge instance, positions and
letters. Card 01 reads the actual initial incumbent from the native `--trace`
output and checks its length against `initial_length`; it does not rebuild the
heuristic in Python. Card 02 explicitly draws the selected convex hulls (the
tour need not visit the original A or B). Card 06 distinguishes the two pieces
of B and shows the corresponding two children. Every tour is checked against
the targets of its own subproblem. Labels stay clear of contacts and cuts.

From the repository root, with matplotlib and shapely available:

```sh
python3 apps/siicusp34/scripts/build_tspn_cards.py \
  --pdf-dir docs/reports/SIICUSP/SIICUSP34/fase-internacional/poster/figures
python3 docs/reports/SIICUSP/SIICUSP34/fase-internacional/resumo/generate_speedup_figure.py \
  --print --output docs/reports/SIICUSP/SIICUSP34/fase-internacional/poster/figures/speedup-figure.pdf
```

The map comes from [`render_print_figures.mjs`](render_print_figures.mjs), which
prints SVG to PDF with Playwright (or the installed Google Chrome) and trims
the margins with `pdfcrop`. From the repository root, with the dashboard's
Playwright installed:

```sh
node docs/reports/SIICUSP/SIICUSP34/fase-internacional/poster/render_print_figures.mjs
```

The speedup figure has the same data and statistics as Figure 2 of the abstract; the logos
come from the first stage (`../../fase-local/poster/figures/` and
`../../fase-local/resumo/assets/`).

## Numbers

From the armijo F rerun
([`fekete-reference-armijo-2026-10-30`](../../../../../../benchmarks/results-saved/fekete-reference-armijo-2026-10-30/README.md)):
Fekete et al. with one thread (configuration B, CHR root) against ours with one
thread (F), both using U−L ≤ 0.001U + 0.001. On the 541 instances both solve:
median speedup 69.0107× (69×), arithmetic mean 183.2106× (183×), ours faster
on all. Ours solves 557 in 300 s and in 900 s; case 189 remains open in F
at 900 s, and its longer run has no result in the local copy. The 89 min
from C is not used for F. With 16 threads Fekete et al. solve
542 in 300 s and 549 in 900 s. The 245 ≈ 1.08 × 226 split uses geometric
means on the 166 instances both solvers close in all three 300 s runs and
where Fekete's fastest run takes at least 1 s; each solver uses its fastest
repetition for this decomposition. This differs from the per-instance median
times used for the 541-instance speedup figure.

Accuracy uses B/CHR versus F, one thread each: for each of all 558 instances,
the largest polygon miss over its three 300 s runs, including unclosed searches.
The CHR median is 2.648977721572538e-6 and maximum 8.229625996190154e-4,
printed as 2.65e-6 and 8.23e-4, within the 1e-3 SOCP tolerance. The old
9.6e-4 maximum belonged to LEFP. Ours stays within 3e-11. The 16-thread
measurement adapts the original script's 24 threads to the armijo machine.
The Brazil tour is a TSPN (no fixed point): 9,962 km, 0.016 s (median of three runs, one thread, Mac), from
`apps/siicusp34/en/data/br-estados-tspn.js` (`apps/siicusp34/scripts/build_br_estados_tspn.py`).
The Portuguese page's tour, which starts and ends in Brasília, is 10,039 km.

The QR code points to `https://yushipy.github.io/TouringPolygons/apps/siicusp34/en/`,
which only works once the English page is merged and published.
