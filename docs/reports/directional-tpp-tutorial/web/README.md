# Static web edition

This directory contains the local, unpublished Phase 1 edition of the fixed-order convex Touring Polygons tutorial.

## Use locally

From this directory:

```bash
./build.sh
python3 -m http.server 8765
```

Open <http://127.0.0.1:8765/>. The generated `index.html` uses MathML and the checked-in SVG assets only; it does not call a server API or require network access after the build. `pandoc` is used locally at build time to transfer the report text and equations from `../main.tex`.

The twelve SVGs in `assets/figures/` are rendered from the original TikZ definitions in `../figures.tex`. They are static Phase 1 figures. Each figure has a disabled `Expand · Phase 2` control location; there is no geometry engine, playback, source overlay, or other interactive behavior in this pass.

The displayed report is authoritative only for the claims in `../main.tex`. The page preserves the report's distinction between exact witnesses, schematic drawings, observed implementation behavior, historical test results, and the unproved general equivalence of the implicit directional maps. Source references retain their revision/line identifiers for a later source explorer; they do not open code in Phase 1.
