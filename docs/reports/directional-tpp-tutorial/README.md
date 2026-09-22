# Fixed-order convex Touring Polygons tutorial

`main.tex` is the complete tutorial, including a one-page visual overview.
`figures.tex` contains all twenty-one
deterministic TikZ vector illustrations; no image downloads or generated raster
assets are needed. The geometric drawings use equal horizontal and vertical
scales. The main text labels any schematic use explicitly.

Build from any directory with a TeX installation containing TikZ, Latin Modern,
`listings`, `titlesec`, `tocloft`, `tcolorbox`, `setspace`, and the other packages declared
in `main.tex`:

```bash
bash docs/reports/directional-tpp-tutorial/build.sh
```

The script writes `directional-tpp-tutorial.pdf` here and keeps auxiliary
TeX files in the repository's `.build/directional-tpp-tutorial/`. It uses `latexmk` when available, otherwise runs
`pdflatex` twice to resolve references. It does not build or test solver code.

The worked call/return trace was checked against a temporary instrumented copy
of the current `intersecting_maps.cpp`; no production solver source was edited.
The document distinguishes exact certificates and local derivations from the
remaining general proof obligation for the implicit directional maps.

The source-line key in the PDF reflects the current working tree after the
disjoint locator's scan removal; it is not pinned to a committed revision.
For validation, see `packages/convex-tpp/cpp/src/core/ordered_path_validation.cpp`,
`packages/convex-tpp/cpp/src/main-directional_tests.cpp`,
`packages/convex-tpp/cpp/src/main-intersection_audit.cpp`, and
`apps/benchmark-dashboard/wasm/test-intersections.mjs`.

The separate `web/` edition is an earlier static port of twelve figures. It
has not yet been synchronized with the expanded LaTeX tutorial; use the PDF
for the current text and diagrams.
