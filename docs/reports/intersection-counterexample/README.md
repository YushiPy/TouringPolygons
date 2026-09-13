# Pseudo-vertex counterexample report

`report.tex` is a self-contained LaTeX report on the failure of the literal
two-single-reflection pseudo-vertex rule in Tan and Jiang, TAMC 2017, Section 4.
It contains four original TikZ figures, exact optimality and uniqueness proofs
for both target witnesses, a derivation of the missing one-sided limiting ray,
the exterior-side bending cone, source locators, and the scope of the conclusion.

Build from the repository root:

```bash
bash docs/reports/intersection-counterexample/build.sh
```

Requirements: `latexmk`, `pdflatex`, TikZ, `tcolorbox`, and the New PX fonts,
as supplied by a full TeX Live/MacTeX installation. No external image files,
BibTeX pass, or network access are needed. On macOS, the script also checks
the standard `/Library/TeX/texbin` location.

The generated PDF is `output/pdf/intersection-counterexample.pdf`; intermediate
LaTeX files stay in `.build/intersection-report/`. Both directories are ignored
by Git, so the source and build script are the reproducible deliverables.

The existing certificate harness was rerun during report preparation:

```text
.build/intersection-audit/tpp-convex --proof-only
Checks=99, failures=0 (proof_only=1)
```

This result checks the independent proof fixtures and auxiliary invariants.
It does not certify the production intersection solver. See the
[fixture instructions](../../../benchmarks/suites/intersection-audit/README.md)
and [full audit](../../algorithms/intersecting-tpp-audit.md) for that separate work.

The counterexample refutes the stated local recipe and its region-adjacency
claim. It does not prove that a corrected polynomial algorithm or the claimed
complexity bounds are impossible. No production solver was changed for this report.
