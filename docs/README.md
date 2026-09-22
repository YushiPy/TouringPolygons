# Documentation

Start with [`architecture.md`](architecture.md) for the maintained code map,
application lifecycle, generated-file policy, and repository decisions.

- `algorithms/`: current algorithm specifications, correctness audits, and
  implementation roadmaps.
- [`research/`](research/README.md): dated benchmark analyses, development
  plans, comparison notes, and historical prompts. These documents preserve
  evidence but are not API specifications.
- `bibliography/`: source papers and TeX conversions used for reading and
  citation.
- `reports/`: Portuguese LaTeX reports, figures, and repository change reports.
- `prompts/`: reusable prompts for independent technical reviews.
- `third-party.md`: policy for local external solver and instance checkouts.

The current cleanup is recorded in
[`reports/repository-refactor-2026-09.md`](reports/repository-refactor-2026-09.md),
and the independent-review prompt is
[`prompts/review-repository-refactor.md`](prompts/review-repository-refactor.md).

Generated PDFs, LaTeX intermediates, browser builds, and scratch data stay
ignored. Report-specific build artifacts remain local to each report directory.
