# Research archive

This directory stores dated evidence from the development of the TPP solvers.
It is intentionally separate from `docs/algorithms/`: research reports record
what was measured and under which conditions, while algorithm documents state
the current contract.

The dates are part of the provenance. A newer report does not automatically
make an older report useless: an older report may contain a baseline,
counterexample, rejected intervention, or reproduction recipe. Before deleting
one, check whether its conclusions and evidence are already represented in a
canonical report.

## Active or canonical evidence

These files are the best current entry points for understanding the maintained
solver and its recent validation:

- [`free-order-oracle-comparison.md`](free-order-oracle-comparison.md):
  controlled comparison of the free-order oracle and whole solver.
- [`free-order-roadmap.md`](free-order-roadmap.md): current follow-up work and
  limitations for the free-order solver.
- [`double-first-certified-convex-tpp-2026-09-16.md`](double-first-certified-convex-tpp-2026-09-16.md):
  certified hybrid convex-oracle implementation and validation.
- [`convex-bnb-comparison-2026-09-15.md`](convex-bnb-comparison-2026-09-15.md):
  comparison of convex-oracle variants inside branch-and-bound.
- [`german-instances-exact-benchmark-2026-09-18.md`](german-instances-exact-benchmark-2026-09-18.md),
  [`german-instances-gurobi13-comparison-2026-09-18.md`](german-instances-gurobi13-comparison-2026-09-18.md),
  and [`german-instances-progressive-benchmark-2026-09-18.md`](german-instances-progressive-benchmark-2026-09-18.md):
  the latest German-instance benchmark series.
- [`unordered-branching-pruning-experiments-2026-09-20.md`](unordered-branching-pruning-experiments-2026-09-20.md)
  and [`unordered-pruning-counterexamples-2026-09-20.md`](unordered-pruning-counterexamples-2026-09-20.md):
  current evidence about search policies and safe geometric pruning.

## Reproducibility and historical evidence

These documents preserve protocols, baselines, and implementation evidence that
may be needed to reproduce or audit the active reports:

- [`unordered-reference-2026-09-05.md`](unordered-reference-2026-09-05.md)
  identifies the historical free-order baseline and hashes.
- [`unordered-comparison-protocol-2026-09-05.md`](unordered-comparison-protocol-2026-09-05.md)
  records the external-solver comparison protocol.
- [`free-order-development-history-2026-09.md`](free-order-development-history-2026-09.md)
  consolidates profiling, contact repair, accepted and rejected changes,
  modularization, validation, and the later ablation series.

## Candidates for a later consolidation pass

- The former dated execution plan and implementation prompt were removed after
  their useful conclusions were incorporated into the current roadmap,
  algorithm documentation, and reports. They were process artifacts rather
  than reproducibility evidence.

This classification is a review aid, not an automatic deletion list. It should
be updated whenever a report becomes the canonical source for an older note.
