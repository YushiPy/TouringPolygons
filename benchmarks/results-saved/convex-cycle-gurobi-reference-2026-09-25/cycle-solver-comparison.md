# C++ cycle solvers, 2026-09-26

The original Gurobi inputs, raw results and configuration are unchanged. This
follow-up uses the same fixed cyclic order, closed disjoint polygons and
Euclidean objective. The C++ solvers do not use Gurobi. Source hashes, compiler,
options and reproduction commands are in `cycle-solver-validation.json`;
`cycle-solver-validation.txt` contains the captured test output.

| Instance | Exact optimum, rounded upward | Gurobi objective minus exact upper | Exact reporting interval width |
| --- | ---: | ---: | ---: |
| two_oblique_triangles | 7.2111025509279791 | 4.4271939004e-7 | 8.8817841970e-16 |
| three_asymmetric_boxes | 10.225599542551457 | 1.0130560248e-6 | 1.7763568394e-15 |
| four_asymmetric_boxes | 15.329642077489737 | 9.1458285212e-7 | 1.7763568394e-15 |
| five_scattered_quadrilaterals | 17.580030557289923 | 5.9755942061e-7 | 3.5527136788e-15 |

Every rational result passed `tpp_convex_verify_cycle_certificate` with
`Optimal`, using its rational contacts without converting them to binary64.
The reported interval widths are outward rounding of irrational lengths, not
optimization error. Exact objectives are retained as sums of square roots of
rational squared link lengths.

The default double backend matched these objectives and all other tested
rational objectives at the reported upper-bound precision. Across **66**
comparisons (known examples, cyclic/reversal variants, 40 seeded random cases
with 2–5 polygons and input-normalization cases), the maximum reported objective
difference was **0**, and the maximum certified double gap was
**7.1054273576010019e-15**. There is no epsilon in the solver. The test-only
rounding allowance is recorded in the configuration. Every constructed cycle
was checked; outward-rounded double contacts were checked again after their
representable inward adjustment.

## Important double limitation

The default double implementation uses the shared double recurrence and double
outer search, but enables **local rational recovery of a failed anchored
query**. The test run counted **178** such queries across the full 66-case
comparison suite. These are visible in `rational_anchor_recoveries`, not hidden
switches to the rational cycle solver. Recovery can be disabled explicitly.

The strict double audit is deliberately retained in the raw output. On
`two_oblique_triangles` it obtained an exact certificate. On
`three_asymmetric_boxes` and `four_asymmetric_boxes` floating contact
materialization failed at some anchors, leaving `FloatingPointLimit` results;
on `five_scattered_quadrilaterals` it returned `OracleFailure`. Thus **the pure
double constructor is not as reliable as the rational one**, even on these
small inputs. The default recovered double variant supplies the objective
agreement stated above. Neither mode labels a merely feasible or
precision-limited candidate exact.

## Interpretation of Gurobi comparisons

The saved configuration's proposed `1e-7 + 1e-8*|objective|` comparison tolerance
is smaller than the observed numerical Gurobi error for these instances. In
particular, its reported objective bound can exceed the independently
certified optimum. The tests preserve the original configuration and compare
both objectives and independently certified intervals for the contracted
Gurobi contacts. Those intervals overlap every exact C++ optimum.

A separate **test-only** `2e-6` objective-regression threshold captures the
observed Gurobi discrepancies; it is not an optimization or feasibility
threshold in either C++ solver. The exact claim rests exclusively on the
rational support certificate. No equality of Gurobi contact coordinates is
required. The displayed milliseconds include extra validation and double
comparisons and are not isolated solver benchmarks.

## Validation

- `main-cycle_tests`: passed; exact rational and default-double comparisons,
  strict-double diagnostics, rational translations beyond binary64 coordinate
  resolution, and rational scales `2^-1100` and `2^1100`.
- `main-cycle_certificate_tests`: passed, including the pre-existing reference
  candidates and invalid-input checks.
- `main-directional_tests --random-boxes 50 --random-convex 50 --corpus
  packages/convex-tpp/cpp/tests`: **1604 checks, zero failures, zero unresolved**;
  no hybrid shadow mismatch.
- `main-intersection_tests`: passed, guarding the existing fixed-source APIs.

The broader repository checks were also attempted:

- `./scripts/sanity_check.sh --no-install`: built the generated-test executable;
  interrupted during the large stress-suite generation to keep this change's
  validation bounded. The full sanity run is **not** claimed passing.
- `RUN_BROWSER=0 npm run test:all`: blocked by unavailable Python dependencies;
  downloading `six==1.17.0` from PyPI failed with DNS/network errors.
- `node wasm/test-intersections.mjs`: blocked because the worktree has no
  generated `static/wasm/tpp_convex_wasm.js` module.

No dependency installation or generated WASM was added to the changes.
