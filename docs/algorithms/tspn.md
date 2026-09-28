# TSPN with the maintained insertion/decomposition branch and bound

`tpp_nonconvex_tspn_solve(polygons, options)` in `tpp/nonconvex/unordered.h`
minimizes a closed Euclidean tour visiting all supplied simple polygons, with
free cyclic order and no prescribed geometric point. Regions may be nonconvex,
overlap, touch or contain each other. Polygon holes are not represented by this
API. The returned `path` repeats its first point at the end. With zero or one
region the optimum is zero.

This is a topology option inside the existing `unordered.cpp` search, not a
second B&B. The endpoint API retains its behavior. The queue, diving, lazy
convex decomposition, missing-region selection, incumbent checks, interrupted
frontier accounting, traces and diagnostics are shared. See
[the endpoint algorithm](unordered-tpp.md) for their contracts.

## Cyclic relaxation and branching

A node stores a partial cyclic sequence. Each entry represents either an
original region's convex hull or one convex decomposition piece. Its relaxation
uses `tpp_convex_solve_cycle_double`, including the exact independent certificate
and counted rational recoveries. If the contact-derived lower bound remains
too weak, the rational cycle solve strengthens it. The retained double path is
independently feasible for the convex regions. No optimization epsilon is added
to the convex-cycle constructor or its certificate.

The root represents region 0's hull and has bound zero. A single-region relaxed
cycle may be any point in that hull. A vertex seeds the initial feasible
heuristic; it is never fixed in subsequent relaxations. Rooting at a *region*
removes cyclic rotation without imposing an artificial depot.

If a missing original region is absent from a sequence of size m, insert its
hull into every cyclic gap, including the closing edge: m positions rather than
the m+1 positions of an endpoint path. Region 0 remains first. At m=2 the two
children are reversals of the same triangle; only one is needed. Subsequent
insertions retain every cyclic gap. If the missing region is already represented
by its hull, branch over its existing convex decomposition pieces.

Every feasible closed tour admits a chosen visit to region 0 as cyclic origin.
The visits to represented regions occur in a cyclic order; any further chosen
visit belongs to one of its gaps. Reversing the full tour preserves its length
and visits, so the size-two symmetry reduction loses no objective value. A
visit to a nonconvex region belongs to some decomposition piece. These facts
prove coverage of the branching alternatives. Omitting regions and replacing
them with convex hulls relaxes the constraints, so the cycle optimum is a valid
node lower bound. A relaxed tour that covers all original regions is a feasible
global incumbent.

As in the endpoint search, crossings of a partial tour and repeated visits to
one region do not authorize pruning. The existing counterexamples and their
reasoning still apply.

## Cyclic insertion bounds

For arbitrary vectors `|u_i| <= 1`, the cycle dual bound is

```
D(u) = sum_i min_(v in C_i) (u_(i-1)-u_i) dot v.
```

The inequality `u_i dot (q_(i+1)-q_i) <= |q_(i+1)-q_i|` telescopes around the
cycle. Support minima are taken over polygon vertices. This formula has no
fixed-endpoint term. Translating every vertex by the same origin leaves it
unchanged.

Splitting edge i to insert a region changes only the new region's support and
the two neighboring supports. These terms are updated from the parent dual.
The one-region case combines both neighbor changes into its single support
term. Unit-ball directions and all support arithmetic are rational; a floating
norm is rounded upward and checked by rational squaring before division. The
reported result is rounded downward and bounded below by zero. This remains
valid even when reference contacts are infeasible or coincident. The endpoint
insertion bound is unchanged.

## Scope of exactness and limits

The convex rational oracle is exact on its input regions, with objective stored
as a sum of radicals. The complete TSPN B&B retains the existing numerical
geometry/normalization, visit tolerance and user-selected global gap contract.
Its `exact` field means that this requested gap was closed; it does **not** mean
a zero-error rational solution of the complete nonconvex TSPN. Bounds and
`termination` are returned for time, call and numerical limits. An oracle failure
does not authorize pruning or a claim of optimality. The time limit is
cooperative; one already-started cycle solve or decomposition may exceed it.

With n regions, orders contribute at most `(n-1)!/2` unoriented cyclic orders
for n>=3, and decomposition choices multiply the worst-case search. Each branch
contains at most one insertion and one refinement per region. The search remains
exponential; the convex oracle's operation/bit complexity is documented in
[convex-cycle.md](convex-cycle.md). Cyclic dual screening reuses all unchanged
supports and avoids solving children already excluded by their lower bound.

## Use and validation

```cpp
tpp::UnorderedTppSolveOptions options;
options.max_seconds = 3;
auto result = tpp::tpp_nonconvex_tspn_solve(polygons, options);
```

The existing CLI accepts `tpp-unordered --cycle`; its text input format stays
unchanged and the two input endpoints are ignored. A supplied `--initial-path`
must be closed and feasible, and can start anywhere. JSON includes `mode` to
prevent comparing different formulations accidentally.

`tpp-tspn-tests` compares the B&B with exhaustive cyclic order/piece enumeration,
checks missing-depot behavior, zero tours, overlaps and nonconvex decomposition,
and checks lower/upper bounds under call caps. Its enumeration intentionally
shares the convex oracle; the external comparison is a separate correctness
check. `tpp-unordered-tests` protects the old endpoint contract.

`python3 benchmarks/tpp.py tspn-benchmark --output NEW_DIRECTORY` compares the
whole native solve with the pinned Fekete B&B's original SOCP backend. It builds
the external source list unchanged outside the submodule. Source revisions,
hashes, settings, inputs, raw tours, intervals, timings and analysis are saved
together. Independent tour validation uses binary-rational intersection and
containment tests plus numerical distance/length diagnostics. Raw feasibility
and reported gap closure are kept separate; coordinate equality is not required.
