# Fixed-order non-convex TPP

`tpp_nonconvex_solve(start, target, polygons, options)` in
`tpp/nonconvex/solver.h` returns a shortest path from `start` to `target` that
visits the simple polygons in the given order. The returned `path` includes
`start` and `target`.

## Method

- If every polygon is convex, one call of the convex solver
  (`tpp_convex_solve_binary_search_lazy`) answers it.
- Otherwise each non-convex polygon is split by the optimal convex partition
  (`packages/optimal-convex-partition`). A breadth-first branch-and-bound then
  chooses one piece per polygon, in the visit order:
  - a node fixes the pieces of a prefix of the polygons;
  - its bound is the convex TPP through those pieces and the convex hulls of
    the remaining polygons;
  - a child is kept only when its bound does not exceed the incumbent;
  - a leaf is solved as a convex TPP and may replace the incumbent;
  - the first incumbent uses the first piece of every polygon.
- `exact` means the queue emptied within `max_calls` and `max_seconds`.
  Otherwise the returned path is the incumbent, which is feasible but not
  certified.

## Contract of the bound

The bound is valid only if every hull contains all the pieces of its polygon,
and if the convex solver returns the optimum on that input. The convex solvers
take convex counterclockwise polygons with **distinct consecutive vertices**.
A repeated vertex is an edge of length zero. Given one, the solvers can choose a
non-optimal contact and return a path longer than the optimum. In a reduced
case from the route P-n55-k15 below (one polygon, the hull of client 27 with
its rightmost vertex repeated, between a vertex of the previous client and the
depot), all seven variants in `tpp/convex/solver.h` returned 22.7752 instead of
22.0553. As a
bound, that value can prune the branch that holds the optimum. The same
precondition applies to the polygons given to `tpp_nonconvex_solve`, because
convex input goes straight to the convex solver.

## Repeated hull vertex (fixed on 2026-10-29)

**Mechanism.**
- The hulls are computed by a monotone chain over the vertices of all the
  pieces. Pieces share the endpoints of their diagonals, so those points
  appear twice.
- The turn test `(b - a) × (c - a) > 0` is evaluated in binary64. With fused
  multiply-add contraction, the cross product of a vector with itself is the
  rounding error of one product, not zero.
- When that error is positive and the repeated point is the first or last in
  the sort, the hull keeps it twice.

**Scope.**
- The defect needs FMA contraction: native builds with clang on arm64 (the
  Mac), where `-ffp-contract=on` is the default. GCC on x86-64 with the default
  target (CMake here sets no `-march`) has no FMA instruction, and WebAssembly
  has none either.
- Three copies of the hull carried it:
  - `solvers/bnb.cpp`, used by `tpp_nonconvex_solve` and the dashboard;
  - `main-bnb_workload_benchmark.cpp`, used by the non-convex campaigns of
    `benchmarks/tpp.py`, whose group hulls merge pieces;
  - `main-tpp_bnb.cpp`, whose hulls are of whole polygons and so were
    affected only by input with a repeated vertex.
- The unordered and TSPN solvers build their hulls with exact turn signs and
  remove repeated points first, so they were not affected.
- No summary in `benchmarks/results-saved/` comes from these binaries.

**Observed.** Every route of the CVRPN solutions of Birgin and Matos
(rsmatos11/cvrpn_results) was solved in its fixed order on the Mac. 2 of the
1,343 routes were marked exact with a path longer than their feasible route:

| Route | Returned | Optimum |
|---|---:|---:|
| P-n55-k15, route 2 | 62.4025 | 61.7306 |
| tai385, route 13 | 613.88 | 603.39 |

**Fix.** All three copies sort the points and remove exact repeats before the
scan.

**Regression test.** `tpp-nonconvex-tests` is part of
`scripts/sanity_check.sh`. It checks the two routes above, and 400 random
instances of four star-shaped polygons, against brute force over all piece
choices. Without the fix, the routes fail on the Mac. The random instances
pass with or without it, because the defect needs a shared vertex at an
extreme of the sort; they are there as general validation.
