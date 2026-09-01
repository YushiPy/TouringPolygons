# Convex TPP With Intersections: Tan-Jiang Implementation Plan

Context: the `Problem Instances` manual campaign exposed several incorrect solutions for convex polygons that intersect. The dashboard visualizer uses the WASM solver from `apps/visualizer-server/static/wasm`, so native C++ changes must be rebuilt into WASM and cache-busted in the dashboard before browser behavior reflects solver changes.

## Current State

- The current intersecting-convex solver is **not** a faithful Tan & Jiang implementation.
- It is currently a hybrid of:
	- Dror-style last-step map machinery for the disjoint case;
	- ad hoc intersecting-case handling in `SolutionLinearSearchIntersecting`;
	- cheap candidate checks for direct paths, single-contact paths, and shared-contact paths in overlapping blocks;
	- fallback to the existing map-style solver.
- The coordinate-descent fallback that was briefly introduced was removed from the hot path because it caused severe WASM slowdowns, including hundreds of milliseconds on small instances.
- The dashboard now displays the solved path length in the status line, e.g. `Solution: exact, length 0.631492, 1 calls, 100.00 us via WASM`.

## Observed Bugs / Symptoms

- Small perturbations of an intersecting polygon can still cause abrupt jumps in the returned path length.
- In the current `Problem Instances` case, moving polygon 2 very slightly can save about `0.01` length, which strongly suggests the solver is choosing the wrong combinatorial contact structure.
- This discontinuity is a correctness issue, not just a rendering issue.
- The correct model should allow a path segment to visit later polygons automatically by crossing them in order, without forcing artificial bend/contact points at polygon intersections.

## What Gabriel Wants

Implement the **real Tan & Jiang algorithm** for convex TPP with possibly intersecting polygons, but use Dror's binary-search point-location/query approach instead of Tan & Jiang's all-at-once linear-time point-location traversal, because the latter is slower in practice in this codebase.

In other words:

- Keep Dror's binary-search query style where it is useful.
- Replace the intersecting-convex branch with Tan & Jiang's actual structural extension.
- Do not add more heuristics or hardcoded candidate patches for individual examples.

## Required Algorithmic Direction

For intersecting convex polygons:

1. Compute all boundary intersections between polygons.
2. Treat boundary intersection points as pseudo-vertices.
3. Split polygon edges into pseudo-edges.
4. Build last-step shortest path maps over the augmented boundary structure.
5. Correctly distinguish:
	- reflection contacts,
	- bending contacts at original vertices,
	- bending contacts at valid pseudo-vertices,
	- crossing/pass-through contacts.
6. Respect Tan & Jiang's condition for pseudo-vertex bending regions:
	- a pseudo-vertex `j` of `P_{i+1}` and `P_h` is relevant only when `j` lies in the interiors of `P_{h+1}, ..., P_i`;
	- and the shortest partial path to `j` does not intersect the interior of `P_h` nor `P_{i+1}`.
7. Query the resulting map using Dror-style binary point location rather than Tan & Jiang's simultaneous monotone scans.

## Files To Study First

- `docs/bibliography/TPP-Tan&Jiang/main.tex`
	- Especially Section `sec:intersecting`.
- `docs/bibliography/TPP-Dror/main.tex`
	- Especially the last-step shortest path map query logic.
- `packages/convex-tpp/cpp/src/core/solution.cpp`
- `packages/convex-tpp/cpp/src/solvers/binary_search.cpp`
- `packages/convex-tpp/cpp/src/solvers/linear_search.cpp`
- `packages/convex-tpp/cpp/src/solvers/tan_jiang.cpp`
- `packages/convex-tpp/cpp/src/core/tests.cpp`
- `packages/convex-tpp/cpp/src/main-intersection_tests.cpp`
- `apps/benchmark-dashboard/static/editor-solver.js`
- `apps/visualizer-server/wasm/build.sh`

## Tests / Verification Needed

- Add regression tests for the current `benchmarks/campaigns/Problem Instances/manual-cases.json` instance.
- Add perturbation/continuity-style tests:
	- move one vertex of polygon 2 by small epsilons;
	- assert the path length changes smoothly, not by a large discrete jump;
	- assert the path still touches every polygon in order.
- Keep previous regressions for:
	- intersecting triangles requiring a shared/intersection contact;
	- polygon 2 being touched for free by the outgoing segment from polygon 1;
	- three or more polygons where intermediate polygons are crossed rather than explicitly contacted.
- Rebuild native targets:
	- `cmake --preset convex-release -DTARGET=main-intersection_tests -DTPP_ENABLE_GUROBI=OFF`
	- `cmake --build --preset convex-release`
	- `build/convex-release/packages/convex-tpp/cpp/tpp-convex`
- Rebuild dashboard/visualizer solver artifacts:
	- `cmake --preset nonconvex-release -DTARGET=main-visualizer_solve -DTPP_ENABLE_GUROBI=OFF`
	- `cmake --build --preset nonconvex-release`
	- `apps/visualizer-server/wasm/build.sh`
- Verify the browser path, not only the CLI:
	- dashboard imports `/visualizer-static/wasm/tpp_convex_wasm.js`;
	- cache keys in dashboard JS must be bumped when the WASM or solver JS changes.

## Performance Expectations

- Two triangles should solve in microseconds in WASM.
- Small two-convex-polygon instances should usually be microsecond scale.
- Millisecond-scale behavior on tiny cases is suspicious and should be investigated before accepting a fix.
- Avoid numeric coordinate descent or iterative optimization in the hot path; it is not Tan & Jiang and previously caused severe slowdowns.

## Important Caveat

Do not treat the current hybrid solver as the final design. It fixed some visible examples, but the discontinuity under tiny perturbations is strong evidence that the intersecting convex branch is still structurally incomplete.
