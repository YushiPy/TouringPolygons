# Intersecting Convex TPP: implementation roadmap

Status: open research work. The disjoint fixed-order solver is maintained; the
intersecting extension is not yet a faithful production implementation.

This document is the implementation specification and verification checklist
for the next algorithmic phase. It is intentionally separate from the audit in
`intersecting-tpp-audit.md`, which records the current mathematical obstruction.

Implement, diagnose, test, and optimize the fixed-order Touring Polygons Problem solver for a sequence of convex polygons that may intersect.

This is an implementation task, not merely a review. Continue until you have either:

1. produced and thoroughly verified a faithful implementation of the published intersecting-polygon construction while preserving the existing solver architecture; or
2. identified a precise mathematical or specification-level obstacle, supported by exact references to the papers and concrete counterexamples.

### Non-negotiable architecture

Preserve the existing Dror-based **binary-search + memoization** algorithm as the core solver.

For pairwise-disjoint convex polygons, the existing implementation is mature, correct, and empirically fast for this project's practical instances, where polygons commonly have at most about 20 vertices. Do **not** replace it with Tan and Jiang's `O(kn)` disjoint-polygons algorithm merely because that algorithm has a better asymptotic bound.

The intended result is specifically a hybrid:

- retain the existing binary-search and memoized Dror-based last-step-map implementation;
- use Tan and Jiang's treatment of polygon intersections to extend that implementation to possibly intersecting convex polygons;
- incorporate pseudo-vertices, pseudo-edges, crossing regions, and all necessary changes to first-contact and query semantics;
- continue using the existing cached binary-search machinery wherever its invariants remain valid;
- preserve the established disjoint fast path and its behavior.

Do not route intersecting instances through the linear-search solver as the final design. The binary-search + memoization implementation must itself correctly support intersecting polygons.

Do not implement Tan and Jiang's disjoint algorithm as a replacement for the existing Dror implementation.

First determine exactly which invariants used by the Dror-based binary-search implementation fail when polygons intersect. Then adapt those invariants using the mathematical construction in Tan and Jiang's intersecting section.

If some part of Tan and Jiang's intersection extension cannot be combined with the Dror-based maps, identify the incompatibility precisely. Do not use that possibility as permission to replace the entire solver prematurely. Preserve as much of the binary-search + memoization architecture as is mathematically valid.

### Mathematical problem

The input consists of:

- a start point;
- a target point;
- a fixed ordered sequence of closed convex polygons;
- polygons that may overlap, contain one another, share boundary portions, or touch.

The output must be a globally shortest Euclidean path from the start to the target that visits every polygon in the prescribed order.

This is the **unconstrained** problem. There are no fences, obstacles, or restrictions on where segments between visits may travel. Do not introduce machinery that exists only for the fenced variant.

A path may satisfy several consecutive polygon visits at the same point or along the same segment when their intersections and the prescribed order permit it. Validity concerns the complete continuous polyline, not only its explicitly stored vertices.

### Primary references

Read these sources before changing the implementation.

Tan and Jiang:

- `docs/bibliography/TPP-Tan&Jiang/original.pdf`
- `docs/bibliography/TPP-Tan&Jiang/main.pdf`
- `docs/bibliography/TPP-Tan&Jiang/main.tex`

The key section is **“Touring a Sequence of Possibly Intersecting Polygons”**, beginning around line 226 of `main.tex`. Also read the preceding disjoint section because the intersecting section explicitly extends the data structure developed there.

Pay particular attention to:

- boundary intersections treated as pseudo-vertices;
- polygon edges split into pseudo-edges;
- interior/crossing regions of the maps;
- shortest partial touring paths to original vertices and pseudo-vertices;
- merging adjacent crossing regions;
- bending and reflection regions involving pseudo-vertices;
- construction and ordering of last-step shortest-path maps;
- the claimed `O(k^2 n)` construction complexity;
- the claimed `O(kn)` path-reporting complexity.

Dror et al.:

- `docs/bibliography/TPP-Dror/original.pdf`
- `docs/bibliography/TPP-Dror/main.pdf`
- `docs/bibliography/TPP-Dror/main.tex`

Use Dror et al. to understand the existing last-step-map implementation and its query recursion. Their general algorithm discusses fences, but fences are explicitly out of scope. Do not introduce fenced-problem machinery unless a component is independently required for the unconstrained algorithm and that requirement is proven.

### Relevant implementation

Primary solver files:

- `packages/convex-tpp/cpp/src/solvers/linear_search.cpp`
- `packages/convex-tpp/cpp/src/solvers/binary_search.cpp`
- `packages/convex-tpp/cpp/src/solvers/tan_jiang.cpp`

Supporting interfaces and geometry:

- `packages/convex-tpp/cpp/include/tpp_convex.h`
- `packages/convex-tpp/cpp/include/tpp_convex_common.h`
- `packages/convex-tpp/cpp/include/tpp/convex/solver.h`
- `packages/convex-tpp/cpp/include/tpp/convex/workspace.h`
- `packages/common-geometry/cpp/include/tpp/geometry/`
- `packages/convex-tpp/cpp/src/core/`

Relevant test entry points:

- `packages/convex-tpp/cpp/src/main-intersection_tests.cpp`
- `packages/convex-tpp/cpp/src/main-verify_solutions.cpp`
- `packages/convex-tpp/cpp/src/core/tests.cpp`
- `packages/convex-tpp/cpp/tests/`

The current intersecting implementation came from an earlier attempt and must not be assumed correct. Inspect critically:

- `SolutionLinearSearchIntersecting`;
- `SolutionBinarySearchIntersecting`;
- `intersects_previous_polygon`;
- `segment_enters_polygon_before_endpoint`;
- `solve_intersecting_convex_sequence`;
- `optimize_intersecting_convex_sequence`;
- polygon-intersection shortcut candidates;
- path validation and fallback behavior;
- the binary solver's delegation to the linear intersecting solver.

Some of this code appears to be an improvised repair or optimization layer rather than a faithful implementation of Tan and Jiang's intersection construction. It may be deleted or replaced when justified. Do not preserve it merely because it already exists.

Inspect `git status` and relevant diffs before editing. The working tree may contain unrelated user changes. Preserve them; do not revert, overwrite, or broadly reformat unrelated work.

### Known incorrect case

The principal regression is:

- `benchmarks/campaigns/Cool Instances/manual-cases.json`
- case name: `Wrong1`

The current result visibly fails to visit at least one required polygon. Reproduce this failure before modifying the solver and convert it into a permanent automated regression test.

A returned path is valid only if the continuous polyline intersects every closed polygon in the required order. A validator must therefore reason about segment-polygon contact and the ordering of contacts along segments, rather than merely checking whether stored path vertices lie in polygons.

Passing `Wrong1` is necessary but not sufficient.

### Required implementation process

#### 1. Establish the specification and invariants

Before editing, document concisely:

- what each last-step map represents;
- what a map query returns;
- the definitions of first and last contact;
- how pass-through/crossing, reflection, and bending regions are represented;
- how overlapping polygons alter those definitions;
- the role of pseudo-vertices and pseudo-edges;
- how several visits can occur at one point or along one segment;
- which disjoint binary-search invariants remain valid;
- which invariants fail when polygons intersect.

Tie each important rule to a section, lemma, figure, or page in Tan and Jiang or Dror et al.

#### 2. Audit the current implementation against the papers

For every piece of intersecting-specific logic, classify it as:

- directly prescribed by Tan and Jiang;
- a valid implementation detail equivalent to the paper;
- an unsupported heuristic;
- or incorrect.

Determine the exact cause of `Wrong1`. Do not attribute it to numerical instability without concrete evidence.

#### 3. Implement the intersecting construction

Implement the intersection extension faithfully, including all required pseudo-vertex, pseudo-edge, crossing, bending, and reflection behavior, while retaining the Dror-based binary-search + memoization core.

Prefer representations that map clearly to the terminology and cases in the papers. Add concise comments with paper references where the correspondence is not obvious.

Do not use any of the following as the production solution:

- iterative contact-point optimization;
- generic convex optimization;
- Gurobi;
- sampling;
- arbitrary candidate enumeration;
- fallback to an unrelated solver;
- post-hoc path repair as a substitute for constructing the correct map;
- validation that silently accepts or returns a merely plausible route.

Assertions and independent validators are encouraged, but validation must expose algorithmic bugs rather than define the algorithm.

#### 4. Preserve the disjoint solver

Run the existing disjoint test suite before and after the change.

Pairwise-disjoint inputs must continue to use the established binary-search + memoization fast path. Preserve its correctness, public behavior, and practical performance. Any unavoidable change must be rigorously justified and measured.

#### 5. Add strong correctness tests

At minimum, test:

- `Wrong1`;
- pairwise-disjoint polygons;
- two polygons crossing at two boundary points;
- one polygon contained in another;
- polygons touching at one vertex;
- polygons sharing or overlapping a collinear boundary portion;
- three or more polygons sharing a feasible common point;
- consecutive visits that can be satisfied at the same point;
- intersections with nonconsecutive earlier polygons;
- a start or target inside a polygon;
- a start or target on a polygon boundary;
- routes that pass through a polygon interior;
- routes that touch only a boundary vertex;
- clockwise and counterclockwise polygon input;
- near-degenerate but valid convex inputs;
- repeated geometric contacts corresponding to different visit indices.

For every returned path, independently validate ordered visitation along the complete continuous polyline.

Where practical, compare lengths with an independent exact or high-confidence oracle on small instances. The oracle may be used for tests only; it must not become production fallback logic.

Use randomized small convex instances and metamorphic tests, including:

- translation;
- rotation;
- uniform scaling;
- reversing polygon vertex orientation;
- inserting redundant collinear vertices when accepted by the input contract.

#### 6. Test continuity across intersection boundaries

Construct parameterized families of otherwise identical convex instances in which two relevant polygons are:

- separated by a very small distance when `delta > 0`;
- exactly touching when `delta = 0`;
- overlapping by a very small amount when `delta < 0`.

Evaluate sequences of `delta` values approaching zero from both sides. Include transitions involving:

- vertex-to-edge contact;
- edge-to-edge contact;
- nearly collinear edges;
- containment beginning at tangency;
- intersections between consecutive polygons;
- intersections with nonconsecutive earlier polygons;
- a common intersection of several polygons appearing or disappearing under a tiny perturbation.

The primary continuity check is the **optimal path length**: small coordinate perturbations must produce correspondingly small changes in the optimum value. Compare with an independent oracle where practical.

Also compare returned geometric paths using a representation-independent distance, such as a segment-based Hausdorff distance after removing redundant collinear points. Do not require identical contact vertices or combinatorial representations: at degenerate inputs, multiple globally optimal paths may exist and a deterministic solver may switch between different optimal representatives even while the optimum value remains continuous.

For every parameter value, independently verify:

- ordered visitation by the continuous polyline;
- consistency between reported length and returned geometry;
- optimality within a justified tolerance;
- absence of an unjustified length jump at the disjoint/touching/intersecting transition;
- convergence of the disjoint fast path and intersecting implementation to compatible answers near the transition.

Scale tolerances relative to coordinate magnitude and perturbation size. Do not loosen tolerances to conceal a failure, and report the chosen tolerance policy.

#### 7. Check complexity and performance

Only after correctness is established:

- identify the construction and query complexity of the hybrid implementation;
- compare it with the bounds and operations described in the papers;
- ensure caches and workspaces are used effectively;
- avoid repeated all-pairs geometry inside query recursion;
- avoid unnecessary allocation in hot paths;
- benchmark disjoint and intersecting inputs separately;
- compare against the current implementation using reproducible cases.

The practical regime usually has at most about 20 vertices per polygon, so constant factors matter. Correctness and fidelity still come first.

### Verification standard

Do not declare success merely because the project compiles or `Wrong1` looks visually correct.

Success requires:

- all relevant existing tests pass;
- `Wrong1` becomes valid and is covered by an automated regression test;
- new intersection and degeneracy tests pass;
- continuous-path ordered validation passes;
- randomized and oracle comparisons reveal no unexplained mismatch within justified tolerances;
- continuity tests pass across disjoint, touching, and overlapping transitions;
- sanitizer runs are clean where available;
- disjoint behavior remains correct and fast;
- the implementation can be mapped case by case to the published intersection construction;
- performance is measured rather than guessed;
- intersecting inputs are solved by the binary-search + memoization implementation, not by delegating to the linear solver.

Use the project's existing build and test system. Do not weaken tolerances or validators to make tests pass.

### Final report

At completion, report:

1. the precise root cause of `Wrong1`;
2. the mathematical invariants that failed in the previous intersecting implementation;
3. how the implementation maps to Tan and Jiang's intersection construction;
4. how the Dror-based binary-search + memoization core was preserved;
5. unsupported heuristics or fallback paths removed or bypassed;
6. files changed;
7. tests, sanitizers, oracle checks, continuity checks, and benchmarks run, with results;
8. achieved or expected asymptotic complexity;
9. remaining mathematical, numerical, or performance limitations;
10. a clear statement of whether the result is faithful to the cited algorithms.

If a faithful hybrid cannot be completed, do not replace it with an undocumented workaround. Instead report:

- the precise incompatibility or missing result;
- exact section, page, lemma, or figure references;
- a minimal counterexample or failing construction;
- what additional theorem, representation, or assumption would be required.

Do not stop after planning or reviewing. Implement and verify the correction unless there is a demonstrated mathematical blocker.

## Completion checklist

### Preparation

- [ ] Inspect `git status` and preserve unrelated user changes.
- [ ] Build and run the relevant baseline tests before editing.
- [ ] Reproduce `Wrong1` with the current implementation.
- [ ] Record the invalid path and identify which polygon/contact is missed.
- [ ] Read the relevant disjoint and intersecting sections of Tan and Jiang.
- [ ] Read the relevant Dror et al. sections describing the existing map/query model.

### Mathematical audit

- [ ] Document last-step-map and query invariants.
- [ ] Document first-contact and last-contact semantics for closed polygons.
- [ ] Determine which disjoint invariants fail under intersections.
- [ ] Map pseudo-vertices and pseudo-edges to concrete data structures.
- [ ] Define crossing, reflection, and bending regions precisely.
- [ ] Handle simultaneous and zero-length consecutive visits.
- [ ] Classify every existing intersecting-specific mechanism as faithful, equivalent, heuristic, or incorrect.
- [ ] Explain the exact root cause of `Wrong1`.

### Implementation

- [ ] Preserve the Dror-based binary-search + memoization core.
- [ ] Preserve the pairwise-disjoint fast path.
- [ ] Implement pseudo-vertex discovery.
- [ ] Split affected polygon boundaries into pseudo-edges.
- [ ] Implement correct crossing-region construction and merging.
- [ ] Implement pseudo-vertex bending regions.
- [ ] Implement reflection behavior for pseudo-edges.
- [ ] Update point-location/query recursion for overlapping polygons.
- [ ] Remove production dependence on linear-search fallback.
- [ ] Remove or isolate unsupported optimizer/repair heuristics.
- [ ] Add paper references in non-obvious implementation comments.

### Deterministic tests

- [ ] Add `Wrong1` as a permanent automated regression.
- [ ] Test ordinary disjoint inputs.
- [ ] Test proper two-boundary-point intersections.
- [ ] Test containment.
- [ ] Test vertex tangency.
- [ ] Test collinear shared boundary portions.
- [ ] Test a common point shared by three or more polygons.
- [ ] Test simultaneous consecutive visits.
- [ ] Test nonconsecutive polygon intersections.
- [ ] Test start/target in polygon interiors and on boundaries.
- [ ] Test interior pass-through and vertex-only contact.
- [ ] Test both polygon orientations.
- [ ] Test near-degenerate valid convex inputs.
- [ ] Test repeated geometric contacts at distinct visit indices.

### Independent validation

- [ ] Validate ordered visits along continuous path segments.
- [ ] Verify returned path length independently.
- [ ] Compare small cases with an independent oracle.
- [ ] Run randomized small-instance comparisons.
- [ ] Run translation, rotation, and scaling metamorphic tests.
- [ ] Run orientation-reversal tests.
- [ ] Run redundant-collinear-vertex tests where supported.

### Continuity tests

- [ ] Build parameterized disjoint/tangent/overlapping instance families.
- [ ] Approach contact from positive and negative `delta` values.
- [ ] Cover vertex-edge and edge-edge transitions.
- [ ] Cover near-collinear and containment transitions.
- [ ] Cover consecutive and nonconsecutive intersections.
- [ ] Check optimal-length continuity with scale-aware tolerances.
- [ ] Compare geometric paths when the optimum is locally unique.
- [ ] Allow representative-path changes when multiple optima exist.
- [ ] Compare both sides of the fast-path dispatch boundary.

### Regression and performance

- [ ] Run all relevant existing convex-TPP tests.
- [ ] Run sanitizers where available.
- [ ] Confirm no regression in disjoint correctness.
- [ ] Benchmark disjoint performance before and after.
- [ ] Benchmark representative intersecting cases before and after.
- [ ] Confirm the intersecting implementation uses caching and binary search.
- [ ] Check implementation complexity against the paper-derived design.

### Handoff

- [ ] Summarize the root cause and mathematical correction.
- [ ] List files changed.
- [ ] Report every test and benchmark result.
- [ ] Document tolerance choices.
- [ ] Document remaining limitations honestly.
- [ ] State explicitly whether the final implementation is faithful to the cited constructions.

## Dashboard integration

The benchmark dashboard owns the browser solver artifacts. After changing the
native intersecting solver, rebuild both the native regression target and the
WASM adapter, then run the adapter's browser/Node regression:

```bash
cmake --preset convex-release -DTARGET=main-intersection_tests -DTPP_ENABLE_GUROBI=OFF
cmake --build --preset convex-release
.build/convex-release/packages/convex-tpp/cpp/tpp-convex

bash apps/benchmark-dashboard/wasm/build.sh
node apps/benchmark-dashboard/wasm/test-intersections.mjs
```

The dashboard imports `/static/wasm/tpp_convex_wasm.js`; bump
`WASM_SOLVER_VERSION` in `apps/benchmark-dashboard/static/editor-solver.js`
when the generated module or its calling convention changes.
