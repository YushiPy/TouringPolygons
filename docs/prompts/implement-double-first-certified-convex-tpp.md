# Implement a double-first certified convex TPP solver

Work in `/Users/gabriel/Documents/Scripts/TouringPolygons` and carry this task through implementation, correctness testing, B&B integration, benchmarking, and a written results report. Inspect the current code and repository instructions before editing. Preserve unrelated work and do not discard existing changes.

## Objective

Build certified hybrid paths for both convex solvers:

1. Compute a candidate with the appropriate disjoint or intersection-capable solver using native `double`.
2. Return one ordered contact per polygon and construct a mathematically sound local optimality certificate.
3. Accept the double result only when feasibility and optimality are rigorously certified.
4. Otherwise fall back to the corresponding `boost::multiprecision::cpp_rational` solver: a rational form of the established disjoint algorithm for pairwise-disjoint inputs, or the rational directional-map solver for intersecting inputs.
5. Integrate the hybrid convex oracle into the nonconvex branch-and-bound solver.
6. Measure speed and the rational-fallback rate on the actual B&B benchmark instances, including fallback reasons.

The expected outcome is a fast convex oracle with no false acceptance of a feasible but suboptimal double solution. The common case remains the current fast double disjoint solver or a native-double directional-map solve; rational arithmetic is paid only after a failed or uncertain certificate.

## Mathematical contract

For input start `s`, target `t`, and convex polygons `P_1, ..., P_k`, the core returned solution contains exactly

```text
x_1, ..., x_k
```

and does not contain `s` or `t`. Preserve duplicate contacts; a contact may equal `s`, `t`, or an adjacent contact.

For intersecting polygons, `x_i` is the first **ordered** contact with `P_i`: the first point on the returned path that belongs to `P_i` after the contact chosen for `P_{i-1}`. It is not necessarily the path's global first intersection with `P_i`, because the path may cross `P_i` before completing earlier required visits.

For pairwise-disjoint polygons, use the analogous explicit vertex/edge contact, but for a straight pass-through visit return the **last** contact with `P_i`. The disjoint last-step map already exposes the exit side, and any point on the straight intersection interval is equally valid for optimality. Do not add expensive first-entry recovery to ordinary disjoint B&B calls. Apply the same rigorous local optimality certificate to disjoint double results; doubles are not assumed infallible merely because the polygons are disjoint.

Provide a compatibility/helper layer that reconstructs a display polyline by adding `s` and `t` and optionally removing redundant collinear points. Do not remove entries from the core `k`-contact representation.

The fixed-order convex TPP optimal polyline is unique. The contact convention above makes the intersecting result canonical while retaining the cheapest useful convention for disjoint pass-through contacts.

## Contact construction

Do not recover contacts with a general linear scan over all polygon edges.

- Vertex region: the contact is the region vertex.
- Edge/reflection region: the contact is the finite-edge reflection point.
- Disjoint pass-through region: return the exit/last contact already identified by the final query machinery.
- Intersecting pass-through/inherited region: find the first ordered contact. Use the directional-map trace or a forward traversal of the final exact bend path. A segment known to cross a convex polygon can locate its entry edge in logarithmic time by binary search over the correctly oriented contiguous boundary chain. Handle circular wraparound, parallel supporting lines, collinear overlap, tangency, and vertex hits explicitly.

Only perform full contact materialization for a final path/certificate query. Length-only internal operations should retain the cheapest representation possible, unless certification is required before their value can safely be used by B&B.

Carry contact provenance rather than relying only on rounded coordinates. Useful provenance includes polygon/vertex index, original edge plus parameter, or crossing segment plus entry/exit edge. Split pseudo-edges are construction details; returned membership is with respect to the original polygon.

## Optimality certificate

Use the local necessary-and-sufficient conditions for the convex contact-coordinate problem. This is the primary certificate; do not substitute feasibility-only validation or accept a path merely because it is close to a reference length.

For the chain

```text
s, x_1, ..., x_k, t
```

verify:

1. Every `x_i` belongs to `P_i` under the required contact convention.
2. Every genuine bend occurs at a polygon boundary contact.
3. An interior/pass-through contact is straight: incoming and outgoing directions agree.
4. At an edge-interior bend, incoming and outgoing angles with the edge agree, with the correct orientation.
5. At a vertex bend, the outgoing direction lies in the incidence cone determined by the incoming direction and the two incident polygon edges.

For convex polygons these local conditions are necessary and sufficient for global optimality. Prefer exact direction/orientation/reflection predicates that avoid square roots where possible.

The certificate must be mathematically sound for the input binary-double coordinates. A rounded coordinate being merely close to an edge or an equal-angle condition is insufficient. Use construction provenance, filtered predicates with proved error bounds, and exact evaluation of an individual uncertain predicate where needed. Predicate-level exact fallback is acceptable and must be timed/counted separately; it should not automatically force construction of all rational directional maps.

If the certified optimal value is passed to B&B as a lower bound, evaluate it conservatively so floating rounding cannot turn it into an overestimate that incorrectly prunes a branch.

### Coincident contacts

Initially use a conservative policy for `x_i == x_{i+1}` or for a segment that cannot be proved nonzero. Fall back to the rational solver unless a simple rigorous zero-link witness is already available. Record these fallbacks separately. Do not silently invent a direction for a zero-length link.

The benchmark must report how often this occurs. If zero-link fallback is a material fraction of practical B&B calls, explain what additional subgradient/direction information would be needed and, if reasonably bounded, implement and benchmark that extension separately. Keep the initial and extended measurements distinguishable.

## Solver architecture

The production binary must contain both native-double and rational implementations so it can fall back during one call. It needs a double and rational form of the established disjoint solver as well as double and rational intersection-capable directional maps. Prefer shared templated geometry and query logic over copied implementations that can diverge. The current `TPP_EXPERIMENT_NATIVE_DOUBLE` compile-time substitution is only an experiment and is not sufficient for the hybrid.

Expose two clearly distinguished API modes backed by the same solver core:

- **Safe/certified mode**, which is the production default: compute the double candidate, construct the required contacts, run the rigorous local certificate, and invoke the matching rational fallback after any failed or uncertain condition.
- **Unchecked mode**, explicitly named as such: return the native-double result without final optimality verification or rational fallback. This mode assumes numerically well-conditioned input and carries no correctness guarantee. It exists for trusted-input experiments, diagnostics, and measuring certificate overhead; B&B must use the safe mode by default.

Do not maintain two copies of the algorithm. Use a policy/option or thin wrappers around one implementation. The unchecked path should skip only work that is exclusively required for certification/fallback. If a caller requests the `k`-contact output, it must still receive the documented contact representation; a length-only unchecked call may skip contact materialization when the underlying algorithm can compute the length without it.

The hybrid attempt must fall back on at least:

- locator or refolding exception;
- nonfinite coordinate or length;
- failure to construct exactly `k` ordered contacts;
- unproved polygon membership or contact ordering;
- uncertain or failed local optimality predicate;
- coincident/zero-length link not handled by a rigorous witness;
- any other open certificate condition.

Give each fallback a distinct counter. A certificate failure means “not certified,” not necessarily “suboptimal.”

The rational fallback must return the same contact contract as the accepted double path. Keep separate rational-disjoint and rational-intersection modes for tests and benchmarks. Do not automatically route a failed disjoint certificate through the much heavier intersection construction when the established disjoint algorithm can be evaluated with exact predicates.

Pairwise-disjoint calls should continue using the established fast disjoint algorithm as their first attempt. Update its returned contact representation and certify it without imposing intersection-map construction on length queries or ordinary B&B calls. Catch locator/query exceptions and fall back to the rational disjoint implementation.

## Existing evidence and code to inspect

Read these before designing the change:

- `packages/convex-tpp/cpp/src/solvers/intersecting_maps.cpp`
- `packages/convex-tpp/cpp/src/solvers/binary_search.cpp`
- `packages/convex-tpp/cpp/src/core/solution.cpp`
- `packages/convex-tpp/cpp/src/core/ordered_path_validation.cpp`
- `packages/convex-tpp/cpp/src/solvers/certified.cpp`
- `packages/convex-tpp/cpp/src/solvers/certified_geometry.cpp`
- `packages/convex-tpp/cpp/src/main-directional_tests.cpp`
- `packages/convex-tpp/cpp/src/main-intersection_audit.cpp`
- `packages/nonconvex-tpp/cpp/src/solvers/unordered.cpp`
- `packages/nonconvex-tpp/cpp/src/main-bnb_workload_benchmark.cpp`
- `docs/convex-bnb-comparison-2026-09-15.md`
- `docs/reports/directional-tpp-tutorial/main.tex`

Important established facts:

- The rational directional-map implementation passes the current campaign but still lacks a publication-level proof covering every intersecting degeneracy.
- The existing disjoint solver has passed the legacy corpus, a 300-case generated corpus, and a seeded 100,000-instance scan-removal probe, but this is evidence rather than a proof that native doubles cannot misclassify a sufficiently ill-conditioned disjoint case.
- Forcing rational maps on controlled disjoint B&B calls was about 1,300–1,700 times slower than the disjoint solver.
- The experimental native-double maps were roughly 170–280 times faster than rational maps, but failed correctness tests.
- The deterministic `floating feasible suboptimal` case in `main-directional_tests.cpp` is essential: native doubles return a feasible path of length about `8.4852806671`, while the rational/certified optimum is about `7.82455542336`. Feasibility-only validation accepts the wrong double path.
- The existing `ordered_path_validation.cpp` verifies ordered feasibility, not optimality.
- The existing `certified` support-bound code is useful evidence and instrumentation, but its greedy contact recovery and heuristic zero-direction filling are not the requested final local certificate.

## Required verification

Add focused tests for the new contact contract and certificate. At minimum verify:

1. Result size is exactly `k`; `s` and `t` are not implicit extra entries.
2. Duplicate contacts are preserved for identical, nested, shared-edge, shared-vertex, and common-point polygons.
3. Intersecting contacts are first ordered contacts, including paths that cross a later polygon too early and return to it after earlier visits.
4. Disjoint pass-through contacts use the exit point and do not change the path geometry or optimal length.
5. Vertex, edge-reflection, straight crossing, tangency, collinear boundary overlap, reversed winding, stationary start/target, and repeated-polygon cases.
6. Reconstructed polylines visit every polygon in order.
7. Every accepted double candidate passes the rigorous local optimality certificate.
8. `floating feasible suboptimal` is rejected by the double certificate and resolved by rational fallback.
9. All existing directional, intersection-audit, disjoint, and unordered/B&B tests continue to pass.
10. Randomized shadow testing runs both hybrid and rational solvers and proves there are no false fast-path acceptances in the tested corpus. Compare objective values using certified/conservative bounds, not feasibility alone.
11. Adversarial disjoint tests cover near-collinearity, very small and very large scales, thin polygons, near-tangency, and coordinates whose exact binary values make orientation/reflection predicates difficult. Every accepted double-disjoint result must agree with its rational-disjoint shadow certificate; any rejected result must take the rational-disjoint fallback.

Avoid tests that merely duplicate implementation details. Preserve deterministic seeds and print enough geometry to reproduce any failure.

## Benchmark and instrumentation

Benchmark Release builds. Use one worker/thread for controlled comparisons and warm up before timing. Compare at least:

- current rational-only intersection solver;
- specialized rational disjoint solver;
- unchecked native-double candidate, for diagnostic/trusted-input timing only;
- safe hybrid double-plus-certificate solver;
- established disjoint solver on pairwise-disjoint calls.

For disjoint inputs, directly measure the incremental cost of adding pass-through contacts and of final certification. The expected work is only `O(1)` per polygon, hence `O(k)` total, but report measured absolute microseconds/nanoseconds per call and percentage overhead rather than assuming it is negligible. Include safe-versus-unchecked comparisons both in a focused convex microbenchmark and end-to-end B&B runs.

Integrate counters and exclusive timings for:

- total convex calls;
- disjoint dispatches;
- certified double-disjoint calls;
- rational-disjoint fallbacks and their reasons;
- intersection double attempts;
- fast-path certifications;
- rational fallbacks;
- fallback reason counts;
- coincident-contact/zero-link cases;
- predicate-level exact evaluations;
- double map construction/query time;
- contact materialization time;
- certificate time;
- rational fallback time;
- complete convex-oracle time.

Run the actual nonconvex B&B benchmark suite, including `benchmarks/suites/canonical-v1.bin`, with paired configurations and identical limits, repetitions, and thread count. Report:

- rational fallback rate overall, separately for disjoint and intersecting calls, and by reason;
- fallback rate weighted by convex-call time;
- median and aggregate convex time per call;
- end-to-end B&B wall time;
- final lower/upper bounds, objective, order, termination, and call counts;
- any search-tree divergence caused by floating decisions;
- results stratified by disjoint/intersecting calls and, where useful, polygon count or degeneracy class.

Use two benchmark modes:

1. **Shadow correctness mode:** run the rational solver even after fast certification and compare every accepted candidate. This mode measures false acceptance and certificate coverage, not speed.
2. **Performance mode:** invoke rational maps only on actual fallback.

Do not report a speedup from workloads with different call counts without clearly separating per-call controlled comparisons from end-to-end B&B effects.

## Deliverables and stopping condition

Deliver:

- the implemented hybrid solver and contact-return API;
- compatibility updates for current callers;
- rigorous local certificate and conservative fallback behavior;
- tests and reproducible benchmark commands;
- B&B fallback/timing instrumentation;
- a Markdown report under `docs/` containing correctness results, fallback rates, timing tables, remaining limitations, and the exact commands used.

The task is complete only when the tests pass, shadow mode shows no false accepted double results in the tested corpus, the B&B benchmark results are recorded, and the final response clearly states whether the hybrid achieved a practical speedup and what caused every significant fallback category.
