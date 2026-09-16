# Double-first certified convex TPP results (16 September 2026)

## Outcome

The production convex oracle now uses a certified double-first path and achieves a practical speedup on the measured canonical B&B workload. Safe mode accepts native-double map decisions only after replaying their contact provenance in exact binary-rational arithmetic, constructing exactly one ordered contact per polygon, checking the exact local convex KKT conditions, and enclosing the radical objective with directed bounds. An uncertified candidate falls back to the matching exact solver.

On the canonical workload, safe hybrid took 6.05 seconds in the convex oracle, versus 7.17 seconds for the established dispatch and 8.73 seconds for rational directional maps: speedups of 1.19x and 1.44x respectively. All 358 disjoint calls certified without fallback; 343 of 744 intersecting calls certified. The overall rational-fallback rate fell from 89.56% in the earlier prototype to 36.39%.

## Implemented architecture

- `tpp_convex_solve_hybrid` exposes safe and explicitly unchecked policies. The core result contains exactly `k` contacts, excludes `s` and `t`, and preserves duplicate contacts. `reconstruct_convex_polyline` is the compatibility/display helper.
- Both established-disjoint and directional-map double solvers emit their final-query combinatorial trace. Vertex definitions retain original-point or original-edge-intersection provenance; edge regions retain the original supporting edge.
- The hybrid replays that trace exactly with `cpp_rational`. This avoids treating rounded double bends as exact while retaining the inexpensive double map construction and locator decisions.
- Pairwise-disjoint fallback uses an exact rational implementation of the established recurrence. Undefined zero-incoming-cone degeneracies recover through exact disjoint directional maps and are counted separately.
- Intersecting fallback uses exact rational directional maps.
- Pass-through contacts use an exact logarithmic line/convex-polygon locator after boundary-angle preprocessing. It finds support extrema and binary-searches the two monotone boundary chains. Tangency, vertex hits, circular wraparound, parallel lines, and collinear supporting-edge overlap are handled explicitly.
- Genuine vertex/edge bends use trace provenance directly, without a boundary scan. Disjoint straight crossings select the exit contact; intersecting crossings select the first ordered contact.
- The local certificate uses the equivalent local feasible-cone test: straightness in the interior, tangent equality plus inward orientation on an edge, and the two incident feasible rays at a vertex. Normalized-direction predicate signs are decided exactly by rational squared comparisons.
- Coincident contacts first use the simple aligned-direction witness. A bounded exact dynamic program additionally propagates admissible directions through several coincident edge/vertex normal cones; every transition is independently checked by the exact KKT predicate.
- Accepted and fallback objectives use 96-bit dyadic integer-square-root enclosures and directed binary64 conversion.
- Unchecked length calls skip exact polygon construction, contacts, certification, and rational fallback. They are diagnostic only and may return an invalid B&B bound.
- The unordered solver uses safe hybrid by default, and the B&B harness reports backend counts, fallback reasons, exact predicates, zero-link witnesses, and exclusive timings.

## Correctness verification

| Campaign | Result |
|---|---:|
| Directional/contact shadow suite, 1,000 integer-box + 200 affine cases | 10,257 checks; 0 failures; 0 unresolved |
| Hybrid outcomes in that campaign | 994 certified fast; 224 fallbacks |
| Independent conservative-interval comparison | 0 shadow mismatches |
| Disjoint calls / rational-disjoint fallbacks | 320 / 1 (0.31%) |
| Exact disjoint degeneracy recoveries during rational shadow evaluation | 6 |
| Legacy intersection tests | Passed |
| Intersection audit | 207 checks; 0 failures |
| Unordered exhaustive/interruption suite | 86 exhaustive-order cases and 344 interrupted-search checks passed |

Shadow mode independently runs the rational backend even when the double candidate certifies. For disjoint calls it also compares the exact established recurrence with exact directional maps. The six disjoint degeneracy recoveries therefore do **not** denote six production fallbacks: they mean the rational established recurrence was undefined during shadow evaluation and the independent exact directional construction completed that comparison. A fast result is retained only when its conservative interval overlaps the independent exact interval.

There was one actual rational-disjoint fallback in the full stress campaign: the deliberately adversarial `disjoint near collinear` fixture, containing three separated boxes only `1e-14` high. Exact replay could not materialize the double trace's contacts (`contact_construction`), so safe mode conservatively used the rational-disjoint solver. The shadow comparison found no objective mismatch, so this is a failure to certify the double candidate, not evidence that its objective was wrong. The deterministic `floating feasible suboptimal` fixture is separately rejected by the exact local certificate and resolved by rational-intersection fallback.

Focused tests cover exact contact cardinality, identical/common-point duplicates, disjoint exit contacts, reversed winding, repeated polygons, stationary endpoints, tangency, collinear boundary overlap, thin and near-collinear polygons, and very small/large coordinate scales.

## Canonical B&B benchmark

Release build, one worker, first 20 accepted cases from `canonical-v1.bin`, at most 40 polygons, at most 128 convex calls per instance, branching cap 6, one repetition.

| Mode | Convex calls | Convex time | Mean time/call | B&B time | Checksum |
|---|---:|---:|---:|---:|---:|
| Safe hybrid, final | 1,102 | 6.045 s | 5,485.79 us | 6.047 s | 10,983,296.8177918 |
| Safe hybrid, initial exact-bounds version | 1,102 | 9.730 s | 8,829.52 us | 9.732 s | 10,983,296.8177918 |
| Established dispatch | 1,212 | 7.167 s | 5,913.63 us | 7.170 s | 10,382,371.7224622 |
| Rational directional maps | 1,089 | 8.729 s | 8,015.77 us | 8.732 s | 10,255,497.6673523 |
| Unchecked diagnostic, final | 738 | 0.041 s | 56.01 us | 0.042 s | `inf` (invalid) |

The safe before/after rows have identical call traces and checksums; the final implementation reduced convex time by 37.9%. Different solver families have different B&B call counts, so their rows are end-to-end comparisons rather than fixed-work speed ratios. The unchecked result demonstrates throughput only: its infinite checksum confirms that it must not be used for correctness-sensitive pruning.

### Final safe counters

| Counter | Value |
|---|---:|
| Total calls | 1,102 |
| Disjoint / intersecting dispatches | 358 / 744 |
| Certified double disjoint / intersection | 358 / 343 |
| Rational disjoint / intersection fallback | 0 / 401 |
| Overall fallback rate | 36.39% |
| Disjoint fallback rate | 0.00% |
| Intersecting fallback rate | 53.90% |
| Locator/refolding exception | 103 |
| Contact construction | 22 |
| Failed local optimality | 20 |
| Unresolved coincident contact | 256 |
| Nonfinite, membership/order, shadow mismatch | 0 / 0 / 0 |
| Exact predicate evaluations | 28,516 |
| Rigorous zero-link witnesses | 1,018 |

Exclusive totals were 0.113 s dispatch, 0.034 s double solving, 0.448 s contact materialization, 0.266 s certification, and 5.142 s rational fallback, for 6.044 s complete oracle time. Rational fallback still consumed 85.1% of total oracle time.

## Fixed-work disjoint microbenchmark

One generated pairwise-disjoint instance with `k=20`, eight vertices per polygon, five warmups and 100 measured repetitions:

| Operation | us/call |
|---|---:|
| Established native-double length | 7.25 |
| Unchecked hybrid length | 11.91 |
| Safe hybrid | 657.11 |
| Exact established disjoint | 4,317.92 |
| Rational directional length | 5,967.45 |
| Rational disjoint-map length | 5,952.20 |
| Rational directional contacts | 6,655.18 |
| Rational disjoint-map contacts | 6,608.85 |

All 105 measured safe calls certified without rational fallback. Safe hybrid is 6.57x faster than the specialized exact established recurrence and 9.08x faster than rational directional length on this fixed input. The remaining cost over the raw double solver is exact polygon construction, trace replay, contact provenance, local certification, and proof-grade objective bounds.

## Matched Gurobi SOCP audit of intersection fallbacks

The canonical run's 401 rational-intersection fallback calls were captured verbatim and replayed through a Gurobi 13.0.3 SOCP matching the German oracle's core formulation: explicit polygon bounding-box and half-plane constraints, quadratic distance epigraphs, presolve disabled, and one solver thread. The corpus contains 317 unique ordered subproblems; repeated B&B states account for the other 84 calls. Returned binary64 contacts were checked both by exact rational half-plane predicates and by geometric distance, and their reconstructed path lengths were compared with the rational oracle's certified interval.

The following complete matrix used a uniform 0.1-second limit per SOCP call. Objective agreement uses `tol * (1 + |objective|)`; exact feasibility means every returned binary64 contact is exactly in its corresponding polygon, with no tolerance.

| Gurobi profile | Incumbents / 401 | Exactly feasible | Objective agrees at `1e-7` | Objective agrees at `1e-9` | `OPTIMAL` statuses |
|---|---:|---:|---:|---:|---:|
| Default (`1e-6` feasibility/QCP) | 401 | 366 | 316 | 53 | 0 |
| Tight (`1e-9` feasibility/QCP) | 347 | 301 | 106 | 34 | 0 |
| Tight + `NumericFocus=3` | 347 | 305 | 100 | 31 | 0 |

All returned incumbents passed the scale-aware `1e-9` geometric membership check; exact membership is lower because a point a few ulps outside a boundary remains feasible to a numerical SOCP tolerance. Every solve stopped with `TIME_LIMIT`, so a returned path is not an optimality proof. On the 317 unique subproblems, the default profile returned 317 incumbents, 297 exactly feasible contact sequences, and 260/51 objective matches at `1e-7`/`1e-9`.

The default-profile objective results by fallback reason were:

| Hybrid fallback reason | Calls | Objective agrees at `1e-7` | Objective agrees at `1e-9` |
|---|---:|---:|---:|
| Coincident contact | 256 | 187 | 33 |
| Locator/refolding exception | 103 | 103 | 0 |
| Contact construction | 22 | 6 | 0 |
| Local optimality | 20 | 20 | 20 |

A longer two-second run used one representative from each category. Only the default locator/refolding case reached Gurobi `OPTIMAL` (in 1.22 seconds); its reconstructed objective agreed at `1e-7` but not `1e-9`. The other 11 profile/case combinations returned incumbents but timed out. Thus Gurobi often finds a useful numerical solution, and it handles all 20 local-optimality rejections accurately at the default profile, but it does not make these normal B&B fallback subproblems numerically trivial or proof-grade. Coincident and contact-construction cases remain the clearest practical difficulties.

## Quality of rejected double candidates and candidate-derived bounds

The same 401 canonical fallback calls were replayed with diagnostic retention of the native-double candidate. A complete ordered candidate exists for 276 calls: all 256 `coincident_contact` rejections and all 20 `local_optimality` rejections. The 103 locator/refolding exceptions and 22 contact-construction failures stop before a complete candidate exists.

The 276 retained candidates are exactly feasible in the internal rational trace replay. Converting their boundary contacts to returned binary64 coordinates makes only 13 strictly feasible under a zero-tolerance rational half-plane test, but every candidate is feasible under the scale-aware `1e-9` geometric check. The median boundary violation is `6.82e-15` distance units, the 90th percentile is `6.04e-13`, and the maximum is `1.23e-12`; this is representation-level boundary rounding rather than the source of the objective error.

The rejected candidate is generally **not** already optimal. Relative error below uses `error / (1 + |certified optimum|)`.

| Candidate category | Calls | Objective agrees at `1e-7` | Median excess | P90 excess | Maximum excess |
|---|---:|---:|---:|---:|---:|
| Coincident contact | 256 | 1 | 0.995% | 2.344% | 3.678% |
| Local optimality | 20 | 0 | 0.166% | 0.628% | 1.143% |
| All retained candidates | 276 | 1 | 0.995% | 2.344% | 3.678% |

For comparison, default Gurobi produced an objective match on 207/276 of these calls at `1e-7` and 53/276 at `1e-9`; 205 and 52 respectively were also exactly feasible in returned binary64 coordinates. Thus Gurobi's time-limited incumbent is usually closer than the rejected directional candidate, but none of these 0.1-second Gurobi solves returned an `OPTIMAL` status. This is an incumbent-quality advantage, not an optimality-proof advantage.

A feasible path length is an **upper** bound, so it cannot safely prune a minimization B&B node as a lower bound. A rigorous lower bound can nevertheless be derived from the candidate's segment directions. For any directions `u_i` with `||u_i|| <= 1`, weak duality gives

`u_n · (t-s) + sum_i min_{x in P_i} (u_{i-1}-u_i) · (x-s) <= OPT`.

The diagnostic implementation normalizes directions outward with 96-bit rational square-root enclosures, evaluates polygon supports exactly, tries several valid choices across zero-length links, and converts the best dual value downward to binary64. All 276 values were safe against the independent rational optimum. This inexpensive candidate-derived certificate was substantially stronger than Gurobi's time-limited SOCP bound, but was not tight enough to replace exact fallback:

| Lower-bound diagnostic on the same 276 calls | Median shortfall from optimum | P90 | Maximum |
|---|---:|---:|---:|
| Candidate-derived exact-support dual | 5.60% | 5.93% | 8.33% |
| Gurobi SOCP bound after 0.1 s | 49.80% | 72.93% | 76.77% |

Combining the candidate dual lower bound with its feasible-path upper bound gave a median certified gap of 6.54%, a P90 of 8.04%, and a maximum of 9.63%; only 36/276 gaps were at most 1%. Consequently it is useful for estimating quality or possibly pruning against a loose incumbent, but it does not establish near-optimality in most failed cases. The production rational fallback remains justified: in the original safe benchmark it averaged about 12.8 ms per fallback and returned a proof-grade optimum, while every matched Gurobi call exhausted its 100 ms limit.

## Remaining limitations

1. Intersecting coincident-contact blocks remain the largest fallback category. The bounded witness proves aligned turns and finite reflection sequences, but 256 canonical calls require a more general continuous subgradient-cone feasibility solver.
2. Native directional locator/refolding invariants still reject 103 canonical calls. Retrying the same source with long double recovered none and was therefore not retained; those cases conservatively use rational maps.
3. The exact and native established-disjoint recurrences still mirror one another rather than sharing all query logic through a scalar-policy template. Their final trace/replay and contact/certificate paths are shared, and shadow testing guards behavioral divergence, but a future structural refactor could consolidate the map recurrence itself.
4. Boundary-angle rotation is prepared in linear time while input polygons are materialized; each crossing query is logarithmic afterward. A reusable preprocessed polygon object would amortize this setup across repeated calls.
5. The canonical benchmark uses one repetition and solver families produce different search trees. More repetitions and a recorded fixed convex-call trace would improve statistical comparison.
6. The rational directional-map construction retains the publication-proof limitation documented by the original implementation, although it passes the current deterministic and randomized campaigns.

## Reproduction

```bash
cmake --preset convex-release -DTARGET=main-directional_tests
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex --random-boxes 1000 --random-convex 200

cmake --preset convex-release -DTARGET=main-intersection_tests
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex

cmake --preset convex-release -DTARGET=main-intersection_audit
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex

cmake --preset nonconvex-release -DTARGET=main-unordered_tests
cmake --build --preset nonconvex-release -j 4
.build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp

cmake --preset convex-release -DTARGET=main-hybrid_benchmark
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex 100

cmake --preset nonconvex-release -DTARGET=main-bnb_workload_benchmark
cmake --build --preset nonconvex-release -j 4
TPP_BENCH_THREADS=1 TPP_BENCH_SOLVER=hybrid_safe \
  .build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  benchmarks/suites/canonical-v1.bin 40 20 128 6 1 \
  benchmarks/results/hybrid-20260916/safe-v6.csv \
  benchmarks/results/hybrid-20260916/safe-v6.md

TPP_BENCH_THREADS=1 TPP_BENCH_SOLVER=hybrid_capture \
  TPP_HYBRID_FALLBACK_CORPUS=benchmarks/results/hybrid-20260916/gurobi-audit-fallbacks.bin \
  .build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  benchmarks/suites/canonical-v1.bin 40 20 128 6 1

cmake --preset convex-release -DTARGET=main-gurobi_fallback_audit -DTPP_ENABLE_GUROBI=ON
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex \
  benchmarks/results/hybrid-20260916/gurobi-audit-fallbacks.bin \
  benchmarks/results/hybrid-20260916/gurobi-audit-fallbacks.bin.csv \
  benchmarks/results/hybrid-20260916/gurobi-audit-details-bounded.csv
```

Use `TPP_BENCH_SOLVER=hybrid_unchecked` only for diagnostic timing; it deliberately provides no safe pruning guarantee.
