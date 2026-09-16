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
| Exact disjoint degeneracy recoveries | 6 |
| Legacy intersection tests | Passed |
| Intersection audit | 207 checks; 0 failures |
| Unordered exhaustive/interruption suite | 86 exhaustive-order cases and 344 interrupted-search checks passed |

Shadow mode independently runs the rational backend. For disjoint calls it also compares the exact established recurrence with exact directional maps. A fast result is retained only when its conservative interval overlaps the independent exact interval. The deterministic `floating feasible suboptimal` fixture is rejected by the exact local certificate and resolved by rational fallback.

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
```

Use `TPP_BENCH_SOLVER=hybrid_unchecked` only for diagnostic timing; it deliberately provides no safe pruning guarantee.
