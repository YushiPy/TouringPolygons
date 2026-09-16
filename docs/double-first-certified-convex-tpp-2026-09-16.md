# Double-first convex TPP prototype results (16 September 2026)

## Outcome

This change implements and exercises a conservative double-first prototype, but it does **not** meet the task's stopping condition and should not be described as the final certified solver.  The exact local certificate rejected the known feasible-but-suboptimal native-double fixture and shadow testing found no false fast-path acceptance.  However, fast-path coverage was only 17.8% on the measured B&B workload, the safe mode was slower than the rational-only baseline, the disjoint fallback still uses the general rational directional-map construction, and the rational fallback's exported binary64 objective interval is not yet a proof-grade directed rounding of the exact radical sum.

The prototype therefore establishes a safe acceptance gate for its fast path and identifies the remaining engineering work, but it does not establish a fully rigorous B&B lower-bound oracle end to end.

## Implemented pieces

- The same directional-map source is compiled twice into the production library: exact `cpp_rational` and explicitly unchecked native binary64.  There are not two maintained algorithm copies.
- `tpp_convex_solve_hybrid` exposes safe/certified and unchecked policies.  Its core result contains exactly one contact per polygon, excludes the endpoints, and preserves duplicates.  `reconstruct_convex_polyline` supplies the compatibility display path.
- Exact binary-rational segment clipping materializes first ordered contacts for intersecting calls and last/exit contacts for disjoint calls.
- The fast-path certificate checks exact polygon membership and the convex local KKT condition.  For each contact `x_i` and every feasible vertex direction `v-x_i`, it proves
  `(<x_i-x_{i-1},v-x_i>/|x_i-x_{i-1}|) - (<x_{i+1}-x_i,v-x_i>/|x_{i+1}-x_i|) >= 0`.
  Signs of the two-radical expressions are decided exactly by sign separation and squared rational comparison; no tolerance is used.  A zero adjacent link always falls back.
- Accepted fast paths receive directed binary64 bounds assembled from exact rational squared distances.
- The unordered solver uses the hybrid entry, and its result/JSON instrumentation now reports exact-predicate work, contact time, and distinct fallback categories.
- The canonical B&B harness accepts `TPP_BENCH_SOLVER=hybrid_safe` and `hybrid_unchecked` and emits aggregate hybrid counters and exclusive timings.

## Correctness results

| Campaign | Result |
|---|---:|
| Directional/contact suite, 100 integer boxes + 20 affine cases | 1,594 checks, 0 failures, 0 unresolved |
| Shadow results in that campaign | 18 fast, 120 fallback, 0 shadow mismatches |
| Intersection audit, 100 random cases | 407 checks, 0 failures; 0 feasible-suboptimal; 0 unresolved |
| Legacy intersection tests | Passed |
| Unordered exhaustive/interruption suite | 86 exhaustive orders and 344 interrupted checks passed |

The deterministic `floating feasible suboptimal` case is rejected by the local certificate and resolved through rational intersection fallback.  In the tested shadow corpus no accepted fast result disagreed with the rational solver.

The contact tests cover exact result cardinality, duplicate contacts, reconstructed ordered visitation, reversed winding, shared/nested/common contacts, stationary endpoints, tangencies, thin polygons, repeated polygons, and the out-of-order later-polygon case.  The rational directional and established disjoint regression campaigns remain separate evidence for the construction backends.

## Canonical B&B measurement

Release build, one worker, first 20 accepted cases from `canonical-v1.bin`, at most 40 polygons, 128 convex calls, branching cap 6, one repetition.  Different call counts mean the end-to-end rows are not fixed-work speed ratios.

| Mode | Convex calls | Convex time | Mean reported time/call | B&B time | Checksum |
|---|---:|---:|---:|---:|---:|
| safe hybrid | 1,273 | 17.790 s | 13,974.95 us | 17.793 s | 13,046,847.7641344 |
| unchecked hybrid | 1,147 | 22.024 s | 19,201.68 us | 22.027 s | 12,043,472.4060223 |
| established dispatch | 1,212 | 7.167 s | 5,913.63 us | 7.170 s | 10,382,371.7224622 |
| rational directional maps | 1,089 | 8.729 s | 8,015.77 us | 8.732 s | 10,255,497.6673523 |

Safe-hybrid counters on the 1,273-call run:

| Counter | Value |
|---|---:|
| Disjoint dispatches | 482 |
| Certified double disjoint | 115 |
| Certified double intersection | 112 |
| Rational disjoint fallbacks | 367 |
| Rational intersection fallbacks | 679 |
| Overall fallback rate | 82.17% |
| Disjoint fallback rate | 76.14% |
| Intersecting fallback rate | 85.84% |
| Fallback-time share of complete oracle time | 96.72% |

Fallback reasons (1,046 fallbacks): contact construction 644 (61.6%), coincident contacts 202 (19.3%), locator/refolding exception 103 (9.8%), and failed exact local optimality 97 (9.3%).  There were no nonfinite, membership/ordering, or shadow-mismatch fallbacks.  Exact classification/dispatch took 0.146 s, the double solver 0.047 s, contact materialization 0.290 s, certification 0.068 s, rational fallback 17.204 s, and the complete oracle 17.788 s.

The checksum and search-tree differences between modes reinforce that end-to-end runtime is not a controlled solver comparison.  In particular, unchecked mode is diagnostic only.

## Fixed-work disjoint microbenchmark

One generated disjoint instance with `k=20`, eight vertices per polygon, 5 warmups and 50 measured repetitions:

| Operation | us/call |
|---|---:|
| Established disjoint length | 7.11 |
| Unchecked hybrid with contacts | 787.45 |
| Safe hybrid | 7,694.07 |
| Rational directional length | 6,387.63 |
| Rational directional contacts | 7,246.52 |

This instance had no fast certification.  Within the hybrid calls, mean dispatch was 145.39 us, double solve 5.79 us, contact materialization 596.76 us, certificate 20.48 us, and rational fallback 3,508.49 us (the aggregate mixes unchecked and safe calls, so fallback time is averaged over both).

The expected small `O(k)` certification overhead was not obtained: reconstructing exact contacts after discarding construction provenance dominates the unchecked path.  Carrying original vertex/edge/crossing provenance through both double cores is required to remove this cost and certify reflection identities without demanding accidental exact equality of rounded exported coordinates.

## Remaining blockers

1. Implement a genuinely specialized rational form of the established disjoint algorithm.  The current `RationalDisjoint` dispatch label still uses rational directional maps, so it does not satisfy the required architecture.
2. Carry contact and combinatorial provenance through the double algorithms.  This should eliminate most of the 644 construction fallbacks and reduce contact recovery from exact clipping of a rounded path to constant work per polygon.
3. Add a rigorous zero-link/subgradient witness.  Zero links caused 19.3% of fallbacks and are material on this workload.
4. Produce proof-grade directed bounds for the rational radical sum before using rational-fallback values as B&B pruning bounds.  The current fallback converts exact contacts to binary64 and expands the measured sum by one ulp, which is not a proof for arbitrary coordinates.
5. Replace the current linear half-plane clip used for final contact materialization with the requested logarithmic contiguous-chain entry search.
6. Re-run paired benchmarks with identical call traces after the above changes, then increase repetitions.  The present single-repetition end-to-end data is diagnostic, not a stable speed estimate.

## Reproduction commands

```bash
cmake --preset convex-release -DTARGET=main-directional_tests
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex --random-boxes 100 --random-convex 20

cmake --preset convex-release -DTARGET=main-intersection_audit
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex --directional-maps --random 100

cmake --preset convex-release -DTARGET=main-intersection_tests
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex

cmake --preset nonconvex-release -DTARGET=main-unordered_tests
cmake --build --preset nonconvex-release -j 4
.build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp

cmake --preset convex-release -DTARGET=main-hybrid_benchmark
cmake --build --preset convex-release -j 4
.build/convex-release/packages/convex-tpp/cpp/tpp-convex 50

cmake --preset nonconvex-release -DTARGET=main-bnb_workload_benchmark
cmake --build --preset nonconvex-release -j 4
TPP_BENCH_THREADS=1 TPP_BENCH_SOLVER=hybrid_safe \
  .build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  benchmarks/suites/canonical-v1.bin 40 20 128 6 1 \
  benchmarks/results/hybrid-20260916/safe.csv \
  benchmarks/results/hybrid-20260916/safe.md
```

Repeat the last command with `hybrid_unchecked`, `binary_search_lazy`, and `directional_maps`; the corresponding result files are under `benchmarks/results/hybrid-20260916/`.
