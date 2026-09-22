# Convex solver comparison inside B&B (15 September 2026)

The nonconvex B&B benchmark now accepts `TPP_BENCH_SOLVER=directional_maps`,
which forces the intersection-capable directional map engine on every convex
path and length call. `TPP_BENCH_REQUIRE_DISJOINT=1` excludes a case whenever
any two convex hulls touch or intersect. This makes the explicit disjoint solver
and the directional engine comparable on the same valid B&B instances. The
normal public `binary_search_lazy` API dispatches disjoint calls to the
disjoint core, so it cannot measure the directional engine on this workload.

Run from the repository root after configuring/building the nonconvex Release
benchmark target. These runs used the canonical-v1 suite, one worker, the first
20 accepted cases, at most 40 polygons, 256 convex calls, branching cap 6,
and five repetitions. Both runs used the same benchmark executable and Mac.

```bash
cmake --preset nonconvex-release -DTARGET=main-bnb_workload_benchmark
cmake --build --preset nonconvex-release -j 4
TPP_BENCH_THREADS=1 TPP_BENCH_REQUIRE_DISJOINT=1 \
  TPP_BENCH_SOLVER=binary_search_disjoint \
  .build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  benchmarks/suites/canonical-v1.bin 40 20 256 6 5 \
  benchmarks/results/disjoint-comparison.csv \
  benchmarks/results/disjoint-comparison.md
TPP_BENCH_THREADS=1 TPP_BENCH_REQUIRE_DISJOINT=1 \
  TPP_BENCH_SOLVER=directional_maps \
  .build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp \
  benchmarks/suites/canonical-v1.bin 40 20 256 6 5 \
  benchmarks/results/directional-comparison.csv \
  benchmarks/results/directional-comparison.md
```

Six cases (indices 1, 5, 10, 11, 15, 17) yielded B&B runs; 14 of the first
20 were filtered or otherwise skipped. All six final lengths matched to the
12 decimal places emitted by the harness. The cases with identical convex
call counts give the most controlled end-to-end comparison:

| Case | Calls per run | Median disjoint solver μs/call | Median directional μs/call | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 149 | 2.41 | 3,308.67 | 1,371× |
| 10 | 85 | 3.27 | 5,586.15 | 1,706× |
| 11 | 17 | 0.65 | 857.85 | 1,316× |

Across these 15 paired runs, each engine made 1,255 convex calls. The
disjoint engine spent 0.003349 s inside its convex calls; the directional
engine spent 4.908695 s, a 1,466× ratio. B&B wall time within the harness
was 0.004047 s versus 4.910148 s. For all six cases combined, the disjoint
run made 2,605 convex calls and spent 0.040664 s in them; the directional run
made 1,350 calls and spent 6.187943 s. Those combined totals are **not** a
fixed-workload speed ratio: floating-point differences at a pruning threshold
altered the search, notably case 15 (256 calls versus 3). Cases 5 and 17 also
used 7 versus 8 calls. No objective difference was reported for these cases.

This comparison measures the cost of forcing the intersection-capable engine
onto *disjoint* geometry. It isolates implementation cost better than comparing
unrelated disjoint and overlapping instances, but does not estimate a typical
overlapping-case workload. The directional solver converts each input double
to an exact binary rational, splits boundaries, builds incident ray queries,
and uses arbitrary-precision rational signs; it currently allocates fresh maps
on each solve. The explicit disjoint solver uses doubles and its reusable
floating-point workspace. Bit growth and map construction explain why the
directional engine can be orders of magnitude slower, especially for input
coordinates that are non-integer doubles. Timings are measurements on this
suite and machine, not a general complexity ratio.

## Native-double diagnostic

An isolated Release build set `TPP_EXPERIMENT_NATIVE_DOUBLE`, replacing the
map's `Scalar` with a wrapper around native `double` while leaving splitting,
symbolic-query control flow, and B&B selection otherwise the same. This is
**not a correct alternative solver**:
the directional test campaign with 100 integer-box and 20 affine-convex cases
reported 20 failures, including feasible but suboptimal routes and locator
invariant exceptions. The rational build reported zero failures on the same
campaign. A concrete deterministic case is now in `main-directional_tests.cpp`.

For the paired B&B cases 1, 10, and 11, the three-way median comparison was:

| Case | Disjoint μs/call, before scan removal | Native-double maps μs/call | Rational maps μs/call | Native/disjoint | Rational/native |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.41 | 16.32 | 3,308.67 | 6.8× | 203× |
| 10 | 3.27 | 32.49 | 5,586.15 | 9.9× | 172× |
| 11 | 0.65 | 3.10 | 857.85 | 4.8× | 277× |

The native-double diagnostic's total
solver time was 0.116696 s over 2,615 calls; its checksum became `inf` in
case 5, exposing a nonfinite intermediate length query. Thus most of the
thousand-fold measured difference comes from arbitrary-precision arithmetic
and its bit growth, but simply replacing it with unguarded doubles breaks
correctness. A filtered-predicate or certificate-gated design needs separate
engineering and validation.

The precision failure has s=(-3,-2), t=(3,-2),
P1=[-2,0]×[0,2], and P2 with vertices
(-10^-6,1), (1.999999,0), (1.999999,2). The native-double map returns
`s → (0,0.9999995) → t`, length 8.485280667131818. Ordered-path validation
accepts it. The rational map returns a path through (0,0) and
(1.6799996719999488,0.15999966400002566), length 7.824555423364391.
The certified solver brackets that optimum between 7.824555423345182 and
7.824555423364391 without its own refinement fallback. A feasibility-only
double-first/rational-second dispatcher would therefore return the longer
route. B&B length bounds need a certified lower bound as well; accepting a
finite but overestimated double bound could wrongly prune a branch.

An independent dual-gap check is a more promising gate than ordered-path
validation alone. In the precision fixture, the existing `certified` API
linked to the experimental native-double map detected the gap, invoked its
long-double numerical refinement, and bracketed the optimum between
7.824555423335094 and 7.824555423401776. In a 100-call local microbenchmark
it took 81.84 μs/call, versus 130.09 μs/call for the rational map's
length-only call. The native map's unchecked length call took 1.11 μs/call
and returned the wrong value. These timings show that double-first plus a
certificate is worth prototyping; they do not make the current numerical
certificate a strict bound suitable for B&B pruning on every input.

Tan and Jiang's O(kn) disjoint and O(k²n) intersecting bounds count geometric
arithmetic as constant-cost operations. They do not bound the number of bits
in the repeated rational reflections used by this implementation. The current
code also does more work than the paper's stated construction: all-pairs edge
checks, split-event sorting, and fresh exact maps on every B&B call. The
thousand-fold timing is a practical implementation failure for this workload,
not a contradiction of the paper's abstract extra-k operation count.

The optional experiment is reproducible without changing production builds:

```bash
cmake -S packages/convex-tpp/cpp -B .build/native-double-experiment \
  -DTARGET=main-directional_tests -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-DTPP_EXPERIMENT_NATIVE_DOUBLE
cmake --build .build/native-double-experiment -j 4
.build/native-double-experiment/tpp-convex \
  --random-boxes 100 --random-convex 20
```

## Disjoint scan removal

The disjoint locator's binary search already returns a circular fan cell.
Previously it scanned all first-contact cells when the selected cell lacked
an incident first-contact edge. A seeded probe of 100,000 disjoint instances
found that every observed scan result was crossing (`-1`). Replacing the scan
with direct crossing classification preserved all paths' aggregate length
checksum (1193440.192786), a bitwise hash of every returned route coordinate
(6635821323594792249), and ordered feasibility on that probe. The
handwritten, small, medium, and many-small legacy corpora passed with the
direct classification. The production code now has no linear escape in this
locator. This remains a predicate/invariant claim rather than a general proof
against every floating-point degeneracy.
The complete 300-case generated corpus also passed a binary-disjoint-only
check against ordered feasibility and its saved reference lengths. The broad
legacy verifier, which runs all five solver entries, was not needed to check
this change to the binary locator.

The first disjoint B&B timing above was recorded before this change. Repeating
the same 20-case/five-repeat run afterward produced the same 2,605 calls and
checksum, with 0.019633 s in disjoint convex calls versus the earlier
0.040664 s. Its short duration makes the exact improvement factor sensitive
to timing variation. The separately reported rational map cost remains large
under either measurement.
