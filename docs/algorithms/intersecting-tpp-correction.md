# Corrected directional last-step maps

Implementation report, 13 September 2026.

The intersection solver now uses binary search and memoized directional maps.
The unsupported optimizer and candidate/fallback dispatcher have been removed.
The established disjoint binary-search class is unchanged. The correction passes
the tests listed below, including the previously failing paper counterexample.
**This is an implemented, tested correction, not a machine-checked proof that
this code solves every possible input.** The global proof obligations and
numerical/performance limits are stated explicitly below.

This report supersedes the implementation-status statements in the
[historical audit](intersecting-tpp-audit.md), while preserving that audit's
counterexample and baseline results. The detailed mathematical witness and
figures remain in [the LaTeX report](../reports/intersection-counterexample/report.tex).

## What was wrong

Tan and Jiang §4, printed p. 621, specifies a pseudo-vertex bending region using
two reflections of the last segment of the shortest prefix **at** the
intersection. That segment can lose a reflected piece whose length tends to
zero. The two incident limits then cannot be recovered by independently
reflecting that one segment.

For the report's proper, nondegenerate intersection at j=(0,0), with
s=(-3,-4), the incoming unit vector at j is u=(3/5,4/5). Let R1 reflect across
x=0 and R2 across y=x/2. The literal rule gives R1u=(-3/5,4/5) and R2u=(1,0).
The incident limits actually give **R2R1u=(7/25,-24/25) and R2u=(1,0)**.
The first of these records two consecutive reflections. The bending region
has a reflection region on both sides, contradicting the stated
crossing/reflection adjacency rule as well.

With target (4,-3), the simultaneous visit s→j→t has certified optimum 10.
With target (-4,-3), the optimum is

    (-3,-4) → (0,-3) → (-18/5,-9/5) → (-4,-3),

of length 13√10/5 = 8.221921916437786. Bending at j instead gives 10,
about 21.6% longer. Exact supporting-vector lower bounds in the earlier report
certify both examples. These targets lie in the same sector of the paper's
specified rays, so selecting the other sector cannot repair the rule.

This is a structural error on an open family, not numerical noise. The
polygons have positive area and proper edge intersections, without shared
edges, triple intersections, or perpendicular supporting lines. The
simultaneous-visit certificate has strict inequalities, and the reflected
solution has contacts inside its edges and a strict objective gap. Small
perturbations preserve the relevant behavior. This refutes the literal
intersection construction; it does not refute the disjoint result, prove
that the stated time bound is unattainable, or automatically invalidate all
applications of the paper.

### Wrong1 is a separate reproduced bug

The saved Wrong1 fixture had mixed winding. The old native entry point returned
1.6920315797064933, whereas the independently certified optimum is
1.6729790817113108. Its first polygon is clockwise, and the old core's
orientation-dependent cone tests require counterclockwise boundaries. General
binary entry points now normalize winding when necessary. Both native modes
and the rebuilt WASM return the certified value with all three ordered visits.

The originally reported *missing* visit in Wrong1 could not be reproduced in
the archived fixture/native/WASM combination; the historical audit records
this limitation. It would be incorrect to claim that particular symptom has a
proved root cause. A separate integer-coordinate three-rectangle fixture did
reproduce a missing ordered visit: the old map incorrectly classified an
unfolded query as crossing and returned s→(0,3)→t. The new map returns the
certified route through (0,2) and (1/2,3), length 2√13+4√5.

## Specification and query semantics

Let F_i(q) be the minimum length of a path from s that visits closed convex
P1,…,Pi in order and ends at q. Equivalently,

    F_0(q) = |q-s|,
    F_i(q) = min over x in Pi of [F_(i-1)(x) + |q-x|].

A visit is a nondecreasing parameter along the complete continuous polyline.
Several visits may have the same parameter. An unstored contact in the middle
of a segment is as valid as a stored vertex. First contact means first contact
*after the earlier visits have been satisfied*, not first intersection of an
arbitrary segment with the polygon. These distinctions matter when polygons
overlap. Last contact and first contact may be different points of the same
segment; neither must be emitted as a polyline vertex for a crossing visit.

Each last-step map answers one prefix problem. Its regions have three actions:

| Region | Recursive action |
| --- | --- |
| Crossing, including the whole closed polygon | Query the preceding map at the same endpoint. |
| Bend at a boundary vertex or pseudo-vertex v | Query the preceding map at v; append the segment v→q. |
| Reflection on a pseudo-edge | Reflect q in the edge's line, query the preceding map, then refold the last segment. |

For q in Pi, F_i(q)=F_(i-1)(q): ending at q already satisfies visit i,
and the preceding problem is a relaxation. A crossing exterior cell similarly
inherits the preceding path, whose already-ordered final ray visits Pi.
Adjacent crossing fragments share the same action; their union is represented
implicitly by that action. They are not physically merged into a larger record.

## Implementation

The new implementation is `packages/convex-tpp/cpp/src/solvers/intersecting_maps.cpp`.

1. **Normalize and split boundaries.** Normalize to counterclockwise order;
   discard consecutive duplicate coordinates and a repeated closing point.
   Split each edge at every intersection with an earlier polygon boundary.
   Collinear overlaps contribute their endpoints. A source lying on an edge is
   also a split event. Retain original-edge identity and the exact edge
   parameter. Original-coordinate bounding boxes reject strictly separated
   edge pairs before expensive arithmetic. They introduce no tolerance.
2. **Query each incident limit separately.** At v, query the preceding prefix
   along each incident boundary direction d. A query represents
   q+εd+ε²e1+ε³e2, with ε positive infinitesimal. An orientation test uses its
   first nonzero coefficient, without substituting a small floating-point ε.
   The secondary directions resolve collinearity with a backwards ray
   extension. Closed polygon membership uses q and d, retaining exact boundary
   inclusion; secondary directions break ties only in exterior point location.
3. **Carry a virtual source.** Within a fixed sequence of map cells the final
   direction is proportional to q−V. A crossing inherits V, a bend sets V to
   its vertex, and a reflection reflects V. If q=V, the incident direction d
   gives the nonvanishing leading direction. Reflections transform the point
   and every symbolic direction. This computes limits through vanishing
   segments without reconstructing them numerically.
4. **Construct outgoing rays.** For each incident side, reflect its incoming
   direction if it enters the current counterclockwise polygon across that
   side; otherwise continue it. The two resulting rays bound the vertex fan.
   This is the incident-limit rule in Dror §4.1, Claim 1 (printed p. 477).
5. **Locate by binary search.** Use the existing circular chord/edge predicate
   with those corrected rays and exact symbolic signs. A separate convex fan
   tests closed membership in logarithmic time, omitting redundant collinear
   corners. A failed final cell-containment check raises an error. There is
   no scan fallback or alternative solver.
6. **Memoize and report.** Compute each vertex's two rays once. Cache a last
   directional query per level, with its complete symbolic query as key.
   Cache fixed-vertex prefix lengths for length queries. Refolding checks
   that both segment and edge parameters lie in [0,1], exposing any operation
   that would remove preceding visits. Compact only exactly collinear forward
   contacts, preserving the continuous path.

Construction and map predicates use Boost arbitrary-precision rationals.
Input doubles are interpreted as their exact binary values; constructed
coordinates remain rational until output. Euclidean norms and returned
coordinates are rounded to floating point. No production optimizer, sampling,
Gurobi call, contact candidate enumeration, post-hoc route repair, or
validation-defined fallback remains in this intersection path.

The binary general APIs, including length and workspace overloads, dispatch to
these maps for intersections. Legacy linear and Tan–Jiang general APIs use the
same corrected intersection implementation; they are not independent oracles.
The established disjoint classes and their binary-search/memoization algorithm
remain in place. General binary APIs copy/reverse only when winding needs
normalization. Explicitly named disjoint APIs retain their existing contract.

Dynamic/static workspace overloads remain usable and are tested while buffers
are reused across varying sizes. The old floating-point cone workspace cannot
store exact split maps: those currently own their allocations per solve call.
There is no persistent, externally reusable preprocessed exact map API yet.

### Platform fix

The WASM smoke test exposed a 32-bit bug in the new locator: a conditional
expression combining `size_t` and `-1` converted the crossing sentinel to
4294967295 before returning it as `long long`. That produced an invalid vertex
index. Separate signed returns fix it. WASM AddressSanitizer reproduced the
bad access before the fix and runs clean afterward. Exact arithmetic was not
the cause. Emscripten uses double conversion for reported norms because Boost
does not support its long-double conversion. The build now locates the already
required Boost headers, includes the new source, and uses a 4 MiB checked stack.
The browser cache version was advanced after the successful production rebuild.

## Mathematical justification and its limits

The objective in contact coordinates is convex on the product of convex
polygons. Hence a feasible path with a supporting-vector certificate is
**globally** optimal; the exact certificates for the report examples do not
rely on the map implementation. Infimal convolution also makes every F_i
convex and 1-Lipschitz in its endpoint. Reflection on a correctly classified
edge is an isometry, giving the usual unfolded recursion. Bending and crossing
follow directly from their certified last-contact structure.

For ordinary cells, composing these affine reflections proves the virtual-
source formula by induction. Evaluating the first nonzero coefficient of a
linear predicate proves the one-sided sign rule. At the counterexample's
intersection this explicitly recovers the two different incident limits.
These local facts are established; testing alone is not being used to justify
them.

Dror §4.1, Claim 1 uses incident limiting rays, and Lemma 7 establishes the
starburst property: outgoing rays fill the plane without crossing. Section
4.2 constructs potential vertices from original vertices and intersections
with earlier boundaries and gives recursive point location/query algorithms.
Thus our counterexample is compatible with Dror's construction and does not
refute it. Their §4.2 instruction to query a vertex must be interpreted together
with Claim 1's incident limits; mechanically using one endpoint segment would
repeat the mistake.

**Proof and implementation review, 4 November 2026.** The reviewed proof in
[intersecting-tpp-proof.md](intersecting-tpp-proof.md) addresses the four
previously open obligations: complete split events and constant contact type;
coverage and circular order; the chord predicate and closed symbolic ties;
and ordered refolding. Its key ingredients are the unit-subgradient ray lemma,
the boundary normal-cone update, a general vertex-fan characterization, and a
terminal-suffix invariant for the stored path. A joint induction in polygon
level handles closed membership separately from secondary symbolic ties.

The proof's implementation scope is exact rational arithmetic, fixed start and
target, fixed order and closed positive-area convex polygons, including
shared boundaries, repeated polygons and redundant collinear vertices. It
covers first-contact extraction, as used by the article's anchored solver;
it does not establish the separate point/segment input branches,
`use_last_contact=true`, or unchecked floating-point variants. The proposed
proof therefore supplies the missing anchored-solver argument for the
article's input class without extending every public implementation contract.
The proof was reviewed against the exact implementation and integrated as
Appendix A and Theorem 5.1 of the manuscript on the same date. Theorem 6.1 now
uses this result in place of the anchored-solver assumption. Additional
terminal-suffix and symbolic-direction tests are recorded in the proof note;
the arithmetic complexity bounds and experimental performance claims are
unchanged.

**Update, 8 October 2026.** The certified hybrid oracle no longer trusts this
construction: every bound it returns is proved on the original polygons, by
the interval primal-dual certificate or by the exact KKT certificate. This now
includes the final complete rational solve. Its exact contacts pass the KKT
certificate; when that fails, the oracle returns only the path's length and
its own dual bound, which remain valid. At that time the open map obligations
affected whether the oracle closed its gap and how fast it did so, not the
correctness of its bounds; the November proof above addresses those
obligations for exact positive-area inputs
([certified-convex-oracle.md](certified-convex-oracle.md)).

## Complexity and performance

Write N for the total original vertices, k for the number of polygons, and M
for the total split boundary vertices. Convex boundary intersection counts give
M=O(kN), including overlap endpoints after duplicates are collapsed.

In an arithmetic-operation model, the splitter tests only the pairs of edges
whose ranges in the lexicographic order of (x, y) overlap, merging the two
monotone chains of each boundary
([convex_contacts.h](../../packages/convex-tpp/cpp/include/tpp/convex/detail/convex_contacts.h)).
That is O(|P_i|+|P_h|) pairs for each pair of polygons, O(kN) in all, and every
pair of edges with a common point is among them. With the sort of each edge's
contacts it costs O(kN+M log M); before 2026-11-17 it tested all pairs, in
O(N²+M log M). The pairwise-disjointness checks of the entry points (binary64
and exact) use the same pairs, in O(kN). Once the necessary rays are built, a
query visits at most k maps, each with logarithmic membership/location cost:
O(k log M). Building all M ray pairs, amortizing the lazy construction of each
pair once, costs O(kM log M). Including splitting, the resulting bound is
O(kM log M), or O(k²N log N) since N≥3k, with O(M+k) stored geometric records
and an O(k)-vertex output path. Lazy calls may build fewer rays; a first query
includes whatever construction it triggers.

These are **not bit-complexity bounds**. Rational numerators/denominators grow
under intersection and reflection, so arithmetic is not constant time and
memory depends on their bit lengths. The memoized maps do not achieve Tan and
Jiang's O(k²N) construction bound. The implementation of their construction
below reaches it if their Lemma 1 holds in the split maps. Reusable exact workspaces could
reduce allocation; that is not necessary to define the current result.

Tan and Jiang's construction for disjoint polygons is implemented in
[tan_jiang.cpp](../../packages/convex-tpp/cpp/src/solvers/tan_jiang.cpp)
and reaches their O(kN) bound. Following their Lemma 2, it locates the vertices
of P_{i+1} together in the maps of P_i, ..., P_1, by scans in the circular
order of the cells, and builds every cone, O(N) stored. The query to the target
is a binary search in each map, O(k log(N/k)), and the disjointness check of
the entry point is O(kN). Two guards are not in their paper. After four rounds
of forward and backward scans, the remaining points of a batch are located by
binary search, and so is a point that no scan finds. Neither guard is reached
in exact arithmetic. By their Lemma 1, the cells of consecutive points follow
the circular order, which two rounds cover. A point located in the map of P_h
is not inside P_h. It is a vertex of a later, disjoint polygon, or an image
whose unfolded segment meets P_h before an edge of a later polygon, and
convexity would put that edge point in P_h, which disjointness excludes. The
cells cover the plane outside P_h. Gaps between binary64 cells could reach the guards, at O(|P_h|) per point.
An instrumented copy of the scans (2026-11-18, not kept), on 36 disjoint
instances of three families with k up to 128 and up to 524,288 vertices, found
no point on either guard. Its scan steps stayed within 2.8 times the sum of
their Lemma 2. In the family with the deepest chains, a corridor of thin
ellipses, with k=8 and m from 256 to 65,536 vertices per polygon, the time per
kN stayed between 10.5 and 12.8 ns, against 17.9 to 40.4 ns for the eager
binary-search construction.

For intersecting inputs, the same entry point builds the split maps above by
Tan and Jiang's construction (`build_by_scans` in
[intersecting_maps.cpp](../../packages/convex-tpp/cpp/src/solvers/intersecting_maps.cpp),
since 2026-11-18). It builds every cone, map by map. The two sides of each
vertex of P_i, in the order of the boundary, are located together in the map of
P_{i-1}. The queries that continue (crossings and reflections) are located
together in the map of P_{i-2}, and so on. Each query carries an anchor, the
open pseudo-edge that holds the boundary point it stands for. For a side of a
vertex it is the vertex's own pseudo-edge. For a reflected query it is the
pseudo-edge of its last reflection. Boundaries are split at all their contacts,
so an open pseudo-edge lies inside, outside or on the boundary of each earlier
polygon. Its midpoint decides which, and the midpoints of one boundary are
classified against one earlier polygon in O(|P_g|+|P_h|) time, by a pointer
over the wedges around an inner point of P_h.

- **Anchor inside P_h.** The query is a crossing, without a membership test.
  For a side of a vertex this is exact. For a reflected query, the levels
  between were crossings, so the reflection contact lies on the last segment
  of the unfolded path. A bend on P_h would leave the boundary of P_h toward
  the outside, never through an inner point. This step replaces the membership
  test that Tan and Jiang's §4 argues only for the vertices of P_{i+1}. It is
  taken only when the reflected query was off both limiting rays of its
  reflection cell, so that the contact lies inside the open pseudo-edge.
  Otherwise the contact may sit at an end of the pseudo-edge, on the boundary
  of P_h, and the query goes to the binary search.
- **Anchor outside P_h.** A scan looks for the cell, starting at the cell of
  the previous query and widening in both directions. A found cell is kept
  only if the query is strictly outside an edge line at the cell. That makes
  `inside` reject the query, so the cell is the one the binary search finds.
- **Other cases.** The first query of each batch, the other anchors and failed
  scans use the binary search. A budget of 4(cells+queries)+16 cell tests per
  batch bounds the scans.

The directional tests locate every query by binary search as well
(`TanJiangScanCounters::verify`), and compare the paths. The proofs that the
locations agree, and the cost analysis against Tan and Jiang's theorems, are in
[tan-jiang-implementation.md](tan-jiang-implementation.md).

If the consecutive queries of a batch meet the cells in circular order, with a
bounded number of reversals, a batch costs O(cells+queries). This is Tan and
Jiang's Lemma 1, on which their own bounds rest. It is proved in
tan-jiang-implementation.md for the batches in the previous map. For earlier
maps it is assumed, as in their paper, and it held in every measured batch.
With it, the construction
costs O(Σ_i Σ_{h<i}(M_h+M_i)) = O(kM) = O(k²N). The splitting adds O(kN + M log
k), and the statuses of the anchors add O(kM), so the total is O(k²N) in
arithmetic operations. Without it, the guards keep the result, at the
O(kM log M) bound of the memoized maps.

Measurements on this Mac, 2026-11-18, with instrumented copies of the
constructions that were not kept:

- **Families.**
  - The intersecting path family of paper 1 has k from 4 to 64, m from 8 to
    4096 and 3 seeds, 60 instances.
  - Its fine-grid version has k in {4,16,64} and m up to 32,768, with 2
    million vertices at most.
  - The line family of the scaling experiment is Dror et al.'s worst case.
  - Rosettes are k regular m-gons around one center, slightly rotated and
    scaled, so every pair crosses 2m times (k up to 16, m up to 1024).
  - Exact arithmetic ran on instances of up to 50,000 vertices.
- **Scan steps.** In exact arithmetic the scans tested at most 0.92 times the
  sum of cells plus queries. No scan failed, no cell was rejected and no
  budget ran out. All 11.7 million locations equaled those of the binary
  search, and the lengths were identical. In binary64 the ratio reached 1.05
  on the paper's family. A few scans failed there, and they fell back to the
  binary search. Per batch, over 24.8 million exact queries of random blobs,
  needles, nested polygons, chains and fans, the worst batch tested 3.23
  times its cells plus queries (tan-jiang-implementation.md, Section 4).
- **Time in binary64.** The table gives the geometric mean of the time of the
  scans over that of each other construction.

  | Family | Scans / memoized | Scans / eager |
  | --- | --- | --- |
  | Paper's family | 0.90 | 0.33 |
  | Fine grid | 0.81 | 0.15 |
  | Line | 0.73 | 0.15 |

  The binary64 maps, memoized ones included, stop at an inconsistency on
  every rosette.
- **Time in exact arithmetic.** Same ratios as in binary64.

  | Family | Scans / memoized | Range | Scans / eager |
  | --- | --- | --- | --- |
  | Paper's family | 0.59 | 0.27 to 2.33 | 0.20 |
  | Fine grid | 1.13 | | 0.15 |
  | Line | 0.95 | | 0.16 |
  | Rosettes | 2.14 | | 0.31 |

  The ratio over the memoized maps falls with k. On the paper's family it is
  1.34 for k=4 and 0.36 for k=64. For k=64 and m=4096, the memoized maps took
  11 to 20 s, the eager construction 30 to 48 s and the scans 3.5 to 5.3 s.
- **Fekete's instances.** These are the anchored paths of the 1638 cycles of
  paper 1, with s=t at a vertex of the first polygon, through the other k-1.
  - On the 900 disjoint ones, the memoized core was the fastest on every path.
    Tan and Jiang's disjoint construction took 2.47 times as long in geometric
    mean, and the eager one 1.34 times.
  - On the 721 intersecting ones, in binary64, the memoized maps were faster
    on all paths but one, and the scans took 1.27 times as long. In exact
    arithmetic they took 1.06 times as long, and were faster on 304 paths.
- **Certified flow.** With the scans proposing at Step 1, all 60 instances of
  the paper's family closed in binary64, with the same bounds as the memoized
  proposal. The total time was 1.10 s, against 1.52 s memoized and 2.50 s
  eager.

Baseline on this Mac, 5,000 calls, seed 1729: the disjoint k=20,m=8 case took
4.122225 μs/call. A corrected run took 4.191475 μs/call (about 1.7% higher);
another run with concurrent verification took 4.552375 μs/call. These are
microbenchmarks, not evidence of a statistically significant regression. The
unchanged disjoint class is still selected. The two-reflection intersection
case took 56.650867 μs/call in the first corrected measurement and 59.932225
μs/call during concurrent tests. Its old 0.750558 μs timing measured the
**incorrect** common-intersection shortcut returning 10, so it is not a valid
speed comparison between correct algorithms. Large exact intersection cases
remain substantially costlier than the mature floating-point disjoint core.

## Validation

The original failing audit had 287 checks and 25 failures with 40 random cases.
All oracle gaps resolved: three random paths were invalid and two feasible
paths were too long. Baseline legacy tests passed 318 cases across five solver
entries, demonstrating why stronger tests were needed.

Current results are recorded here only after their executions finish:

| Check | Result |
| --- | --- |
| Intersection audit, 2,000 seeded random convex cases | 4,207 checks; zero failures, invalid paths, suboptimal paths, or unresolved oracle gaps. |
| Directional/public suite: degeneracies, six continuity families, workspace reuse, 3,000 integer-box cases, 200 affine-convex cases | 16,708 checks; zero failures or unresolved oracle gaps. |
| Native ASan + UBSan, 3,000 integer-box cases plus deterministic/continuity/workspace checks | 15,708 checks; no failures or sanitizer diagnostics. |
| WASM production smoke | All six cases pass: both paper targets, ordered rectangles, backwards ray extension, shared source, Wrong1 raw winding. |
| WASM AddressSanitizer, same six cases | All pass without diagnostics. |

The 3,000 integer-box instances use k=2…7, mixed winding, integer endpoints,
frequent shared edges/vertices, and a repeated polygon every fifth case.
The additional 200 affine-convex instances use k=2…12 and 3…20 vertices, with
mixed winding, independent rotations, and thin aspect-ratio-1/32 polygons.
Other tests cover containment, start/target in interiors or on boundaries,
stationary paths, nonconsecutive intersections, simultaneous visits, exact
reference lengths, and redundant collinear vertices. The audit also tests
translation, rotation, uniform scaling by 10^-5 and 10^5, and winding reversal.

The six continuity families exercise vertex/edge contact, edge/edge contact,
nearly collinear edges, containment at tangency, nonconsecutive intersections,
and the appearance of a three-polygon common intersection. They use both signs
of δ in {10^-3,10^-6,10^-9} and δ=0. They validate the whole continuous route,
length/API consistency, oracle bounds, the family-specific 2|δ| value bound,
and a segment-based Hausdorff upper bound. They do not compare lists of contact
vertices. The Hausdorff computation bounds a source segment by its distances to
a single target segment, then minimizes over target segments; it is a rigorous
upper bound, not just a sample of polyline vertices.

Tolerance policy is unchanged from the audit. Ordered clipping uses
64·epsilon(double)·max_abs_coordinate + 10^-12·extent as a distance tolerance.
Exact-reference and API comparisons use 2·10^-12·max(1,length). The box/affine
oracle must close its primal–dual gap within 2·10^-8·max(1,length), independently
pass ordered feasibility, and bracket the production value within that scale.
An unresolved oracle is counted separately. Continuity uses the analytic
2|δ| bound plus rounding tolerance, and a deliberately stated geometric
convergence bound 20√|δ|+2·10^-9 for the chosen families. This is a check on
these families, not a universal Hausdorff theorem for all problems.

The oracle is the existing certified support-dual/interior-point solver. It can
warm-start with the production geometry, so it is not entirely independent
software. Its lower-bound certificate is independent of last-step-map region
classification. The paper witnesses additionally have explicit analytical
certificates. No oracle is called by the new production maps.

## Reproduction

Run from the repository root:

```bash
cmake -S packages/convex-tpp/cpp -B .build/intersection-directional-tests \
  -DTARGET=main-directional_tests -DCMAKE_BUILD_TYPE=Release
cmake --build .build/intersection-directional-tests -j 4
.build/intersection-directional-tests/tpp-convex --public \
  --random-boxes 3000 --random-convex 200

cmake -S packages/convex-tpp/cpp -B .build/intersection-directional \
  -DTARGET=main-intersection_audit -DCMAKE_BUILD_TYPE=Release
cmake --build .build/intersection-directional -j 4
.build/intersection-directional/tpp-convex --random 2000 --bench 5000

cmake -S packages/convex-tpp/cpp -B .build/intersection-corrected-asan \
  -DTARGET=main-directional_tests -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build .build/intersection-corrected-asan -j 4
.build/intersection-corrected-asan/tpp-convex --public \
  --random-boxes 3000 --random-convex 200

packages/convex-tpp/cpp/run_generated_tests.sh

# Use a canonical cache path matching the installed Emscripten version.
EM_CACHE=/private/tmp/tpp-intersection-emscripten/cache \
  bash apps/benchmark-dashboard/wasm/build.sh
node apps/benchmark-dashboard/wasm/test-intersections.mjs
```

The optional `TPP_TEST_PROGRESS=1` environment variable reports each legacy
case/solver, making the large corpus's progress visible. WASM sanitizer checks
use `EMCC_CFLAGS='-g2 -fsanitize=address'` on the same build script; afterward
rebuild without that flag for the production artifact. Compiled WASM is ignored
by git, as before; source and the build/cache-version changes are tracked.

## Remaining work and publication assessment

There is enough material for a carefully scoped correction note: an explicit
counterexample to a specific published rule, exact optimality certificates,
a robust family of failures, and a repair based on one-sided limits. An
implementation paper becomes stronger with the global proof obligations above,
a sharper complexity analysis, broader performance comparisons, and a public
reproducibility package. Publication and novelty are not established by this
implementation exercise.

A targeted literature search on 13 September 2026 did not find an erratum for
Tan and Jiang's intersection rule. That is not an exhaustive novelty check.
The bound is still cited in the 2026 SoCG paper
[A Branch-And-Bound Algorithm for the Traveling Salesman Problem with Difficult Neighborhoods](https://drops.dagstuhl.de/storage/00lipics/lipics-vol367-socg2026/LIPIcs.SoCG.2026.46/LIPIcs.SoCG.2026.46.pdf),
§3.1. This supports relevance, not a claim that that paper's algorithm fails.
The 2026 ICALP paper
[Touring a Sequence of Orthogonal Polygons](https://doi.org/10.4230/LIPIcs.ICALP.2026.50)
also treats Dror's convex Euclidean result as prior work; its orthogonal metric
results are a different problem. No author has been contacted in this task.

Before making a publication claim, independently review the original figure
and proof interpretation, check subsequent versions/corrections and related
algorithms, obtain expert review of the exact witness, and establish the
corrected algorithm theorem if claiming a general replacement. Dror's original
limiting-ray formulation survives this counterexample; we have not proved an
error in Dror's theorem, and should not present the result as one.

The code contract remains finite, positive-area convex polygons. It is not a
nonconvex or fenced solver. The retained disjoint fast path and dispatch tests
still use floating-point geometry and historical tolerances. Exact predicates
inside the new maps do not make the entire public API exact for arbitrarily
extreme coordinate scales. Output rounding can also matter at exact contacts.
Very large k can exhaust the recursive stack, and rational bit growth can make
large instances expensive. These are practical limits and separate from the
specific paper counterexample.

## Final handoff

At the user's request, investigation stopped after the following final checks:

* The complete generated legacy corpus finished: all 300 cases in all six
  suites passed across all five solver entry points, including the largest
  40-polygon/20,000-vertex cases.
* The final native ASan + UBSan run also finished: **16,708 checks, zero
  failures, zero unresolved oracle comparisons, no sanitizer diagnostics**,
  including the additional 200 affine-convex cases.
* The earlier handwritten regression run passed all 18 cases across all five
  solver entries; the existing intersection target passed its 10 cases and
  memoization check.
* The final production WASM build passes all six smoke cases. The separate
  WASM sanitizer build passed the same cases before restoring production.
* `git diff --check` passed. Changes are in the working tree, not committed.

No known failing test remains. The main remaining research task is the full
representation/degeneracy correctness proof described above. Performance
optimization and exact reusable workspaces are optional future engineering.
Do not rerun the large completed corpus merely to resume this task; start by
reading this report and inspecting the current diff.
