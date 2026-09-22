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

A full publication-level theorem for **this implementation** still needs a
complete equivalence proof between its boundary-only, implicit-crossing
representation and Dror's first-contact tree in every closed degeneracy. In
particular it must establish:

* completeness of the split events and homogeneous contact type of each open
  pseudo-edge, including repeated boundaries and multiple simultaneous visits;
* circular order and completeness of the exterior vertex/edge regions after
  inherited interior branches are represented by recursion;
* validity of the chord predicate for every resulting subdivision and the
  consistency of its secondary symbolic tie rules with closed membership;
* preservation of ordered visits by recursive refolding in all such cells.

The implementation checks cell containment, contact-type consistency away from
tangency, and refolding parameters and has passed targeted degeneracy tests.
Those checks expose failures but are not a replacement for the general proof.
There is currently no known failing case in the completed tests. It would
nevertheless overstate the result to label this an independently proved,
universally correct implementation or to claim literal fidelity to Tan and
Jiang's incorrect local rule.

## Complexity and performance

Write N for the total original vertices, k for the number of polygons, and M
for the total split boundary vertices. Convex boundary intersection counts give
M=O(kN), including overlap endpoints after duplicates are collapsed.

In an arithmetic-operation model, the current all-pairs splitter costs
O(N²+M log M), with inexpensive bounding-box filtering in practice. Once the
necessary rays are built, a query visits at most k maps, each with logarithmic
membership/location cost: O(k log M). Building all M ray pairs, amortizing the
lazy construction of each pair once, costs O(kM log M). Including splitting,
the resulting bound is O(N²+kM log M), or O(N²+k²N log(kN)), with O(M+k) stored
geometric records and an O(k)-vertex output path. Lazy calls may build fewer
rays; a first query includes whatever construction it triggers.

These are **not bit-complexity bounds**. Rational numerators/denominators grow
under intersection and reflection, so arithmetic is not constant time and
memory depends on their bit lengths. The implementation does not achieve or
claim Tan and Jiang's O(k²N) construction bound. A sweep/convex intersection
algorithm could improve splitting, and reusable exact workspaces could reduce
allocation; neither is necessary to define the current result.

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
