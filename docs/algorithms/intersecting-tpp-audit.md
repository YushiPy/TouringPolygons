# Intersecting convex TPP: mathematical audit and unfinished implementation

Status (2026-09-13): the requested faithful production hybrid is **not implemented**.
A counterexample to the literal pseudo-vertex construction in Tan and Jiang §4
is established below. The disjoint binary-search/memoization solver is preserved.
This is the mathematical/specification-obstacle outcome allowed by
`TODO-Intersection.md`, not a claim that an exact hybrid is impossible.

## Specification and invariants (recorded before solver edits)

Let π_i(q) minimize length from s through closed P_1,...,P_i, in order, to q.
A map query must represent this prefix problem, independently of the ultimate
target. The existing `Solution::query(q,i)` returns the preceding geometric point
of its last segment, whereas `query_full` reports the prefix. A cache must include
the visit index: the same geometric point at different indices is a different
problem.

* Dror §3, Lemmas 1–3, original PDF pp. 475–476: T_i is the locus of first
  contact **after satisfying the preceding visits**. For disjoint polygons this
  is a connected boundary chain. Bending cells append a boundary vertex;
  reflection cells unfold the query across an edge, recurse, and refold the
  suffix; pass-through cells recurse on the unchanged point.
* Tan and Jiang §2, pp. 616–617, and §3, pp. 617–620: their crossing contacts use
  the boundary exit, in contrast to Dror's first-contact source. A geometric
  boundary exit is not automatically an ordered visit marker when polygons
  overlap. A valid route has nondecreasing path parameters τ_i with
  γ(τ_i) ∈ P_i. Equality permits consecutive visits at one point; several
  ordered contacts can lie inside one stored segment.
* Dror §4.1, original p. 476: T_i can include parts of T_{i-1} inside P_i,
  together with parts of ∂P_i. It need not be a chain on ∂P_i. Interior and
  closed-boundary endpoint queries can already satisfy visit i.
* Tan and Jiang §4, pp. 620–621: split each boundary at intersections with
  earlier boundaries. Each split record would need its original edge, earlier
  polygon/edge incidence, and ordered boundary position. Pseudo-edges retain
  the original supporting line. Interior crossing cells recurse unchanged.
  Adjacent crossing fragments may merge only when their query semantics agree.
* The disjoint binary locator uses circular boundary order, a pair of rays per
  vertex, and homogeneous first-contact status on an edge. Original edges can
  lose that homogeneity at intersections. Worse, a single arriving segment at a
  pseudo-vertex does not determine both incident limiting rays (counterexample
  below). Adding split coordinates alone does not restore the invariant.
* The recursive prefix objective, unfolding on a correctly classified reflection
  cell, index-sensitive memoization, and binary search on a **proved ordered
  subdivision** remain applicable. No fences are necessary for these facts.

Sources checked: [Tan and Jiang original PDF](../bibliography/TPP-Tan&Jiang/original.pdf),
[Dror et al. original PDF](../bibliography/TPP-Dror/original.pdf),
their `main.pdf` transcriptions, and the
relevant `main.tex` sections under `docs/bibliography/TPP-Tan&Jiang/` and
`docs/bibliography/TPP-Dror/`. The original Tan and Jiang Figure 4 and its text
were visually checked. The problematic rule is present in the original PDF;
it is not a TeX transcription error.

## Counterexample to the literal pseudo-vertex rule

Use these counterclockwise convex polygons:

```
s = (-3,-4)
P1 = [(0,-6), (6,-6), (6,6), (0,6)]
P2 = [(-8,-4), (8,4), (8,9), (-8,9)]
j = (0,0)
```

Their boundaries intersect properly at j and (6,3), each strictly inside two
edges. There are no coincident edges, triple intersections, or degenerate
polygons. The incident supporting lines at j are x=0 and y=x/2.
The shortest partial path to j is the segment s→j, with unit incoming vector
u=(3/5,4/5). It intersects neither polygon interior. Thus the two conditions
in Tan and Jiang §4, original p. 621, paragraph below Figure 4, hold (there are
no intermediate polygons).

Reflection across x=0 gives R1(u)=(-3/5,4/5).
Reflection across y=x/2 has matrix [[3/5,4/5],[4/5,-3/5]], giving R2(u)=(1,0).
The paper says the bending region is bounded by these two single-reflection
rays. Neither angular sector between them is the correct region:

1. With target t_a=(4,-3), s→j→t_a is globally optimal, length 10.
   Set w=(4/5,-3/5) and z=(1/10,4/5), with |z|=√65/10<1.
   For arbitrary visits a∈P1, b∈P2, the supporting-vector norm inequalities give

   |a-s|+|b-a|+|t_a-b|
   ≥ 10 + (u-z)·a + (z-w)·b
   = 10 + (1/2)a_x + (7/5)(b_y-b_x/2) ≥ 10.

   Equality holds at a=b=j. Both multipliers are positive and |z|<1;
   this is a genuine region of simultaneous visits, not an isolated tie.

2. With target t_b=(-4,-3), the globally optimal path is
   s→(0,-3)→(-18/5,-9/5)→t_b, of length (13/5)√10.
   Its three unit directions are U=(3,1)/√10, Z=(-3,1)/√10,
   W=(-1,-3)/√10. For arbitrary feasible a,b, the same inequalities give

   length ≥ (13/5)√10 + (6/√10)a_x
                           + (4/√10)(b_y-b_x/2) ≥ (13/5)√10.

   The displayed path attains equality. The bend-at-j candidate has length 10
   and is strictly suboptimal by approximately 1.77807808356.

Both targets are in the same sector between R1(u) and R2(u) (the sector
containing the lower half-plane). The other sector excludes t_a. Therefore no
choice of sector with the paper's two specified boundaries describes the
pseudo-vertex bending region. The incident edges are not perpendicular, and
s,t_a,t_b are outside the required first/last polygons, respectively.

### The missing directional information

Approach j along the P2 boundary. For ε>0:

* q_+(ε)=(2ε,ε): π_1(q_+) is straight, with limiting last direction u.
* q_-(ε)=(-2ε,-ε): π_1(q_-) reflects on x=0 at
  (0,-11ε/(3+2ε)); its nonzero last direction tends to R1(u).

At ε=0 both geometric paths converge to s→j. The vanishing reflected segment
on the negative side is absent from that unique path. Reflecting both limiting
incoming directions across P2 gives **R2(R1(u))=(7/25,-24/25)** and R2(u)=(1,0).
These are the actual limiting rays of the two incident reflection regions.
The intervening cone contains t_a and excludes t_b. In particular this
pseudo-vertex has a bending region between **two reflection regions**, contrary
to the same paragraph's crossing/reflection-adjacency assertion.

Dror §4.1, original p. 477, Claim 1 immediately before Lemma 7
(`main.tex` lines 302–308; transcription Claim 1, `main.pdf` p. 8), uses
**limiting rays on incident first-contact-tree edges**. That formulation retains
the information lost by the literal Tan and Jiang endpoint recipe.

### What is required to continue

An exact corrected hybrid needs one-sided incident-ray queries or an explicit
first-contact-tree/starburst representation. Query keys may need a direction
and side in addition to (point,index) at coincident/zero-length contacts. It
must establish the complete exterior subdivision, its circular search order,
and correct inherited crossing regions using those limits. Multiple incident
edges, repeated contacts, and collinear overlaps need an extension beyond the
paper's no-three-edges assumption. The resulting construction and storage
bounds must be proved and tested; the cited single-ray rule does not prove them.

This counterexample does **not** refute existence of a polynomial algorithm,
Dror's limiting-ray claim, or all of Tan and Jiang's results. It prevents an
honest claim that a production solver implementing the literal §4 construction
is faithful and globally correct for the requested input contract. No numerical
perturbation, optimizer, or linear-solver fallback has been substituted for the
missing construction.

## Existing intersecting-code audit

| Mechanism | Classification and consequence |
| --- | --- |
| Interior query → preceding map | Prescribed by Tan and Jiang §4; closed-boundary handling must also retain visit semantics. |
| Pairwise-intersection detection/cache | Valid dispatch detail, not a map construction. All-pairs work is outside the mature disjoint locator but already in its public wrapper. |
| `segment_enters_polygon_before_endpoint` | Insufficient order test. `query` returns the last stored bend, which can precede the completion of earlier visits on that same segment. Mere contact somewhere on the segment cannot justify skipping visit i. |
| Original-vertex cones reused on intersecting polygons | Incorrect generally: no pseudo-vertices, no split-edge state, no one-sided limiting rays. |
| `SolutionBinarySearchIntersecting` | Unused by the public binary APIs; also inherits the invalid original-boundary representation. |
| Binary APIs delegating to linear APIs | Actual production behavior on intersections; fails the requested architecture. |
| Direct s→t shortcut, if independently feasible in order | Valid lower-bound equality via triangle inequality. |
| Single-polygon optimum, if feasible for the full problem | Valid relaxation certificate: full optimum ≥ relaxed optimum, and feasibility gives the reverse inequality. |
| Common-intersection shortcut with immediate feasible return | Unsupported/incorrect optimality claim. Requiring two visits at one point restricts the feasible set; it does not relax the original problem. The example above returns 10 instead of (13/5)√10. |
| 20,000-iteration contact optimizer, snapping, 1e-6 merging | Unsupported as an exact production method. The present dispatcher only invokes it with a single contact set, so its multi-set iterative body is unreachable from that dispatcher. |
| Return `SolutionLinearSearchIntersecting(...).solve(...)` after candidate failure | Unvalidated fallback, not the published construction; can expose invalid cone/query decisions. |
| `tan_jiang.cpp` intersecting entry point | Delegates through the binary API to the linear intersecting solver; it is not an independent oracle. |
| Existing test validator | Mixes feasibility with disjoint local bend rules; arbitrary first edge intersections and target-on-edge/vertex early returns can lose visit order or remaining visits. It cannot certify intersecting optimality. |
| Old `expect_touches_all_polygons` | Ignores order completely. |

## Wrong1 and verification results

Results, reproducible commands, modifications, and remaining work are appended
below after running the audit harness.

### Wrong1: distinguish the reported symptom from reproduced results

The exact current campaign fixture was exported without backgrounds or map
metadata to `benchmarks/suites/intersection-audit/Wrong1.bin`. Its first polygon
is clockwise. On this checkout:

| Invocation | Length | Ordered visits |
| --- | ---: | --- |
| Fresh native binary lazy/eager, original mixed winding | 1.6920315797064933 | 3/3 |
| Fresh native binary lazy/eager, normalized CCW | 1.6729790817113108 | 3/3 |
| Existing visualizer WASM, `_tpp_solve`, original input (wrapper normalizes) | 1.6729790817113108 | 3/3 |

The normalized/native/WASM path is:

```
(0.2, 0.1)
(0.72298228800026387, 0.72032948303824984)
(0.86758984783212589, 0.54880487332933281)
(0.94572929388204030, 0.53857018327967254)
(0.60000000000000009, 0.1)
```

For each of its three contacts, the incoming minus outgoing unit vectors
support the corresponding polygon. Summing these supporting inequalities gives
a lower bound of 1.672979081711311; the geometry has length
1.6729790817113108. The approximately 2.2e-16 difference is floating-point
rounding, not a substantive certificate gap. The analytical construction and
certificate are recorded in the fixture README and automated audit.

The raw mixed-winding route replaces the second point by
(0.60400633107575763, 0.72032948303824984). It is feasible but longer by
0.0190524979951825. `Solution::build_cone`, `solution.cpp` lines 153–186, uses
orientation-dependent cross-product signs for first-contact membership. The
public convex API does not normalize winding; the nonconvex/WASM wrapper does.
That is a reproduced winding-contract bug, not evidence of a missed polygon.

**The reported missed visit in Wrong1 could not be reproduced in either the
fresh native build or the saved WASM.** No polygon/contact is claimed missing
from those paths. Its precise reported root cause therefore remains unresolved;
a different earlier fixture or deployed binary would be needed to attribute
that symptom. No browser rebuild or cache change was made.

### A reproduced invalid path, with integer coordinates

The audit also permanently constructs:

```
s=(-4,-4), t=(4,-4)
P1=[(0,2),(2,2),(2,4),(0,4)]
P2=[(0,-2),(3,-2),(3,2),(0,2)]
P3=[(-1,3),(1,3),(1,4),(-1,4)]
```

Both binary modes return `s→(0,3)→t`, length `2√65 ≈ 16.1245154966`.
The incoming segment has x<0 until its endpoint: it first reaches P1 and P3
at (0,3), outside P2. On the outgoing segment, P2 is first reached at
(4/7,2), after leaving P3. P3 cannot then be visited again. The independent
validator therefore reports only **2 of 3 visits in order**, despite every
polygon touching the geometric path somewhere.

A certified optimum is `s→(0,2)→(1/2,3)→t`, length
`2√13+4√5 ≈ 16.1553744609`. Visits 1 and 2 occur simultaneously at (0,2).
A norm-inequality certificate uses unit directions
u0=(2,3)/√13, u2=(1,2)/√5, u3=(1,-2)/√5 and, for the zero-length step,
u1=(1/√5,3/√13), whose norm is less than one. The successive differences
support P1's left edge, P2's top edge, and P3's bottom edge. The audit checks
primal-dual equality independently.

Temporary diagnostic instrumentation of an isolated source copy established
this exact recursion (the production source was not instrumented): shortcut
candidates fail, and the code enters `SolutionLinearSearchIntersecting`.
`locate_point((4,-4),3)` returns edge 0 (location 1), so the target unfolds in
P3's bottom edge to (4,10). At level 2, the inherited original-vertex locator
returns -1 for (4,10), even though its level-1 straight path has no P2 contact.
Level 1 also returns -1. Refolding the resulting invalid prefix creates the
incorrect bend (0,3). This is a concrete false pass-through classification,
not a numerical tolerance issue. The unchecked final fallback exposes it.

## Changes made

* `packages/convex-tpp/cpp/include/tests.h`: declares a separate
  `OrderedPathValidation` result and feasibility API.
* `packages/convex-tpp/cpp/src/core/ordered_path_validation.cpp`: implements
  independent long-double half-plane clipping, ordered segment contacts,
  finite-coordinate/endpoint checks, winding-independent closed membership,
  zero-length segments, stationary paths, and geometry length measurement.
* `packages/convex-tpp/cpp/src/core/tests.cpp`: runs ordered feasibility before
  the existing local-bend checks. No previous local checks were relaxed.
* `packages/convex-tpp/cpp/src/main-intersection_tests.cpp`: replaces the
  order-insensitive touching helper with complete ordered-path validation.
* `packages/convex-tpp/cpp/src/main-intersection_audit.cpp`: exact certificates,
  the pseudo-vertex counterexample, an invalid ordered path, Wrong1, six
  metamorphic variants, a seven-parameter edge-contact continuity family,
  optional seeded random/oracle checks, and reproducible microbenchmarks.
* `benchmarks/suites/intersection-audit/Wrong1.bin` and its `README.md`: isolated
  permanent fixture, independently certified expected route, and run commands.
* `packages/convex-tpp/cpp/tests/README.md`: documents the stronger validator and
  explicitly failing production audit.
* This report records the mathematical obstruction and outstanding work.

`binary_search.cpp`, `linear_search.cpp`, `tan_jiang.cpp`, `solution.cpp`, the
workspace implementation, and all public solver dispatches are unchanged. No
heuristic/fallback was removed or introduced; the incorrect intersecting
production behavior remains exposed by failing regression tests. The user's
untracked `TODO-Intersection.md` was left unchanged.

## Verification and tolerance policy

Toolchain: AppleClang 21.0.0.21000334, macOS, CMake Release (`-O3`), existing
project targets and OpenMP configuration. The initial
`run_generated_tests.sh` invocation found stale caches referring to a different
home directory (`/Users/gabrielushijima`). Rather than overwrite them, the same
CMake targets were built in fresh `.build/intersection-*` directories.

| Check | Result |
| --- | --- |
| Baseline existing intersection target | All 10 cases plus memoization check passed. |
| Baseline handwritten corpus, all five solver entries | 18 cases passed. |
| Baseline generated corpus, all five solver entries | 300 cases across six suites passed. |
| Both corpora after the ordered validator change | All 318 cases passed, all five entries. |
| Existing intersection target after replacing unordered helper | Passed. |
| Audit `--proof-only` | 99 checks passed; no production correctness claim. |
| Full audit `--random 40` | 287 checks, 25 failures; exit code 1, intentionally visible. |
| Random seed 20260913 | Trials 5, 6, 28 invalid; 15 and 18 feasible but suboptimal; 0 unresolved oracle gaps. |
| Exact two-reflection counterexample | Both binary modes return 10 instead of (13/5)√10. |
| Metamorphic variants of that counterexample | Same optimal-length failure after translation, rotation, scaling by 1e-5 and 1e5, winding reversal, and inserted collinear vertices. |
| Edge-edge continuity | Seven δ values: +1e-3,+1e-6,+1e-9,0,-1e-9,-1e-6,-1e-3. Exact certificates, ordered feasibility, geometry/API lengths, and optimal-value comparisons passed. |
| ASan + UBSan, handwritten/generated corpora | All 318 cases passed across all five entries, no sanitizer diagnostics. |
| ASan + UBSan, proof-only and full audit | Proof checks pass; full audit reproduces the same 25 algorithmic failures with no sanitizer diagnostics. |

The continuity family uses s=(-3,-2), t=(3,-2),
P1=[-2,0]×[0,2], P2=[δ,2]×[0,2]. For δ>0 its optimum is
`s→(0,0)→(δ,0)→t`; for δ≤0 it is `s→(0,0)→t`.
Support-vector certificates handle the zero-length contact between visits.
The optimal length is continuous with a bound of 2|Δδ| for this family.
It exercises both sides of the actual public dispatch boundary, but is **not**
the full set of continuity families requested in the task.

The independent validator uses a distance tolerance
`64 * epsilon(double) * max_abs_input_coordinate + 1e-12 * input_extent_from_start`.
Each half-plane expands by that distance times the edge length; clipping uses
long-double arithmetic and exact zero-slope branching. No dimensionless fixed
parameter epsilon is used to reorder contacts. Certificates are checked within
`2e-12 * reference_length`; nominal unit dual vectors are contracted by about
1e-14 for floating-point norm safety. Production length comparisons use
`2e-12 * reference_length`. The random test oracle must close its reported
primal-dual gap below 2e-8 and independently pass ordered feasibility; otherwise
it is counted as unresolved. None of these tolerances were increased to accept
a failing solver output. In particular the errors of size 1.778 and the missed
ordered rectangle visit are many orders above tolerance.

The random oracle is the existing, separately implemented support-dual and
interior-point `tpp_convex_solve_certified` API. It is only called from the audit
executable. It initially tries the geometric solver, so it is not wholly
independent software, but its dual certificate is independent of the last-step
map classification. The exact deterministic certificates do not depend on it.

### Reproduction commands

Run from the repository root; generated files remain under `.build/`:

```bash
cmake -S packages/convex-tpp/cpp -B .build/intersection-baseline \
  -DTARGET=main-intersection_tests -DCMAKE_BUILD_TYPE=Release
cmake --build .build/intersection-baseline -j 4
.build/intersection-baseline/tpp-convex

cmake -S packages/convex-tpp/cpp -B .build/intersection-generate-baseline \
  -DTARGET=main-generate_tests -DCMAKE_BUILD_TYPE=Release
cmake --build .build/intersection-generate-baseline -j 4
.build/intersection-generate-baseline/tpp-convex .build/intersection-generated-tests
cmake -S packages/convex-tpp/cpp -B .build/intersection-verify-baseline \
  -DTARGET=main-verify_solutions -DCMAKE_BUILD_TYPE=Release
cmake --build .build/intersection-verify-baseline -j 4
TPP_TEST_DIR=packages/convex-tpp/cpp/tests .build/intersection-verify-baseline/tpp-convex
TPP_TEST_DIR=.build/intersection-generated-tests .build/intersection-verify-baseline/tpp-convex

cmake -S packages/convex-tpp/cpp -B .build/intersection-audit \
  -DTARGET=main-intersection_audit -DCMAKE_BUILD_TYPE=Release
cmake --build .build/intersection-audit -j 4
.build/intersection-audit/tpp-convex --proof-only
# This next command currently exits 1. Do not suppress the known failures.
.build/intersection-audit/tpp-convex --random 40 --bench 5000

cmake -S packages/convex-tpp/cpp -B .build/intersection-audit-asan \
  -DTARGET=main-intersection_audit -DCMAKE_BUILD_TYPE=Debug \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
  '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined'
cmake --build .build/intersection-audit-asan -j 4
.build/intersection-audit-asan/tpp-convex --proof-only
.build/intersection-audit-asan/tpp-convex --random 40
```

For the sanitized handwritten/generated corpus, use the same sanitizer flags
with `TARGET=main-verify_solutions` and build directory
`.build/intersection-verify-asan`, then run the two `TPP_TEST_DIR` invocations.

## Performance measurements

The exact public binary API and solver sources are unchanged. A small harness
linked against the untouched baseline library from
`.build/intersection-generate-baseline` and then against the audit build used
identical seeded input, 50,000 calls per sample, five samples per library.
Median wall-clock microseconds per call were:

| Instance | Before validation changes | After validation changes |
| --- | ---: | ---: |
| Disjoint, k=20, m=8, seed 1729 | 4.60691 | 4.54345 |
| Intersecting two-reflection counterexample (incorrect answer) | 0.849379 | 0.845170 |

Checksums agreed. These small timing differences are measurement variation;
there was no solver optimization. The maintained audit's `--bench 5000` also
measured 4.34365 μs and 0.82370 μs respectively in a separate run. Its same
seeded inputs can reproduce the workload. A correct intersection implementation
has not been benchmarked, and no performance bound is inferred from timings of
an incorrect answer.

## Complexity and remaining implementation work

No corrected hybrid complexity has been achieved. The preserved disjoint core
has the paper's construction bound O(kn log(n/k)) with O(n) map storage;
its existing linear-locator escape path should be distinguished from an
unconditional worst-case binary locator claim. The public dispatch additionally
performs pairwise intersection tests: its naive edge-pair worst case is O(n²),
with bounding-box rejection helping the practical disjoint workload. These
pre-existing costs were not changed.

Tan and Jiang claim O(k²n) construction, O(kn) storage, and O(kn) reporting
(Theorem 2, original p. 622). A corrected endpoint/directional binary-query
hybrid might target O(k²n log(kn)) construction and O(k log(kn)) per reported
query, but those are **conditional design targets**, not established bounds
for an implementation here. Proving bounds requires counting the actual split
records, incident limits, inherited regions, and recursion/caching operations.

Outstanding work, in dependency order:

1. Resolve the §4 pseudo-vertex rule against Dror Claim 1. Formalize one-sided
   limiting rays and their query semantics, including zero-length visits.
2. Implement and validate pseudo-vertex discovery, incidence records, boundary
   splitting, crossing-region merging, pseudo-vertex bends, and per-side
   pseudo-edge reflections with the corrected semantics.
3. Prove the ordering needed by the existing binary point locator; adapt its
   predicates where necessary. Preserve index-sensitive memoization and grow
   workspace capacity for split geometry without repeated all-pairs recursion.
4. Extend the construction to closed-boundary queries, containment, repeated
   polygons, coincident edges, multiple incident edges, and CW/CCW input.
5. Route intersecting inputs through that binary implementation; remove the
   unsupported restricted-intersection shortcut/unchecked fallback and isolate
   the unreachable iterative optimizer. No replacement solver is authorized
   as the production solution by this task.
6. Make the new production audit pass, retaining exact certificates and strict
   ordered validation. Resolve all random invalid and suboptimal cases.
7. Complete deterministic solver coverage for every requested degeneracy.
   Current validator tests cover several of them, but validator coverage is
   not proof that the solver handles those cases.
8. Extend oracle/random checks beyond 40 cases, including more general convex
   shapes and the full requested metamorphic suite on correct returned paths.
9. Add vertex-edge, nearly collinear, containment-tangency, nonconsecutive
   intersection, and multi-polygon common-intersection continuity families.
   Add representation-independent geometric path-distance comparisons and
   handle nonunique optima explicitly. Only the edge-edge family is complete.
10. Establish complexity, benchmark a **correct** intersection implementation,
    validate workspace and public overload behavior, and repeat the relevant
    sanitizers/regression checks after solver changes.
11. Reproduce the reported Wrong1 missed-visit symptom if its earlier input or
    deployed binary becomes available; do not replace that investigation with
    the separately reproduced winding and invalid-order bugs.

**Faithfulness statement:** no faithful Tan-and-Jiang/Dror intersection hybrid
is delivered. The allowed mathematical-obstacle outcome is supported by an
explicit nondegenerate counterexample and independent global-optimality proofs.
The useful completed implementation is stronger test validation and a durable
failing audit, with the mature solver architecture and existing behavior
preserved pending a corrected construction.
