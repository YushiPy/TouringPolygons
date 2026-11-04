# Chazelle and Dobkin, Section 4: implementation notes and status

Companion to [`steiner-convex-partition.md`](steiner-convex-partition.md), which
covers the problem, the theory and the Section 3 (polynomial) algorithm. This
document is the record of the Section 4 work (`Options::algorithm =
Algorithm::cubic`): what is implemented, how each part maps to the text, where
and why it departs from it, the evidence, and what remains to reach the
text's O(n + c³), with a design for each open item. It is written so the work
can be resumed without the conversation that produced it. Nothing here has
been published.

## Sources

- B. Chazelle, D. P. Dobkin, *Optimal convex decompositions*, in
  *Computational Geometry*, G. T. Toussaint (ed.), North-Holland, 1985,
  pp. 63–133. Section 4 ("The cubic algorithm") is the reference. Page and
  lemma numbers below are from this text.
- B. Chazelle, *Computational geometry and convexity*, PhD thesis, Yale, 1980
  (CMU-CS-80-150). It holds the proofs the 1985 text omits (Lemma 10, part of
  Lemma 14, Lemmas 18 to 20). **We could not obtain it.** Every construction
  marked "ours" below replaces a proof or procedure that is only there.
- Copies of the 1985 text and the 1979 STOC paper were used during the work;
  they are not in the repository.

The text's figures did not survive the text extraction we worked from, so
statements that depend on a figure (orientations of angles, which side of a
segment) were derived again rather than copied.

## Status at a glance

| Part | Text | Code | State |
|---|---|---|---|
| Dynamic program (S, B, F, reconstruction) | Sections 3–4 | `dynamic_program.cpp` | shared with the polynomial version |
| Convex chains, pseudo-notches | Lemma 8 | `chains.*`, `pseudo_notches` | faithful, chains cut further (ours) |
| Superranges | Lemma 10 | `superrange.*` | ours (rotational sweep), O(n log n) per notch instead of O(c log n) |
| Diagonals (X2) | Lemma 13 | `Superranges::sees_vertex` | faithful (first contact) |
| X3 / junction tests | Lemmas 12, 14, 15 | `JunctionSolver` (cubic path) | per-arc half-planes as in the text; arm visibility checked by O(log n) queries instead of the hexagon Q of Lemma 14 |
| Loose X4, rays r_v, points A_ik | Lemmas 16, 17 | `loose_x4.*` | faithful, side conditions derived in our orientation |
| Safety region C, D | Lemma 18 | `LooseX4::compute` (walk) | ours: event walk over chains, O(min(n, c log n)) per set instead of O(c) |
| g1, g2, f, x | Lemmas 19, 20 | `LooseX4::compute` | linear algorithm as in the text, its claims checked at run time; f by binary search |
| E(i, j) in the DP | Lemma 21, STEP 3 | `compute_e`, X4 block of `compute_s` | faithful |
| Naive decomposition | Lemma 9 | `assembly.cpp` (shared) | not the text's bound |

## Conventions (read before the code)

- The polygon is counterclockwise (`prepare_polygon` reverses clockwise input).
  Notches are numbered 0..c−1 counterclockwise; DP intervals [i..j] use
  unwrapped indices i ≤ j ≤ i + c − 1, reduced modulo c.
- The text is clockwise. Under the mirror map, its clockwise order is our
  counterclockwise order, its edge R_i is our *next edge* (from v to v + 1),
  its L_i our *previous edge*, and its counterclockwise angles are our
  clockwise ones. No angle test was transcribed; each was re-derived.
- At a notch v, directions are ordered counterclockwise starting from the
  next edge (`Superranges::compare_directions`: half-plane split at the next
  edge, then orientation). The corner (interior angle, above 180°) runs from
  the next edge to the previous edge.
- Boundary positions seen from notch v: `2 (u − v)` for vertex u and
  `2 (e − v) + 1` for a point inside edge e (from vertex e to e + 1), indices
  modulo n (`Offset`).
- "Left/right of r" means of the directed line through the notch along the
  ray. `r_v` is the extension of v's next edge beyond v, i.e. the line from
  v + 1 through v, directed away from v + 1.
- All predicates are exact. Rational points (ray crossings, hits) are
  homogeneous big-integer points (`exact::HPoint`) with a long double
  approximation and an error bound (`LooseX4::RPoint`); predicates are
  filtered and fall back to big integers only when the filter cannot decide.

## Map from the text's symbols to the code

| Text | Code |
|---|---|
| SR(v), pairs (a_p, b_p) | `Superranges::Range::directions`; `Direction::after` / `before` are the hits of the ray turned slightly counterclockwise / clockwise |
| R(v, D) (ray cut at the chord a_p b_p) | not stored; queries use the exact first contact, which is never nearer than the chord |
| jump of SR(v_i) over v_j | `Superranges::last_direction(a, b)` / `first_direction(a, b)` (tabulated) |
| r_i = R(v_i, R_i) | `LooseX4::hits_` (end of r_v), rays from `after(t)` through `notch(t)` |
| A_ik = r_i ∩ r_k | `LooseX4::Crossing` in `crossings_[i]` with `other = k`, sorted along r_i |
| E(i, j) | `DynamicProgram::e_[i][length]`, entries `LooseX4::Match {outer, inner}` |
| A_1 … A_p, B_1 … B_q | `as`, `bs` in `LooseX4::compute` (sorted from the notch outward) |
| chains C, D | `x_hull` (pieces hanging from X = B_q A_1), `y_hull` (from Y = A_p B_1) |
| g1, g2, f, x | `g1`, `g2`, `f`, the pointer `lb` in `LooseX4::compute` |
| M in STEP 3 | the X4 block of `compute_s` (two topologies, Fig. 30) |

## The parts in detail

### Convex chains (`src/chains.*`)

The maximal convex chains of Lemma 8 (between consecutive notches), each cut
again when its direction has turned by half a turn. A piece turns left or
goes straight at every inner vertex and turns less than 180° overall, so for
any line the signed distances of its vertices are strictly monotone, constant,
strictly monotone (in that order, any part possibly empty). `edges_meeting`
finds, per piece, the O(1) edges meeting a line by three binary searches.
It returns every crossing edge and every edge with an endpoint on the line,
except the inner edges of a run of vertices lying on the line (the run's end
edges are kept). That is enough for the callers: the nearest contact of a ray
is a run end, and a segment whose endpoints are interior points of P cannot
meet a run without meeting one of its end edges. When the pieces are many for
the polygon's size, scanning all edges is cheaper and is used instead (the
cost is O(min(n, c log n)) per query). The line is given as an object with
two exact signs (`side`, `turn`), so lines through rational points work too
(`LooseX4::RLine`).

### Superranges (`src/superrange.*`)

A notch v's superrange lists the directions from v toward the pseudo-notches
strictly inside its corner, plus the two edge directions, in counterclockwise
order. Each direction records its first contact (vertex or edge) and the edges
hit by the ray turned slightly either way. Between two consecutive directions
the visible boundary is part of one convex chain, so `sees(v, s)` is two
binary searches and one orientation test.

**Construction (ours).** Lemma 10 builds superranges in O(c log n) per notch
by rewriting a linear visibility algorithm over convex chains; its proof is in
the thesis. We use one rotational sweep per notch over the n edges: vertices
in the corner sorted by direction, edges crossing the sweep ray in a balanced
tree ordered by distance. Two edges crossed by a common ray are ordered by the
side of each other's line they lie on (`before`), a total order because edges
do not cross. Each edge is active over the angular interval from its endpoint
s to its endpoint t counterclockwise, clipped to the corner; an edge passing
behind the notch can be active over two pieces of the corner. At each target
direction, the nearest edge before and after the vertices at that angle give
the turned hits, and the nearest of those vertices and of the edge crossing
through gives the contact. If the tree ever refused an insertion (a tie it
cannot order, which should not happen), the notch falls back to one direct
trace per direction (`trace`, O(n) each, also used by the cross-check).

**Tables.** `last_direction(a, b)` is the target of the direction of SR(a) at
the jump over b's position on the boundary (−1 if a sees b); `first_direction`
is the mirror from b. They are computed for all notch pairs at construction,
so the junction constraints are table lookups.

### Junction tests in the cubic version (`JunctionSolver`, cubic path)

A Steiner point s joining notches a, b, c (counterclockwise) lies in the
triangle abc and satisfies the role half-planes at each notch (subtree arms,
angles at most 180°), exactly as in the polynomial version. Instead of
scanning the three boundary arcs, the cubic version adds two strict half-planes
per arc from the tables: s strictly counterclockwise of the ray from a toward
`last_direction(a, b)`, and strictly clockwise of the ray from b toward
`first_direction(a, b)`. These are the edges through the notches of the
text's hexagon Q (Lemma 14). The same exact search then picks a candidate,
and its three arms are checked with `sees` (O(log n) each).

This is where we stop short of Lemma 14: the text also cuts each of those rays
at its visible length (r_ij, l_ji), requires the corners C, A, B of Q to lie
within both cut segments, and then needs no visibility query at all. See
[open item 1](#1-constant-time-junction-tests-lemma-14).

Why the half-planes alone do not suffice: with them, the first boundary point
on the arm from a can only belong to the opposite arc (b to c), but an arc can
still leave the triangle through a far side. The polynomial version detects
that explicitly when it scans an arc (`scan_arc`); the text detects it through
the corners of Q; the cubic version currently detects it through `sees`.

### Loose X4-patterns and E(i, j) (`src/loose_x4.*`)

**Loose configuration (Lemma 16).** An X4-pattern that is not reducible to
Y-patterns can be reduced to a loose one, whose 16 extreme configurations are
all valid. We use the one where every arm runs along its notch's r_v. So the
Steiner point joining k and i is A = r_k ∩ r_i, a rational point determined by
the two notches.

**Usable crossings (ours, derived).** For notches t then s (counterclockwise
in the pattern), r_t ∩ r_s serves as their Steiner point iff v_t is strictly
left of r_s, v_s is strictly right of r_t, (v_s − v_{s+1}) × (v_t − v_{t+1}) < 0
(both rays reach the point going forward), and the point lies strictly before
the ends of both rays. All are exact double predicates except the last two,
which compare rational points. Computed once for all pairs and sorted along
each ray: O(c² log c).

**The set E(i, j).** Computed right after S(i, j). For each notch k outside
[i..j] it holds at most one inner notch l such that (k, i, l, j), with
A = r_k ∩ r_i and B = r_l ∩ r_j, is a valid pattern. Inner notches obey
relation (2), S(i+1, l−1) + S(l+1, j−1) = S(i, j); other choices of l can never
beat a split of the interval, so nothing is lost.

**Conditions (ours, derived).** For the pattern (k, i, l, j):

- A in the closed triangle (v_k, v_i, B) iff B is weakly right of r_i (a filter
  on the B's, fixed for the interval) and weakly left of r_k (an interval of
  the B's, "alpha").
- B in the closed triangle (v_l, v_j, A) iff A is weakly right of r_j (a fixed
  filter on the A's) and weakly left of r_l (an interval of the A's, "beta").
- The arms lie on r_v before its end, so they are clear; A and B are interior.
- The middle edge AB must be clear (the safety region, below), and the arms of
  A must not meet the arms of B (checked on the chosen pair).

**Safety region (Lemma 18; our construction).** Every middle edge lies in
the quadrilateral of the extreme candidates A_1 A_p B_1 B_q (convex; its
orientation σ is checked, and a degenerate one falls back to testing the
pairs one by one). The boundary never touches its sides on r_i and r_j, so the
boundary inside it consists of pieces entering and leaving through the free
sides X = B_q A_1 and Y = A_p B_1. A piece touching both sides cuts every
middle edge (the set is empty). Otherwise a middle edge AB is clear iff the
vertices of the X-pieces lie strictly on X's side of AB and those of the
Y-pieces strictly on Y's side; only notches can be extreme toward the middle,
and every notch is a chain end. The pieces are found by a walk, from v_i, over
two kinds of events: edges meeting X or Y (from two chain line queries) and
chain ends inside the quadrilateral. Between events the boundary stays on one
side of the quadrilateral. Each side's obstacles are stored as a convex hull
split into its lower and upper chains (these are the text's chains C and D),
so testing AB against them costs O(log h).

The text computes C and D in O(c) by wrapping around the obstacles with the
function NEXT over superranges (pp. 119–125); we did not reproduce it, see
[open item 2](#2-the-safety-region-in-oc-lemma-18).

**Matching (Lemmas 19, 20).** For a fixed A, the B's whose middle edge passes
the X-obstacles are a prefix and those passing the Y-obstacles a suffix
(the two parts of the quadrilateral cut by AB shrink and grow monotonically).
The text's claims: g2 (last B passing X) and g1 (first B passing Y) never
decrease along r_i; each B accepts a prefix of the A's (beta, the A's up to
f(B)); and the angle at A accepts a prefix of the B's (alpha). Then x(A) is
the first B at or after g1(A) with f(B) ≥ A, found by one pointer, and alpha
only needs checking at x(A). g1 and g2 are two pointers (O((p + q) log h));
f(B) is a binary search (O(log p)), where the text merges in O(1) amortized.
**Each claim is checked when used**; an interval where one fails is matched
by a general sweep instead (balanced tree over beta intervals, all
intervals by binary search). `Result::section4` counts both.

**In the dynamic program.** At (i, j), both topologies of an X4-pattern
joining i and j are read from E (Fig. 30): E(a, j) with i outside (pairs
{i, a} and {l, j}, value 1 + S(i+1, a−1) + S(a, j)) and E(i, b) with j outside
(pairs {l, b} and {j, i}, value 1 + S(i, b) + S(b+1, j−1)). O(c) per interval.

**Realizing the Steiner points.** A and B lie exactly on the ray lines, so
their rounded doubles can leave a notch angle above 180° by a rounding error.
`realize_loose` moves each slightly into the open angles at its two notches
(along the sum of the right normals of the two rays), trying distances from
10⁻¹² to 10⁻² of |AB|, then the polynomial version's candidates. Every try
is validated exactly; a pattern that cannot be realized is not used. This is
done lazily, only when the DP would take the pattern, and cached.

### Why single intervals may differ from the polynomial version

The reductions behind Lemmas 6 and 16 move Steiner points until an edge
touches a boundary vertex, which may then join the pattern. A non-loose
X4-pattern inside an interval can therefore reduce to a loose one that uses a
notch outside it. Found on a random polygon (case 2959 of the differential
generator, seed 7): the polynomial version uses {5, 6 | 1, 2} inside the
interval of notches 5..2, which has no loose equivalent there; it reduces to
the loose (1, 2 | 4, 6). The opposite also happens (case 1464): the
polynomial version's fixed candidates miss a loose pattern the cubic version
finds. In 4,350 random polygons, 2 intervals were higher and 2 lower in the
cubic version; every final count agreed. The text guarantees only the final
count, so the differential test asserts that the cubic count is never worse.
`DpOutput::table` (filled under `cross_check`) holds every S(i, j) for such
comparisons.

## Evidence

Tests (all in `ctest`, from `packages/steiner-convex-partition/cpp`):

- `steiner_convex_partition_superrange_tests [count] [seed]`: `sees` and
  `sees_vertex` against brute force on random stars and integer skylines
  (full of collinear vertices), with every swept direction also compared with
  its direct trace. 3.36 million point queries and 79,492 notch pairs (1,000
  polygons, seed 17): no mismatch.
- `steiner_convex_partition_cubic_differential [count] [seed] [--timing]
  [files.bin...]`: both versions on random stars, skylines and convex polygons
  with thin V-notches (the family that produces X4-patterns), the cubic one
  with `cross_check`. Fails if the cubic version is ever worse, or if any
  cross-check throws. `--timing` turns validation and cross-checks off.
- `steiner_convex_partition_tests --cubic`, `steiner_convex_partition_brute_force
  --cubic`: the fixed cases (V-notch triangle, X4 square) and the one-sided
  optimality oracle in cubic mode.

`Options::cross_check` verifies, during a cubic run: every superrange
direction against a direct trace; every diagonal (`sees_vertex`) and every
junction's arm visibility against the direct tests; and, up to 40 notches,
every set E(i, j) against a search of all quadruples using a direct edge scan.

Results (2026-11-04):

- 17,450 random polygons (20,000 generated, seed 11; 2,550 not simple), 204
  X4-patterns used: cubic never worse, never better, all cross-checks passed.
- All corpora (Fekete, Brazilian states, São Paulo districts, USP buildings):
  the same, with cross-checks.
- The claims of Lemmas 19 and 20 held in every matching: 58,364 linear
  matchings (51,729 random, 6,635 corpora), no fallback to the general sweep.
  Of 76,866 sets with candidates on both rays, 1,840 were cut by a piece
  touching both sides and 16,662 had a degenerate quadrilateral (mostly one
  candidate per ray).

Timing (cloud container, Intel Xeon at 2.10 GHz, one thread, GCC 13, `-O2`,
`--timing`):

| Input | n | c | Polynomial | Cubic |
|---|---:|---:|---:|---:|
| `fekete-instances.bin` (16,301 polygons) | | | 5.04 s | 5.30 s |
| `br-estados` (27) | | | 1.65 s | 1.81 s |
| `sp-bairros` (32) | | | 5.83 s | 6.80 s |
| `usp-butanta-50` (50) | | | 0.22 s | 0.25 s |
| random star | 100 | 49 | 0.21 s | 0.18 s |
| random star | 200 | 90 | 1.58 s | 0.97 s |
| random star | 400 | 190 | 21.2 s (395 MB) | 7.2 s (22 MB) |
| random star | 800 | 364 | not run | 58.7 s (70 MB) |

The random stars are `test_polygons::random_star(rng, 2 * size, false)` with
`std::mt19937 rng(1000 + size)`, the first simple one of up to 50 draws,
decomposed with `validate = false`. On the corpora the polynomial version's
pruning wins slightly; on large polygons the cubic version grows as about
c^3.2 against c^3.5 and uses far less memory (the polynomial version caches
candidate X4-patterns by quadruple).

## Complexity

| Step | Text | Here |
|---|---|---|
| Convex chains, pseudo-notches | O(n) | O(n) |
| Superranges | O(n + c² log n) (Lemma 10) | O(c n log n) |
| Diagonals, jump tables | O(c² log n) | O(c² log n) |
| Junction tests | O(1) each, O(c³) | O(log n) each, O(c³ log n) |
| Ends of r_v, crossings A_ik | O(n + c² log n) | O(c min(n, c log n) + c² log c) |
| E(i, j) | O(c) each | O(min(n, c log n) + c log c) each |
| X4-patterns in S | O(c) per interval | O(c) per interval |
| Naive decomposition | O(n + c² log n) (Lemma 9) | O(n) per cut, up to O(m²) per cut while looking for a diagonal to a face vertex |
| Simplicity check of the input | (assumed) | O(n²) |

Overall O(c n log n + c³ log n + c² min(n, c log n)) plus the assembly,
against O(n + c³). The arithmetic model also differs: the text counts real
operations at O(1); here every predicate is exact, and the degrees involved
are bounded (crossings of lines through input points), so each costs O(1) in
the bit model too, with a large constant when the filter fails.

## Open work toward O(n + c³)

In order of value. Each item keeps the existing invariant: the cubic count is
never worse than the polynomial one, and the cross-checks pass.

### 1. Constant-time junction tests (Lemma 14)

Goal: drop the three `sees` queries per junction.

Lemma 14 (p. 105) defines, for each ordered notch pair, the point r_ij: the
far end of the visible segment from v_i along its extreme direction toward v_j
(the jump ray, or the notch's own range edge, or the side v_i v_j, whichever
is most restrictive), and l_ji symmetrically from v_j. With C = v_i r_ij ∩ v_j
l_ji (and A, B for the other pairs), the hexagon Q = v_i C v_j A v_k B has
edges on visible segments, so the boundary cannot enter it and every point of
its kernel sees all three notches. The X3-pattern exists iff r's and l's are
defined, A, B, C exist (the segments meet within their lengths), Q is simple
and its kernel is nonempty.

Design in our setting:

1. Per ordered pair (a, b), store the truncation of the jump ray: the ray
   from a through `last_direction(a, b)` is grazing at its target, so its
   visible part runs to the hit of the slightly counterclockwise-turned ray,
   i.e. the intersection of the ray's line with the direction's `after_edge`
   (a rational point; mirror with `before_edge` for `first_direction`). Also
   store the first contact along the side a→b (b itself when a sees b). Per
   notch, store the truncation along each edge extension (the "alone" role).
   O(c² log n) in total.
2. Per subtree (`Half`), when it is stored, compute the truncation along its
   arm's line from its root notch, both ways (O(log n) each, O(c² log n)
   total), and carry it in `RoleSpec`.
3. In `JunctionSolver::solve`, for each of the three notches and each side,
   pick the most restrictive of the candidate lines through the notch
   (nested wedges: an O(1) angular comparison), take its truncation point,
   and check that each corner (intersection of the two lines meeting between
   two notches) lies ahead of both notches and before both truncation points
   (exact, rational of low degree; reuse `LooseX4::RPoint` and its filtered
   predicates, moved to a shared header). If a corner fails, return no point;
   otherwise accept the searched point without `sees`.
4. Special cases: both lines between a and b are the side ab (then a must see
   b); collinear lines; a corner at a notch.
5. Validation: under `cross_check`, also run the current search plus `sees`
   and throw if the decisions differ in either direction (a "yes" whose
   point fails `sees` breaks sufficiency; a "no" where the old path finds a
   point breaks necessity). Using the exact first contact instead of the
   text's chord cut is never more restrictive, which helps necessity.

### 2. The safety region in O(c) (Lemma 18)

Today: an event walk costing O(min(n, c log n)) per set, which makes E the
O(c³ log n) part in the worst case. The text (pp. 119–125) builds C by
wrapping a line around the obstacles of the boundary portion V between v_i
and the notch w after a_p: from the current vertex x it locates the next one
by scanning SR(x) counterclockwise, using the function NEXT(x, y) that maps a
notch pair to a superrange pair and the global `cnext`, which only moves
counterclockwise, so the scans add up to O(c). The proof that the result is
the hull of S1 (the boundary pieces on one side) and the termination rule
(substitute B_q once it lies in the wedge of consecutive edges) are on
pp. 124–125. Reproducing it needs the superrange pairs (a_p, b_p) indexed so
that each notch can be located between consecutive pairs in O(1) (an O(c)
table per notch, O(c²) in total). The current walk is a good cross-check for
it.

### 3. Superranges in O(c polylog) per notch (Lemma 10)

Today: a sweep over the n edges, O(n log n) per notch, which breaks
O(n + c³) when n is much larger than c². Design: sweep over the O(c) convex
pieces of `ConvexChains` instead. Seen from the notch, each piece splits at
its tangent point (binary search, O(log n)) into at most two angularly
monotone parts; events are the parts' ends; the tree compares two parts along
the sweep ray by intersecting the ray with each (O(log n)). This gives
O(c log c log n) per notch, O(c² log c log n) in all, which is
O(n + c³) as the text's own c² log n term is. The direct trace and the
existing superrange test are the oracle.

### 4. f(B) without binary search (Lemma 19)

f(B) is found by binary search over the A's (O(q log p) per set). The text
merges the points f(B) with the A's along r_i in O(p + q); this needs f to be
monotone in B or a merge argument. Small gain, since the walk dominates.

### 5. The naive decomposition (Lemma 9)

`assembly.cpp` is shared by both versions. It keeps faces as vectors of point
ids and, for each remaining notch, first looks for a diagonal to a face vertex
inside the notch's cone (O(m) candidates, each checked against the face in
O(m)), which keeps cut endpoints exact, then falls back to extending an edge.
Lemma 9's bound needs each cut found by ray shooting (a superrange query
against the boundary plus a check against the O(c) cuts already placed) and
faces kept in a doubly connected edge list so that splitting is O(1). The
diagonal preference would have to go or become a bounded search; it changes
coordinates, not counts.

### 6. Input check

`prepare_polygon` checks simplicity by testing all edge pairs (O(n²)). The
text assumes a simple polygon. An option to trust the input would remove the
term; a sweep (Shamos–Hoey) would make it O(n log n).

### 7. Rational Steiner points (optional)

Steiner points are doubles in both versions, which keeps arithmetic bounded
(no growth along chains of subtrees) but means a pattern whose only feasible
Steiner point is not representable can be missed. Carrying exact rational
Steiner points through the DP would make the count exactly the text's
optimum, at the cost of big integers growing with subtree depth (outside the
real-RAM bound). See the main document's limitations.

## How to resume

```bash
cd packages/steiner-convex-partition/cpp
cmake -S . -B .build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build .build
ctest --test-dir .build --output-on-failure
```

```bash
./.build/steiner_convex_partition_cubic_differential 20000 11
```

```bash
./.build/steiner_convex_partition_cubic_differential 0 --timing ../../../benchmarks/suites/fekete-instances.bin
```

- Start any change to the cubic version by running the differential test with
  cross-checks; keep it at "worse on 0" and no throw.
- To look at a failing case, the differential test prints its index; the
  generators are deterministic (`tests/test_polygons.h`), so the same
  `(count, seed)` reproduces it. `DpOutput::table` (under `cross_check`)
  exposes every S(i, j) for interval-level comparisons.
- History: polynomial version `089af63`; Section 4 `aea5c88`; chain queries
  `b343410`; superrange sweep `4ad12cf`; logarithmic hull test `37d2260`;
  Lemmas 19–20 `6a06203`. The branch-and-bound experiment (removed) is in
  `b7cb532` and `c122555`.
