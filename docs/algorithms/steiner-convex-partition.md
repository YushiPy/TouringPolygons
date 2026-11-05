# Minimum convex decomposition with Steiner points (Chazelle and Dobkin)

Package: `packages/steiner-convex-partition` (standalone, no dependencies).
References, cited below by lemma number of the 1985 text:

- B. Chazelle, D. P. Dobkin, *Decomposing a polygon into its convex parts*,
  STOC 1979, pp. 38–48 (the first polynomial algorithm, O(n⁶));
- B. Chazelle, D. P. Dobkin, *Optimal convex decompositions*, in
  *Computational Geometry*, G. T. Toussaint (ed.), North-Holland, 1985,
  pp. 63–133 (the polynomial algorithm `ConvDec` of its Section 3 and the
  O(n + c³) refinements of Section 4, from Chazelle's 1980 thesis).

As far as we know, this is the first implementation. Both algorithms of the
1985 text are implemented, selected by `Options::algorithm`: `polynomial`
(Section 3, the default) and `cubic` (Section 4, see
[below](#the-section-4-algorithm-algorithmcubic) and
[its own document](steiner-convex-partition-section4.md)).

## Formulation

Input: a simple polygon P with n vertices, c of them notches (reflex
vertices). Output: convex polygons with pairwise disjoint interiors whose union
is P, as few as possible. Piece vertices may be **Steiner points**: points
inside P, or points of the boundary that are not vertices. Angles of 180° are
allowed in pieces. Trivially ⌈c/2⌉ + 1 ≤ OPT ≤ c + 1.

Greene's dynamic program (`optimal-convex-partition`) solves the version
without Steiner points, so its count is an upper bound for this one.

## The theory used

- **X-patterns** (Definition 2). A tree drawn inside P whose nodes of degree 1
  or 2 are notches, removed by their incident edges, and whose nodes of degree
  3 are Steiner points with no angle above 180°. An X_k-pattern has k notches
  and adds k − 1 pieces.
- **Counting** (Lemmas 2 to 5). Some optimal decomposition is an
  X-decomposition, and one with p compatible X-patterns plus one cut per
  remaining notch (the *naive decomposition*) has exactly c + 1 − p pieces. So
  the problem is to maximize the number p of compatible patterns.
- **Compatibility** (p. 85). Two X-patterns do not cross if and only if their
  notch sequences do not interleave along the boundary. Packing patterns is
  therefore a non-crossing interval problem on the circular sequence of
  notches.
- **Y-patterns** (Definition 3, Lemma 6). Every X-pattern that is not an
  X4-pattern can be reduced, without losing patterns, to a Y-pattern: no edge
  joins two Steiner points (condition 1), and on a path N2–N3–N2 the two
  notches of degree 2 have their other edges on opposite sides of the path
  (condition 2). So Y-patterns and X4-patterns suffice.
- **Relation (2).** If a Y-pattern is split at a notch v of degree 2 into two
  subtrees, one inside an interval V(i, j) of notches, then without loss of
  generality the patterns in the gaps of that subtree number exactly S(i, j),
  the optimum inside V(i, j). Among such subtrees only the angle of the edge
  at v (the *arm*) matters to the rest, so one subtree per interval and side
  suffices: the one whose arm leaves the most room (STEP 4, B(i, j) and
  F(i, j)).
- **X3 locus** (Lemma 14). The Steiner points of an X3-pattern on three notches
  form a convex region.
- **Loose X4-patterns** (Definition 4, Lemma 16). An X4-pattern not reducible
  to an X3- or Y5-pattern can be moved so that the edge at each of its four
  notches may lie along either extension of the notch's polygon edges; all
  16 such configurations are valid.

The 1985 text calls its proofs sketches and refers to the thesis for details.
This implementation relies on these statements as published.

## The dynamic program as implemented

`src/dynamic_program.cpp`. Notches are numbered 0..c−1 counterclockwise; an
interval [i..j] has i ≤ j ≤ i + c − 1 and indices are reduced modulo c. At a
notch, directions are ordered counterclockwise from the edge toward the next
vertex. With two arms, the one toward the later notches (the *forward* subtree)
comes first, so the three gaps are next edge → forward arm → backward arm →
previous edge, each at most 180°.

The text's angle conventions (clockwise notch order, ∠ measured
counterclockwise, "u < 180") do not match its figures consistently. The
orders and comparisons here are derived from this face structure instead.

Tables, for 2 ≤ len ≤ c, all starting notches:

- **S(i, j)**: maximum number of compatible patterns with notches in [i..j];
- **B(i, j)** (len < c): best forward subtree rooted at i. The arm leaves i,
  the other notches are in (i..j], the gaps hold S(i, j) patterns, and the arm
  is the most counterclockwise one;
- **F(i, j)** (len < c): the mirror image, rooted at j, with the most
  clockwise arm.

S(i, j) is the maximum of:

1. split: S(i, k) + S(k + 1, j);
2. X4-pattern on i < a < b < j, pairing {i, a}{b, j} or {a, b}{j, i}:
   1 + S(i+1, a−1) + S(a+1, b−1) + S(b+1, j−1);
3. a Y-pattern containing i and j, by the shape of its path from i to j:
   - a notch k of degree 2 on the path: F(i, k) and B(k, j) patched at k,
     1 + S(i, k) + S(k, j);
   - the diagonal i–j, with an optional forward subtree B(i, k−1) at i and an
     optional backward subtree F(k, j) at j;
   - one Steiner point joined to a third notch k. Condition 2 leaves two shapes:
     forward subtrees at i and k (optional) with j alone, or backward subtrees
     at k and j (optional) with i alone.

B(i, j) keeps the best arm among the candidates whose gaps hold exactly
S(i, j):

- B(i, k) with a wider trailing gap;
- the diagonal i→k to a notch k carrying B(k, j);
- the diagonal i→j to j carrying F(i+1, j), or to j alone;
- a Steiner point joined to k and j, each carrying a backward subtree or none
  (the only shape allowed by condition 2 when i is the root).

F is symmetric. The answer is max over i of S(i, i + c − 1). In exact
arithmetic every start gives the same value, and `Result` reports the minimum
and maximum as a consistency check.

### Where this departs from or fills gaps in the text

- **Empty subtrees.** The text initializes B(i, i) and F(i, i) to "0" and uses
  that value both for "no subtree" and "infeasible". Here a notch with a
  single arm is an explicit case of every recurrence (notch *alone*).
- **Arm order at a patch.** The kept subtree has the best arm, which can make
  its arm pass the other arm at the shared notch. Then that arm alone removes
  the notch, so the subtree plus its root is a complete pattern. Because of
  relation (2), that pattern would beat S on the subtree's interval, which the
  recurrence already covers. A patch therefore requires the arms in order, and
  nothing is lost.
- **Junction shapes.** STEP 2 and B4/F4 enumerate exactly the subtree
  combinations at a Steiner point that condition 2 allows. The derivation is
  in the comments of `try_junction_combos` and `compute_forward`.
- **Cost of the polynomial algorithm.** Steiner points are found by a direct
  convex search, and X4-patterns by a per-pair filter plus a test on four
  notches. The work is O(c³) junction searches of O(n) each and O(c⁴) X4
  candidates, pruned by value bounds and a per-pair necessary condition. On
  the corpora below the largest polygon (673 vertices, 341 notches) takes
  4.5 s, against 303 s for all 32 São Paulo districts with Greene's port. The
  Section 4 version below replaces both searches.

### Steiner points

`JunctionSolver` looks for a Steiner point s joining notches a, b, c
(counterclockwise on the boundary, triangle abc counterclockwise). Every
condition is a closed or strict half-plane in s:

- the angles at s are at most 180° iff s lies in the triangle;
- each notch's gaps, given the arm already there, are at most 180°
  (half-planes through the notch), and s is off the rays of existing arms;
- the boundary arc from a to b must stay, inside the triangle, in the
  sub-triangle (a, b, s) and off the arms. This holds iff no arc edge meets the
  sides [a, c] or [b, c] except at a and b; s is strictly left of a→b when an
  arc edge touches the open side (a, b); and the arc vertices in the triangle
  give two binding strict constraints, one through a and one through b
  (the most clockwise vertex seen from b, the most counterclockwise one seen
  from a). The same holds for the other two arcs.

These constraints are clipped in long double. Candidates are the extreme
vertex of the region for the subtree's objective, its double neighbors and
points moved toward the centroid. When the region is thin, the candidates
come from the vertices of the arrangement of constraint lines, computed with
big integers and rounded. A candidate is accepted only after an exact check
of every constraint and a direct exact check of its three arms against the
whole boundary.

**Steiner points are doubles.** A subtree's arm is therefore within a rounding
step of the true extreme. Dominance compares the arms actually realized,
exactly. This keeps every later computation on doubles, so there is no growth
of rational coordinates along chains of patterns.

### X4-patterns

`X4Solver` intersects the two edge extensions of each notch of a pair. For a
loose pattern all four intersections must exist, and some of them (or their
average) must be joined to both notches by clear segments that remove both
angles. This is the per-pair filter. It then tries the averages, the bisector
intersections and the 16 corner combinations for the two Steiner points,
rounded to doubles, and checks each candidate exactly: both angle triangles,
every notch removed, five clear segments, no crossings.

## The Section 4 algorithm (`Algorithm::cubic`)

Section 4 of the 1985 text keeps the dynamic program and makes its geometric
tests cheap. The cubic version here shares the tables, choices and
reconstruction with the polynomial one, and replaces:

- the arc scans of junction searches by **superranges** (what a notch sees,
  compressed to the directions of the pseudo-notches) and per-pair jump
  tables (Lemmas 8 to 14);
- the enumeration of X4-patterns by the sets **E(i, j)** of loose
  X4-patterns (Lemmas 16 to 21), decided exactly with every arm along its
  notch's edge extension.

It never returns more pieces than the polynomial version (checked on 17,450
random polygons and every corpus, with self cross-checks), is about as fast on
the corpora and faster on large polygons (2.9 times at 190 notches, with a
twentieth of the memory). It does not yet reach the text's O(n + c³).

**Everything about it is in
[`steiner-convex-partition-section4.md`](steiner-convex-partition-section4.md):**
the mapping to the text, the conventions, each construction that replaces a
proof only found in Chazelle's thesis, the evidence, the complexity table and
the open work with a design for each item.

## Assembly and validation

`src/assembly.cpp` keeps faces as counterclockwise cycles of exact points:
homogeneous coordinates with big integers in a common power-of-two frame.

1. Every pattern is inserted as polyline chords, each checked exactly to lie
   inside one face.
2. Every remaining notch gets one cut: a diagonal to a visible vertex of its
   face inside its cone when one exists, otherwise the extension of one of its
   edges up to the face boundary. The extension's endpoint is the rational
   intersection of two lines through doubles, and both faces sharing that edge
   receive it.

Each insertion splits one face, so the faces tile P by construction. Then
`validate_partition` (`src/validate.cpp`) checks independently and exactly:

- every piece is weakly convex with one revolution;
- no polygon edge meets a piece's open interior, and one interior point per
  piece lies inside P;
- pieces are pairwise separated by one of their edges;
- the doubled areas add up. This uses exact fractions grouped by denominator,
  with a long double check above 48 distinct denominators.

## Branch-and-bound experiment (removed)

**Status: an experiment, not part of the solvers.** The library was wired into
`nonconvex-tpp` as an alternative to Greene's decomposition, measured, and
removed again because it did not change the running time. `nonconvex-tpp`, its
CLIs, its tests and the dashboard's WASM build are back to Greene only, and
this package is linked by nothing. To repeat the experiment, recover the
integration from commit `b7cb532` (options, `--decomposition`,
`tpp-decomposition-timing`, tests) and `c122555` (results); the summary is in
[`benchmarks/results-saved`](../../benchmarks/results-saved/README.md#steiner-decomposition-cloud-2026-11-04).

How it was wired: `tpp::decompose_polygon(polygon, method)` chose Greene's
port (`greene`, the default) or this decomposition (`steiner`), and the
free-order TPP and TSPN branch and bound and the fixed-order solver passed the
choice on. The branch and bound decomposes a region lazily, when it first
branches on its pieces. The cubic version did not exist yet; the experiment
used the polynomial one.

What the search needs, and why it held (still true of the library's output):

- **Lower bounds** need every point of the region to lie in some piece. The
  pieces tile the region exactly; with rounded cut endpoints they are a convex
  cover (see the output coordinates above).
- **Upper bounds** are checked against the original regions (`covered`, with
  the feasibility tolerance), never against pieces.
- **Robustness.** The independent validation is skipped there (`validate =
  false`), because each insertion and each piece is already checked exactly
  while the partition is built, and the validation would add about 30%. A
  region the library rejects as not simple (two of Fekete's 16,303 polygons
  cross themselves) gets Greene's decomposition, as before.

The tests solve every exhaustive free-order and TSPN case with both
decompositions and compare them with the exhaustive optimum over the pieces of
each decomposition.

**Measured effect** (Fekete's 558 instances, 2026-11-04;
[`benchmarks/results-saved`](../../benchmarks/results-saved/README.md#steiner-decomposition-cloud-2026-11-04)):

- The decomposition itself is slower: decomposing every nonconvex polygon of
  every instance takes 5.44 s against 3.27 s (1.9× per instance, geometric
  mean, from 0.3× to 8.7×), for 1.6% fewer pieces.
- In the TSPN search it stays a small share: 1.3% of the solve time against
  0.4%. It is visible only on instance 553, whose 34 large nonconvex polygons
  are all decomposed (0.56 → 1.87 s).
- The search does not change measurably. With the same bounds on all 556
  instances closed by both, nodes change by 0.04% overall. Of the 43
  instances where it decomposes into fewer pieces, 30 expand exactly the same
  nodes and the others move between −6% and +1.4%. On the 85 instances
  that take at least 0.5 s or get different pieces (median of 3 runs), the
  time without the decomposition is 152.8 s against 152.9 s.
- The free-order TPP (one run) behaves the same: both close the same 551
  instances. Nodes change by −0.06%, and the decomposition goes from 0.30 to
  0.45 s, 0.08% to 0.12% of the solve time. The total, 384.5 against
  390.1 s, is within the noise of the shared machine.

The search branches almost only on the visiting order. In the TSPN run,
decomposition branches are 0.08% of the branch events (2,173 against 2.79
million insertions), and 0.07% in the TPP run. Only 198 of the 556 instances decompose any region, 602
regions in all, at 4.4 pieces each, and the Steiner decomposition saves 42 of
their 2,640 pieces. So the integration was removed. It could matter
for a search that branches more on pieces, or for regions where Steiner points
remove many pieces; nothing in the current solvers is like that.

## Status of exactness and limitations

- **Validity**: exact, always checked; a failure throws `std::logic_error`.
- **Optimality**: the DP maximizes p over Y-patterns and loose X4-patterns
  exactly as above. The count equals the optimum under the 1985 lemmas, with
  three caveats:
  1. Steiner points are rounded doubles inside the feasible region. A pattern
     whose feasibility needs a subtree arm at an extreme that is not
     representable, or a region of zero area at a non-representable point,
     can be missed. Validity is unaffected.
  2. X4-patterns are found only through the loose configurations of
     Lemma 16. The cubic version decides each loose configuration exactly;
     the polynomial one tries a fixed set of candidates, so it can miss one
     (it never did on the final counts of our tests).
  3. The proofs of the lemmas in the 1985 text are sketches.

  The result is a valid decomposition that is believed optimal. It is not
  certified optimal: no lower bound beyond ⌈c/2⌉ + 1 is computed.
- **Output coordinates**: cut endpoints on edges are rational; `pieces`
  rounds them when they are not doubles (`exact_coordinates = false`). All
  pieces share each rounded point, on the boundary it goes on or outside the
  edge, and a piece with a rounded point is returned as its exact convex hull.
  The pieces are then a convex cover of the polygon, exceeding it and
  overlapping by at most rounding errors. That is what a branch and bound
  needs: every point of the region lies in some piece.
- **Complexity**: the polynomial version is O(n c³) plus O(c⁴) X4
  candidates; the cubic version is O(c n log n + c³ log n + c² min(n, c log n))
  plus the shared assembly (see the
  [Section 4 document](steiner-convex-partition-section4.md#complexity)),
  against O(n + c³) in the text.

## Validation evidence (2026-11-04)

- Unit tests (`steiner_convex_partition_tests`): fixed cases with known
  optima and 600 random polygons. The triangle with three V-notches has 3
  pieces against 4 with diagonals only; the X4 square has 4 pieces against 5
  without X4-patterns.
- One-sided optimality oracle (`steiner_convex_partition_brute_force`): on
  37,427 random polygons with up to 9 notches, 2,891 of which use interior
  Steiner points, an independent search packs X2-patterns, sampled X3-patterns
  and two-diagonal chains with its own interval DP. It never found more
  patterns than the decomposition.
- Section 4 version: see the
  [Section 4 document](steiner-convex-partition-section4.md#evidence)
  (superrange queries against brute force, differential test against the
  polynomial version, self cross-checks).
- Comparison with Greene's port (`steiner_convex_partition_compare`), never
  more pieces on any corpus. See the package README for the table: 535 of
  16,301 Fekete polygons, 17 of 27 Brazilian states, 11 of 32 São Paulo
  districts and 8 of 50 USP buildings get fewer pieces.
  Every gain uses Steiner points (48 interior, 487 only on edges in the Fekete
  corpus), none comes from polygon vertices alone. All starting notches of
  the circular DP agree on every polygon.
