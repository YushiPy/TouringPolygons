# Steiner Convex Partition

Minimum convex decomposition of a simple polygon with Steiner points, after
Chazelle and Dobkin:

- B. Chazelle, D. P. Dobkin, *Decomposing a polygon into its convex parts*,
  STOC 1979, pp. 38–48;
- B. Chazelle, D. P. Dobkin, *Optimal convex decompositions*, in
  *Computational Geometry*, G. T. Toussaint (ed.), North-Holland, 1985,
  pp. 63–133.

Unlike Greene's dynamic program (CGAL's `optimal_convex_partition_2`, ported in
`../optimal-convex-partition`), pieces may have vertices that are not vertices
of the polygon: Steiner points inside it, and points where a cut meets an edge.
The minimum with Steiner points is never larger than the minimum without them,
and is often smaller.

The library is standalone: C++20, no dependencies, its own exact arithmetic.
The contract, the algorithm as implemented and its correctness argument are in
[`docs/algorithms/steiner-convex-partition.md`](../../docs/algorithms/steiner-convex-partition.md).

```cpp
#include <steiner_convex_partition/steiner_convex_partition.h>

steiner_convex_partition::Polygon polygon = {
	{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2},
};
steiner_convex_partition::Result result = steiner_convex_partition::decompose(polygon);
// result.pieces: counterclockwise convex pieces; result.pattern_count,
// result.patterns, result.validated, ...
```

Both algorithms of the 1985 text are available: `Options::algorithm` selects
`Algorithm::polynomial` (Section 3, the default) or `Algorithm::cubic`
(Section 4: superranges and the sets E(i, j)). They share the dynamic program
and the assembly. The cubic version decides X4-patterns exactly in their loose
configuration, so it never returns more pieces. It is about as fast on the
repository's corpora and faster on large polygons (2.9 times at 190 notches).
`Options::cross_check` makes it verify each of its shortcuts against the
direct computation. Its status, its departures from the text and the open
work are in
[`docs/algorithms/steiner-convex-partition-section4.md`](../../docs/algorithms/steiner-convex-partition-section4.md).

## What the result guarantees

- **Validity, exactly.** Every piece is convex, the pieces lie in the polygon,
  their interiors are disjoint and their areas add up to the polygon's. Every
  predicate is exact (filtered doubles, floating-point expansions, then big
  integers). `decompose` checks this before returning (`Options::validate`,
  on by default) and throws if it fails; it has never failed in testing.
- **Piece count.** With c notches and p patterns the partition has
  c + 1 − p pieces (Lemmas 3 and 4 of the 1985 paper); the dynamic program
  maximizes p. See the algorithm document for where this is exact and where it
  depends on rounding of Steiner points to doubles.
- **Coordinates.** Polygon vertices and interior Steiner points are doubles
  and exact. A point where a cut meets an edge is rational and in general not
  a double; `pieces` then holds its rounded value and `exact_coordinates` is
  false (the validation ran on the exact value). Every piece that has such a
  point uses the same rounded value, on the polygon boundary it is rounded to
  a double on or outside the edge, and such a piece is returned as its exact
  convex hull. The pieces are then a convex cover of the polygon, exceeding it
  and overlapping by at most rounding errors. This is what a branch and bound
  needs from a decomposition: every point of the region lies in some piece.

It is not linked by the solvers. An experiment used it in `nonconvex-tpp`'s
branch and bound in place of Greene's decomposition; the running time did not
change, so the integration was removed (see
[the algorithm document](../../docs/algorithms/steiner-convex-partition.md#branch-and-bound-experiment-removed)).

Input: a simple polygon in either orientation; repeated consecutive vertices
are ignored and straight (180°) vertices are allowed. Polygons that are not
simple raise `std::invalid_argument`.

## Build and test

```sh
cmake -S cpp -B .build -DCMAKE_BUILD_TYPE=Release
cmake --build .build
ctest --test-dir .build --output-on-failure
```

`steiner_convex_partition_tests` runs fixed cases (an L-shape, an H-shape, a
triangle with three V-shaped notches that needs one Steiner point, a square
whose four notches need an X4-pattern) and random star-shaped and rectilinear
polygons, checking every result exactly. `steiner_convex_partition_brute_force`
packs X2-patterns, X3-patterns found by sampling Steiner points, and chains of
two diagonals with its own interval DP, and fails if that packing beats the
decomposition. Both also run with `--cubic`.

For the Section 4 version, `steiner_convex_partition_superrange_tests` compares
superrange queries with brute force, and
`steiner_convex_partition_cubic_differential [count] [seed] [--timing] [file.bin...]`
runs both algorithms on random polygons and fixture files, with the cubic
one's cross-checks on. It fails if the cubic version ever returns more pieces.

## Comparison with Greene's algorithm

The optional target compares piece counts with `../optimal-convex-partition`
on fixture files (it needs Boost headers, like that package):

```sh
cmake -S cpp -B .build -DSTEINER_CONVEX_PARTITION_BUILD_GREENE_COMPARISON=ON
cmake --build .build
./.build/steiner_convex_partition_compare ../../benchmarks/suites/fekete-instances.bin
```

Results of 2026-11-04 (cloud container, Intel Xeon at 2.10 GHz, one thread, `RelWithDebInfo` build):

| Corpus | Polygons | Pieces (ours) | Pieces (Greene) | Fewer pieces | More pieces | Time ours / Greene |
|---|---:|---:|---:|---:|---:|---:|
| `fekete-instances.bin` | 16,301 | 41,562 | 42,113 | 535 | 0 | 7.9 s / 3.7 s |
| `br-estados` | 27 | 844 | 881 | 17 | 0 | 1.8 s / 10.6 s |
| `usp-butanta-50` | 50 | 447 | 457 | 8 | 0 | 0.3 s / 0.06 s |
| `nonconvex/custom_tests.bin` | 9 | 21 | 21 | 0 | 0 | 0.0 s / 0.0 s |
| `sp-bairros` | 32 | 1,248 | 1,267 | 11 | 0 | 6.4 s / 303.4 s |

Every partition passed the exact validation and every starting notch of the
circular DP gave the same pattern count. The two polygons of the Fekete corpus
that are left out are not simple. Of the 535 Fekete polygons with fewer pieces,
48 use interior Steiner points and 487 only cut endpoints on edges; none uses
polygon vertices only, so none of the gains comes from Greene's port being
suboptimal on collinear vertices. 1,575 Fekete partitions have a cut endpoint
that is not a double.
