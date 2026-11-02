# Optimal Convex Partition

Standalone C++ port of `CGAL::optimal_convex_partition_2` for simple 2D polygons.

The core library does not depend on CGAL. It exposes:

```cpp
#include <optimal_convex_partition/optimal_convex_partition.h>

optimal_convex_partition::Polygon polygon = {
	{0.0, 0.0},
	{1.0, 0.0},
	{1.0, 1.0},
	{0.0, 1.0},
};

optimal_convex_partition::Partition pieces =
	optimal_convex_partition::decompose_polygon(polygon);
```

## Collinear vertices

CGAL's algorithm (Greene's dynamic program) assumes that no three vertices are
collinear. Its visibility graph admits segments through vertices and along
edges, and its collinear additions put a vertex into a piece even when no chain
of diagonals reaches it, counting it as a free start. On polygons with
collinear vertices this can close a piece through a reflex vertex. Paula's
50pcb442, polygon 6, gets a piece that runs along x=300 through (300,900) and
(300,1000). CGAL itself fails its postcondition `convex_partition_is_valid_2`
there.

`decompose_polygon` therefore checks every piece in exact arithmetic: every
turn left or straight ahead, one revolution. When all pieces pass, the result
is CGAL's, unchanged. When one fails, the same dynamic program runs again over
proper diagonals only, whose open segments lie in the open interior:

- A chain starts at the first vertex of its chord and costs nothing there. No
  other vertex is a free start.
- A pivot reuses only records whose turn is not to the right.
- A vertex belongs to a piece when it sees the chord's ends along a straight
  chain of edges and proper diagonals, as a vertex on a straight side does.

Every sub-polygon is then simple and every turn is checked. A polygon that is
not simple has no convex partition at all. For one, the first result is kept,
as before. A simple polygon whose pieces still fail raises an exception.

Validation (2026-10-27):

- **Real polygons.** Over 123,343 polygons (the suites, Paula's 297 files, the
  German and SoCG corpora, São Paulo, and the smoothed instances up to 1,000
  vertices), only the six simple polygons with a non-convex piece change.
  They are 50pcb442 polygon 6, and five from `npol-gt-100`. Each now has the
  minimum piece count found by exhaustive search.
- **Random polygons.** On 46,386 random simple grid polygons, with collinear
  vertices in every direction, every partition is convex. Every partition
  that CGAL got right is unchanged. All 1,344 recomputed partitions have the
  minimum piece count found by exhaustive search; the largest has 52 vertices.
- **What stays as before.** Where CGAL's pieces are convex, its count is kept,
  even when it is not minimal: one or two pieces too many in 1,249 of the
  42,018 random polygons checked by exhaustive search.
- **Cost.** All three builds ran in one process, alternating per polygon, on
  the real corpora. The check adds at most 0.6% (Paula's polygons, about 9 µs
  each) and is within noise elsewhere. The recomputation, when it happens,
  takes 0.55 times CGAL's pass on the 27 nonconvex polygons of 150 to 700
  vertices: 64 s against 115 s on sp-bairros' 673-gon. It is up to 7.6 times
  slower on near-convex 256-gons, 0.08 s at most, where its exact visibility
  test, O(n^3), dominates.

## Build

```sh
cmake -S cpp -B .build -DCMAKE_BUILD_TYPE=Release
cmake --build .build --config Release
```

## CGAL Parity Test

The optional parity executable requires CGAL and compares this port against
CGAL output. CGAL is run in isolated child processes so known CGAL crashes in
the fixture corpus are reported as skipped instead of killing the test run. A
partition that differs only where CGAL's pieces are not convex, and whose own
pieces are, is counted apart, as is the nonconvex package's
`tpp-compare-cgal-partition`. On Paula's six polygons, both report six such
partitions.

```sh
cmake -S cpp -B .build-parity \
	-DCMAKE_BUILD_TYPE=Release \
	-DOPTIMAL_CONVEX_PARTITION_BUILD_CGAL_PARITY_TEST=ON
cmake --build .build-parity --config Release
./.build-parity/optimal_convex_partition_cgal_parity \
	../../benchmarks/suites/nonconvex/test_cases.bin
```

Current validation result on the TouringPolygons fixture corpus:

```text
matched 16303 polygons, skipped 0 CGAL crash polygon(s), 0 where only the standalone pieces are convex
```
