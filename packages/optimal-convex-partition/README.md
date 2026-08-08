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

## Build

```sh
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## CGAL Parity Test

The optional parity executable requires CGAL and compares this port against
CGAL output. CGAL is run in isolated child processes so known CGAL crashes in
the fixture corpus are reported as skipped instead of killing the test run.

```sh
cmake -S cpp -B build-parity \
	-DCMAKE_BUILD_TYPE=Release \
	-DOPTIMAL_CONVEX_PARTITION_BUILD_CGAL_PARITY_TEST=ON
cmake --build build-parity --config Release
./build-parity/optimal_convex_partition_cgal_parity \
	../nonconvex-tpp/cpp/tests/test_cases_simplified2.bin
```

Current validation result on the TouringPolygons fixture corpus:

```text
matched 16299 polygons, skipped 4 CGAL crash polygon(s)
```
