# Touring Polygons Problem

This repository contains implementations and experiments for the **Touring Polygons Problem (TPP)**: given a starting point $s$, an ending point $t$, and polygonal regions in the plane, find the shortest Euclidean path from $s$ to $t$ that visits every region. It includes both the fixed-order variant, where the sequence is part of the input, and the free-order variant, where the solver chooses the sequence. The path may touch the boundary of a polygon or pass through its interior.

The problem is introduced in [Dror et al. (2003)](#bibliography) and can be seen as a special case of the Traveling Salesman Problem with Neighborhoods (TSPN), where the regions are polygons. The repository preserves the fixed-order legacy solver while also maintaining the non-convex free-order branch-and-bound path.

The maintained interactive tools live in `apps/benchmark-dashboard`; the SIICUSP
event package is a separate static, self-contained publication.

---

## Repository Structure

```
apps/
├── benchmark-dashboard/  # Maintained local workbench and event viewer
├── siicusp34/             # Static archival event page
benchmarks/
├── scripts/              # Benchmark and instance-generation command internals
├── suites/               # Tracked canonical benchmark suites
├── campaigns/            # Ignored generated campaigns
├── results/              # Ignored benchmark outputs
└── archive/              # Historical benchmark data kept for reference
packages/
├── common-geometry/      # Shared C++ vector and geometry primitives
├── convex-tpp/           # Maintained convex TPP solvers and legacy prototypes
├── nonconvex-tpp/         # Fixed-order and free-order non-convex solvers
├── optimal-convex-partition/ # Shared CGAL decomposition library
├── fenced-tpp/           # Fenced TPP code (legacy)
└── instance-generation/  # Reproducible instance generation
tspn-comparison/
└── solver-oracle/        # Pinned fork of the German comparison solver
docs/
├── algorithms/           # Current algorithm specifications and audits
├── bibliography/         # Source papers and LLM-friendly TeX conversions
└── reports/              # Portuguese LaTeX reports
```

The repository map and lifecycle decisions are documented in
[`docs/architecture.md`](docs/architecture.md). Local checkouts of external
data remain ignored; the modified German solver is a pinned submodule. See
[`docs/third-party.md`](docs/third-party.md).

This repository is organized as a research monorepo. The maintained C++ code is
split into reusable CMake targets:

```text
tpp_geometry -> tpp_convex -> optimal_convex_partition -> tpp_nonconvex
```

The common geometry package owns `Vector2`/`Vec2` and low-level geometric
helpers. The convex package owns the exact convex TPP solvers. The non-convex
package owns fixed-order and free-order Branch and Bound and calls the convex
solver package instead of carrying a second convex implementation. The old
Python implementations remain available as legacy research code, but the
maintained public path is the C++ API.

Repository boundaries:

- `packages/` is for maintained package code and intentional regression fixtures.
- `benchmarks/suites/` contains tracked canonical benchmark suites. Campaigns,
  split outputs, run outputs, and generated instance matrices are ignored.
- `benchmarks/generate_instances.py` regenerates ignored benchmark campaign
  inputs with a single command.

For a fresh clone or a restored machine, install all system and application
dependencies with:

```bash
scripts/install_dependencies.sh
```

The installer is safe to rerun. It installs missing system packages, recreates
the locked Python and Node environment for the maintained dashboard, and installs the
Chromium runtime used by browser tests.

Then run the fresh-clone sanity check:

```bash
scripts/sanity_check.sh
```

The script checks required local tools, installs missing system dependencies
when possible, regenerates generated suites, builds the C++ targets, verifies
convex correctness, and runs a small non-convex benchmark.

The benchmark CLI is centralized at `benchmarks/tpp.py`. A typical synthetic
benchmark workflow is:

```bash
python3 benchmarks/tpp.py create smoke --vertices 8 --polygons 20 --instances 100
python3 benchmarks/tpp.py run smoke --threads 8 --max-calls 1000000 --max-seconds 30
python3 benchmarks/tpp.py status smoke
```

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for maintenance conventions, generated
file boundaries, and the intended command entry points.

---

## The Convex TPP

When all polygons are convex and disjoint, the problem is solvable in polynomial time. The maintained C++ implementations live in `packages/convex-tpp/cpp` and share the same algorithmic framework based on two geometric structures:

**First contact region $T_i$:** the subset of the boundary of $P_i$ that can be the first point of contact of an optimal path arriving at $P_i$ — equivalently, the edges of $P_i$ whose exterior side faces the direction the path arrives from.

**Last-step map $S_i$:** a partition of the entire plane into three types of regions, determined by how the optimal $i$-path arrives at each point $p \in \mathbb{R}^2$:
- **Vertex region** (associated to a vertex $v \in T_i$): the set of points $p$ for which the optimal $i$-path ends with the segment $\overline{vp}$. Each such region is an angular cone emanating from $v$, computed from the two edges of $P_i$ incident to $v$ and their membership in $T_i$.
- **Edge region** (associated to an edge $e \in T_i$): the set of points $p$ for which the optimal $i$-path arrives at $P_i$ through the interior of $e$, determined by reflecting $p$ across $e$ and tracing the $(i-1)$-path to the reflection.
- **Pass-through region**: the remaining points, for which the optimal $i$-path simply passes through the interior of $P_i$ without touching its boundary — the $i$-path coincides with the $(i-1)$-path.

Given $S_1, \dots, S_k$, the full optimal path to any point can be recovered with at most $k$ region queries: each query either terminates at a vertex (directly) or recurses on a reflected point (edge region) or the same point (pass-through), reducing $i$ by one each time.

Three implementations are provided:

| Implementation | Key idea | Worst-case complexity |
|---|---|---|
| Linear search | Locate point in $S_i$ by scanning all vertex and edge regions | $O(n^2)$ |
| Binary search | Locate point in $S_i$ using angular binary search over the sorted cone boundaries | $O(nk \log(n/k))$ |
| Memoized (original) | Same as binary search, but cache $i$-path queries to avoid recomputation across calls | $O(nk \log(n/k))$ worst case, significantly faster in practice |

The memoized approach is believed to be original and is not described in the reference paper.

A fourth implementation follows the $O(nk)$ algorithm of Tan & Jiang, but performs worse in practice since it operates in a pure dynamic programming style that does not admit memoization.

---

## The Non-Convex TPP

When polygons may be non-convex, the problem is NP-hard. The approach decomposes each polygon into convex pieces and searches over all combinations using Branch and Bound.

### Convex Decomposition

Each non-convex polygon is partitioned into convex pieces using Greene's algorithm (via CGAL). This produces on the order of 10 pieces per polygon in practice (for the city-neighborhood instances used in testing). The decomposition is stored as a list of convex polygons per input polygon.

### Incumbent (upper bound)

An initial feasible solution is obtained by a shortest-path heuristic on a graph whose nodes are the vertices and edge midpoints of the convex hull of each polygon. A straight-line $s$-to-$t$ path is checked first; otherwise a DP over graph nodes gives an initial incumbent. More midpoints can be added at the cost of a larger graph, trading computation for a tighter initial bound.

### Branch and Bound

The search tree assigns one convex piece per polygon. Each node represents a partial assignment $(c_1, \dots, c_j)$ for the first $j$ polygons. The lower bound at a node substitutes the convex hull of each remaining polygon and solves the resulting all-convex TPP exactly. If this bound exceeds the incumbent, the subtree is pruned.

Exploration is DFS. Pieces that intersect the segment between adjacent incumbents' path points are explored first as a heuristic for stronger early bounds.

**Empirical performance:** on 60-polygon instances with ~10 pieces each (full enumeration cost $\approx 10^{33}$ convex calls), the solver makes roughly 2 million convex calls and terminates in ~25 seconds.

### MILP formulation

An alternative exact approach models the non-convex TPP as a Mixed Integer Linear Program, solvable with commercial solvers (Gurobi). This serves as a correctness baseline and provides a comparison point for the Branch and Bound.

---

## Free visit order (non-convex TPP)

The maintained unordered solver finds an endpoint path through arbitrary simple
polygons. It combines insertion branching and lazy convex decomposition in one
search tree, reusing the convex geometric solver and existing decomposition.
A support-function dual certificate protects lower bounds; a commercial-solver-free
interior-point fallback handles subproblems that the geometric solver cannot certify.
The result includes the visit order, path, bounds, termination reason, and search statistics.

```bash
scripts/verify_unordered.sh
.build/unordered/tpp < packages/nonconvex-tpp/cpp/tests/unordered-example.txt
```

See [`docs/algorithms/unordered-tpp.md`](docs/algorithms/unordered-tpp.md) for the
algorithm, numerical guarantees, API, input format, and reproducible comparisons.
The new certified convex API uses Eigen and Boost headers, available on macOS with
`brew install eigen boost` and on Debian/Ubuntu with
`sudo apt-get install libeigen3-dev libboost-dev`.

## Open Directions

- **Better decomposition:** Chazelle–Dobkin optimal decomposition (allowing Steiner points) produces strictly fewer pieces than Greene's, potentially reducing the branching factor. Convex *covers* (whose union covers the polygon, with possible overlaps) may use even fewer pieces; no implemented algorithm for optimal covers exists.
- **Geometric pruning:** The last-step map structure used in the convex case reveals, for a given configuration of surrounding polygons, which parts of a polygon's boundary can actually be reached by an optimal path. Extending this reasoning to non-convex polygons could identify cases where, despite many convex pieces, the geometry forces the optimal path through a specific edge — collapsing a polygon to a single piece without branching.
- **VRP extension:** Generalizing from a single path to a vehicle routing setting, in collaboration with Leandro Coelho (Université Laval), as part of a planned research visit.

---

## Requirements

**C++ (convex and non-convex solvers):**
- C++23
- CMake
- CGAL (for convex decomposition in the non-convex solver)
- Gurobi (optional, for MILP baseline)

The root `CMakePresets.json` exposes separate `convex-release` and
`nonconvex-release` presets so IDEs can configure one solver at a time.
Configure the convex preset with `-DTARGET=main-storage_benchmark` to compare
the convenience API against reusable dynamic and static workspaces.
Configure the non-convex preset with `-DTARGET=main-bnb_workload_benchmark` to
run a bounded Branch-and-Bound-like convex-call workload generated from real
CGAL decompositions of the non-convex test case files.

For high-volume callers such as Branch and Bound, the convex C++ solver exposes
workspace overloads that avoid per-call storage allocation. Use
`DynamicConvexTppWorkspace` when the memory should live on the heap and
`StaticConvexTppWorkspace<MaxPolygons, MaxTotalVertices>` when the caller can
provide a fixed stack-capacity workspace.

**Python (prototypes and visualizations):**
- Python 3.12+
- `matplotlib`
- `pygame` (interactive tools)

---

## Bibliography

- Dror, M., Efrat, A., Lubiw, A., Mitchell, J. S. B. (2003). *Touring a sequence of polygons.* STOC 2003. https://doi.org/10.1145/780542.780612
- Tan, X., Jiang, B. (2017). *Efficient algorithms for touring a sequence of convex polygons and related problems.* TAMC 2017. https://doi.org/10.1007/978-3-319-55911-7_44
- Arkin, E. M., Fekete, S. P., Mitchell, J. S. B. (2005). *The Traveling Salesman Problem with Neighborhoods: A Survey.* In: The Traveling Salesman Problem and Its Variations, Springer.
