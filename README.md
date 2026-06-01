# Touring Polygons Problem

This repository contains implementations and experiments for the **Touring Polygons Problem (TPP)**: given a starting point $s$, an ending point $t$, and an ordered sequence of polygons $P_1, \dots, P_k$ in the plane, find the shortest Euclidean path from $s$ to $t$ that visits each polygon in order. The path may touch the boundary of a polygon or pass through its interior.

The problem is introduced in [Dror et al. (2003)](#bibliography) and can be seen as a special case of the Traveling Salesman Problem with Neighborhoods (TSPN), where the regions are polygons and the visit order is fixed.

An interactive visualizer for the convex case is available [here](https://yushipy.github.io/PersonalPage/Visualizer/index.html) (desktop only).

---

## Repository Structure

```
src/
├── UnconstrainedTPP/     # Convex TPP: Python prototypes and C++ implementations
├── NonConvexTPP/         # Non-convex TPP: Branch and Bound, MILP, decomposition
├── FencedTPP/            # Fenced TPP (legacy, deprioritized)
└── Canvas/               # Web-based interactive visualizer
Relatorio/                # LaTeX report (Portuguese)
```

---

## The Convex TPP

When all polygons are convex and disjoint, the problem is solvable in polynomial time. All implementations are in C++ (`src/UnconstrainedTPP/C++`) and share the same algorithmic framework based on two geometric structures:

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

**Python (prototypes and visualizations):**
- Python 3.12+
- `matplotlib`
- `pygame` (interactive tools)

---

## Bibliography

- Dror, M., Efrat, A., Lubiw, A., Mitchell, J. S. B. (2003). *Touring a sequence of polygons.* STOC 2003. https://doi.org/10.1145/780542.780612
- Tan, X., Jiang, B. (2017). *Efficient algorithms for touring a sequence of convex polygons and related problems.* TAMC 2017. https://doi.org/10.1007/978-3-319-55911-7_44
- Arkin, E. M., Fekete, S. P., Mitchell, J. S. B. (2005). *The Traveling Salesman Problem with Neighborhoods: A Survey.* In: The Traveling Salesman Problem and Its Variations, Springer.