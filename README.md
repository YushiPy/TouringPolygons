
# TouringPolygons

Repository for implementations of the problem of touring polygons

## Problem Statement

Given a starting point $A$ and end point $B$, along with a sequence of polygons $P_1, P_2, \dots, P_k$, find the shortest path from $A$ to $B$ that visits each polygon at least once. The polygons must be traversed in the order they are given, the path may also cross itself and the polygons.

We may also be given fences $F_0, F_1, \dots, F_{k}$ such that the polygons $P_i$ and $P_{i + 1}$ are contained within the fence $F_i$. In this case, we assume that $P_0 = A$ and $P_{k + 1} = B$, and that each fence $F_i$ is a simple polygon (possibly non-convex).

Given these fences, the path between polygons $P_i$ and $P_{i + 1}$ must be contained within the fence $F_i$.

### Variations

We will consider several variations of the problem, including:

- Polygons are convex or non-convex.
- There is an encompassing polygon that must not be crossed.
- The polygons may intersect.
- The problem is in a higher dimension (e.g., 3D polygons).
