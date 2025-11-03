
# Touring Polygons

The goal of this repository is to explore the Touring Polygons Problem (TPP), a generalization of the Traveling Salesman Problem (TSP) where instead of visiting a set of points, we must visit a set of polygons.

We will store various algorithms that solve different variations of the TPP, as well as helper scripts for visualizing and testing these algorithms.

## Bibliography

The main reference for this project is the paper "Touring a Sequence of Polygons" by Joseph S. B. Mitchell, which can be found [here](https://www.researchgate.net/publication/2836312_Touring_a_Sequence_of_Polygons).

We will implement the algorithms described in this paper, as well as explore other variations of the problem, which were left as open problems in the paper.

## Problem Statement

Given a starting point $A$ and end point $B$, along with a sequence of polygons $P_1, P_2, \dots, P_k$, find the shortest path from $A$ to $B$ that visits each polygon at least once. The polygons must be traversed in the order they are given, the path may also cross itself and the polygons.

We may also be given fences $F_0, F_1, \dots, F_{k}$ such that the polygons $P_i$ and $P_{i + 1}$ are contained within the fence $F_i$. In this case, we assume that $P_0 = A$ and $P_{k + 1} = B$, and that each fence $F_i$ is a simple polygon (possibly non-convex).

Given these fences, the path between polygons $P_i$ and $P_{i + 1}$ must be contained within the fence $F_i$.

## The Unconstrained TPP

This is the first variation of the problem we will consider, where there are no fences and the polygons will not intersect. 

## The general TPP

This is the general case of the problem, where there may be fences and the polygons may intersect. We still require the polygons to be convex.

## Considering Non Convex Polygons

It can be shown that when dealing with non convex polygons we get an NP-hard problem. One simple way to solve this problem would be to break each polygon into an union of finitely many convex polygons such as triangles, then we test every combination and pick the one that minimizes the length of the path.

However, this approach is clearly very slow. While we cannot solve the problem in polynomial time, we can try to improve upon this approach by attempting to model the problem as an integer optimization problem with a quadratic target and linear constraints. 

We believe that a Branch and Bound strategy mnay work nicely for this problem, as we can use the convex hull of the non convex polygons as an upper bound for the problem.
