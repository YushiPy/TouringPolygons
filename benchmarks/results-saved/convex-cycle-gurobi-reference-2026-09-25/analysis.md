# Results and limitations

All four SOCP instances finished with Gurobi status `OPTIMAL` (status 2). The
reported best bound equals the reported objective in every record. Objectives
are in the same units as the input coordinates.

| Instance | Polygons | Gurobi objective / bound | Contracted feasible length |
| --- | ---: | ---: | ---: |
| `two_oblique_triangles` | 2 | 7.211102993647369 | 7.211134426753929 |
| `three_asymmetric_boxes` | 3 | 10.225600555607482 | 10.225637574383779 |
| `four_asymmetric_boxes` | 4 | 15.329642992072589 | 15.329685771268224 |
| `five_scattered_quadrilaterals` | 5 | 17.580031154849344 | 17.580067814188496 |

The C++ cycle certificate checker accepted all four `feasible_contacts` as
feasible and returned a primal-dual gap below `1e-4` for each. It independently
certifies the candidate's bounds; the Gurobi optimum itself remains a
floating-point solver result rather than an exact proof. The small difference
between each feasible length and Gurobi's objective comes from the `1e-5`
contraction toward polygon centroids used to remove boundary-rounding issues.

These references cover only pairwise disjoint polygons and fixed cyclic order.
They are useful for checking objective values on small instances. Contact
coordinates are not a reliable equality target: a problem can have several
optimal contact tuples, and a small objective difference is the meaningful
comparison.
