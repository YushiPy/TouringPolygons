# TSPN: maintained B&B versus Fekete SOCP B&B

Native-call medians in milliseconds; see config.json for matched gap, validation tolerance and timing scope.

| Instance | k | Ours ms | Fekete ms | Ours valid/closed | Fekete valid/closed | Objective spread | Interval separation |
|---|---:|---:|---:|---|---|---:|---:|
| two_oblique_triangles | 2 | 0.093 | 0.147 | 3/3 | 3/3 | 5.070770470183561e-10 | 0 |
| three_asymmetric_boxes | 3 | 0.253 | 0.337 | 3/3 | 3/0 | 5.930544944021676e-11 | 0 |
| four_asymmetric_boxes | 4 | 0.381 | 0.796 | 3/3 | 3/3 | 5.275779813018744e-13 | 0 |
| five_scattered_quadrilaterals | 5 | 2.083 | 4.683 | 3/3 | 3/3 | 1.894395751378397e-09 | 0 |
| free_anchor | 2 | 0.110 | 0.489 | 3/3 | 3/3 | 6.503064753360377e-11 | 0 |
| common_point | 3 | 0.018 | 0.139 | 3/3 | 3/0 | 3.394802479182559e-14 | 0 |
| containment | 3 | 0.221 | 0.522 | 3/3 | 3/0 | 1.3144582311497288e-08 | 0 |
| nonconvex_u | 2 | 0.242 | 0.561 | 3/3 | 3/3 | 1.176836406102666e-13 | 0 |
| nonconvex_l | 4 | 0.603 | 0.312 | 3/3 | 3/3 | 6.643574579356937e-12 | 0 |
| seeded_boxes_6 | 6 | 1.644 | 8.116 | 3/3 | 3/3 | 6.124878382252064e-12 | 0 |
| seeded_concave_6 | 6 | 0.759 | 5.323 | 3/3 | 3/3 | 7.193392548288102e-10 | 0 |
| seeded_boxes_8 | 8 | 2.821 | 19.661 | 3/3 | 3/3 | 8.181189059541794e-11 | 0 |
| seeded_concave_8 | 8 | 2.789 | 2.316 | 3/3 | 3/3 | 3.069544618483633e-12 | 0 |
| seeded_boxes_10 | 10 | 7.801 | 32.017 | 3/3 | 3/3 | 5.968558980384842e-13 | 0 |
| seeded_concave_10 | 10 | 11.384 | 17.909 | 3/3 | 3/3 | 3.296918293926865e-12 | 0 |
| seeded_boxes_12 | 12 | 1.818 | 11.241 | 3/3 | 3/3 | 3.225864020350855e-12 | 0 |
| seeded_concave_12 | 12 | 7.668 | 36.605 | 3/3 | 3/3 | 2.212630079156952e-11 | 0 |
| german_1_n5 | 5 | 4.715 | 1.589 | 3/3 | 3/3 | 6.072298219805816e-10 | 0 |
| german_2_n5 | 5 | 215.438 | 0.225 | 3/3 | 3/0 | 1.972466634470038e-11 | 0 |
| german_3_n10 | 10 | 19429.147 | 6.648 | 3/3 | 3/0 | 5.9969806898152456e-12 | 0 |
| german_4_n10 | 10 | 1.461 | 1.630 | 3/3 | 3/3 | 0.00011796010224429665 | 0 |
| german_5_n15 | 15 | 44.346 | 29.496 | 3/3 | 3/3 | 9.265903599953162e-09 | 0 |
| german_6_n15 | 15 | 4747.261 | 31.085 | 3/0 | 3/3 | 11.193658295267767 | 0 |
| german_7_n20 | 20 | 77.086 | 90.676 | 3/3 | 3/0 | 2.802380549837835e-11 | 0 |
| german_8_n20 | 20 | 38.874 | 86.870 | 3/3 | 3/0 | 4.284061105863657e-09 | 0 |

Counts refer to repetitions. A numerical bound from the external solver is not an exact certificate. Raw tour validation and claimed optimality are reported separately. Timed-out searches remain in the table; their times are not times to optimality.
