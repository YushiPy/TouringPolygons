# TSPN: maintained B&B versus Fekete SOCP B&B

Native-call medians in milliseconds; see config.json for matched gap, validation tolerance and timing scope.

| Instance | k | Ours ms | Fekete ms | Ours valid/closed | Fekete valid/closed | Objective spread | Interval separation |
|---|---:|---:|---:|---|---|---:|---:|
| two_oblique_triangles | 2 | 0.102 | 0.197 | 3/3 | 3/3 | 5.070770470183561e-10 | 0 |
| three_asymmetric_boxes | 3 | 0.185 | 0.345 | 3/3 | 3/0 | 5.930544944021676e-11 | 0 |
| four_asymmetric_boxes | 4 | 0.330 | 0.739 | 3/3 | 3/3 | 5.275779813018744e-13 | 0 |
| five_scattered_quadrilaterals | 5 | 2.067 | 4.630 | 3/3 | 3/3 | 1.894395751378397e-09 | 0 |
| free_anchor | 2 | 0.104 | 0.430 | 3/3 | 3/3 | 6.503064753360377e-11 | 0 |
| common_point | 3 | 0.014 | 0.151 | 3/3 | 3/0 | 3.394802479182559e-14 | 0 |
| containment | 3 | 0.235 | 0.554 | 3/3 | 3/0 | 1.3144582311497288e-08 | 0 |
| nonconvex_u | 2 | 0.325 | 0.689 | 3/3 | 3/3 | 1.176836406102666e-13 | 0 |
| nonconvex_l | 4 | 0.621 | 0.220 | 3/3 | 3/3 | 6.643574579356937e-12 | 0 |
| seeded_boxes_6 | 6 | 1.654 | 7.650 | 3/3 | 3/3 | 6.124878382252064e-12 | 0 |
| seeded_concave_6 | 6 | 0.752 | 5.264 | 3/3 | 3/3 | 7.193392548288102e-10 | 0 |
| seeded_boxes_8 | 8 | 2.739 | 19.227 | 3/3 | 3/3 | 8.181189059541794e-11 | 0 |
| seeded_concave_8 | 8 | 2.778 | 2.304 | 3/3 | 3/3 | 3.069544618483633e-12 | 0 |
| seeded_boxes_10 | 10 | 7.727 | 31.880 | 3/3 | 3/3 | 5.968558980384842e-13 | 0 |
| seeded_concave_10 | 10 | 11.771 | 17.653 | 3/3 | 3/3 | 3.296918293926865e-12 | 0 |
| seeded_boxes_12 | 12 | 1.894 | 11.161 | 3/3 | 3/3 | 3.225864020350855e-12 | 0 |
| seeded_concave_12 | 12 | 7.747 | 35.396 | 3/3 | 3/3 | 2.212630079156952e-11 | 0 |
| german_1_n5 | 5 | 4.357 | 1.743 | 3/3 | 3/3 | 6.072298219805816e-10 | 0 |
| german_2_n5 | 5 | 24.015 | 0.212 | 3/3 | 3/0 | 1.972466634470038e-11 | 0 |
| german_3_n10 | 10 | 19025.976 | 6.580 | 3/3 | 3/0 | 5.9969806898152456e-12 | 0 |
| german_4_n10 | 10 | 1.482 | 1.619 | 3/3 | 3/3 | 0.00011796010224429665 | 0 |
| german_5_n15 | 15 | 43.696 | 28.799 | 3/3 | 3/3 | 9.265903599953162e-09 | 0 |
| german_6_n15 | 15 | 4619.092 | 31.351 | 3/0 | 3/3 | 11.193658295267767 | 0 |
| german_7_n20 | 20 | 77.966 | 91.786 | 3/3 | 3/0 | 2.802380549837835e-11 | 0 |
| german_8_n20 | 20 | 39.950 | 87.488 | 3/3 | 3/0 | 4.284061105863657e-09 | 0 |

Counts refer to repetitions. A numerical bound from the external solver is not an exact certificate. Raw tour validation and claimed optimality are reported separately. Timed-out searches remain in the table; their times are not times to optimality.
