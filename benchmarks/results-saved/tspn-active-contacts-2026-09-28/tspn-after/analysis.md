# TSPN: maintained B&B versus Fekete SOCP B&B

Native-call medians in milliseconds; see config.json for matched gap, validation tolerance and timing scope.

| Instance | k | Ours ms | Fekete ms | Ours valid/closed | Fekete valid/closed | Objective spread | Interval separation |
|---|---:|---:|---:|---|---|---:|---:|
| two_oblique_triangles | 2 | 0.716 | 0.222 | 3/3 | 3/3 | 5.070770470183561e-10 | 0 |
| three_asymmetric_boxes | 3 | 0.699 | 0.524 | 3/3 | 3/0 | 5.930544944021676e-11 | 0 |
| four_asymmetric_boxes | 4 | 0.925 | 1.116 | 3/3 | 3/3 | 5.275779813018744e-13 | 0 |
| five_scattered_quadrilaterals | 5 | 3.546 | 6.769 | 3/3 | 3/3 | 1.894395751378397e-09 | 0 |
| free_anchor | 2 | 0.539 | 0.607 | 3/3 | 3/3 | 6.503064753360377e-11 | 0 |
| common_point | 3 | 0.390 | 0.232 | 3/3 | 3/0 | 3.394802479182559e-14 | 0 |
| containment | 3 | 0.686 | 0.788 | 3/3 | 3/0 | 1.3144582311497288e-08 | 0 |
| nonconvex_u | 2 | 0.849 | 0.902 | 3/3 | 3/3 | 1.176836406102666e-13 | 0 |
| nonconvex_l | 4 | 1.298 | 0.340 | 3/3 | 3/3 | 6.643574579356937e-12 | 0 |
| seeded_boxes_6 | 6 | 2.893 | 11.637 | 3/3 | 3/3 | 6.124878382252064e-12 | 0 |
| seeded_concave_6 | 6 | 1.558 | 7.673 | 3/3 | 3/3 | 7.193392548288102e-10 | 0 |
| seeded_boxes_8 | 8 | 4.545 | 28.168 | 3/3 | 3/3 | 8.181189059541794e-11 | 0 |
| seeded_concave_8 | 8 | 4.520 | 3.308 | 3/3 | 3/3 | 3.069544618483633e-12 | 0 |
| seeded_boxes_10 | 10 | 12.070 | 47.327 | 3/3 | 3/3 | 5.968558980384842e-13 | 0 |
| seeded_concave_10 | 10 | 19.641 | 26.384 | 3/3 | 3/3 | 3.296918293926865e-12 | 0 |
| seeded_boxes_12 | 12 | 3.217 | 16.574 | 3/3 | 3/3 | 3.225864020350855e-12 | 0 |
| seeded_concave_12 | 12 | 12.429 | 53.062 | 3/3 | 3/3 | 2.212630079156952e-11 | 0 |
| german_1_n5 | 5 | 7.606 | 2.510 | 3/3 | 3/3 | 6.072298219805816e-10 | 0 |
| german_2_n5 | 5 | 2.656 | 0.322 | 3/3 | 3/0 | 1.972466634470038e-11 | 0 |
| german_3_n10 | 10 | 16.219 | 9.667 | 3/3 | 3/0 | 6.0254023992456496e-12 | 0 |
| german_4_n10 | 10 | 2.436 | 2.458 | 3/3 | 3/3 | 0.00011796010224429665 | 0 |
| german_5_n15 | 15 | 44.954 | 43.153 | 3/3 | 3/3 | 9.265903599953162e-09 | 0 |
| german_6_n15 | 15 | 107.530 | 48.150 | 3/3 | 3/3 | 1.9184653865522705e-11 | 0 |
| german_7_n20 | 20 | 86.818 | 137.686 | 3/3 | 3/0 | 2.802380549837835e-11 | 0 |
| german_8_n20 | 20 | 60.219 | 133.578 | 3/3 | 3/0 | 4.284061105863657e-09 | 0 |
| german_17_n5 | 5 | 1.784 | 1.002 | 3/3 | 3/3 | 1.9418905594648095e-08 | 0 |
| german_18_n5 | 5 | 5.746 | 4.675 | 3/3 | 3/3 | 1.0944347650365671e-09 | 0 |
| german_19_n10 | 10 | 5.898 | 7.162 | 3/3 | 3/3 | 6.409095476556104e-11 | 0 |
| german_20_n10 | 10 | 5.424 | 3.726 | 3/3 | 3/3 | 2.1677237782569136e-10 | 0 |
| german_22_n15 | 15 | 14.385 | 34.519 | 3/3 | 3/3 | 9.833911462919787e-11 | 0 |
| german_37_n15 | 15 | 6.213 | 43.735 | 3/3 | 3/3 | 2.874969595723087e-09 | 0 |
| german_23_n20 | 20 | 354.606 | 404.701 | 3/3 | 3/3 | 7.846665539545938e-10 | 0 |
| german_24_n20 | 20 | 323.616 | 338.162 | 3/3 | 3/0 | 1.4220802313502645e-08 | 0 |

Counts refer to repetitions. A numerical bound from the external solver is not an exact certificate. Raw tour validation and claimed optimality are reported separately. Timed-out searches remain in the table; their times are not times to optimality.
