# TSPN: maintained B&B versus Fekete SOCP B&B

Native-call medians in milliseconds; see config.json for matched gap, validation tolerance and timing scope.

| Instance | k | Ours ms | Fekete ms | Ours valid/closed | Fekete valid/closed | Objective spread | Interval separation |
|---|---:|---:|---:|---|---|---:|---:|
| two_oblique_triangles | 2 | 0.165 | 0.206 | 5/5 | 5/5 | 5.070770470183561e-10 | 0 |
| three_asymmetric_boxes | 3 | 0.276 | 0.499 | 5/5 | 5/0 | 5.930544944021676e-11 | 0 |
| four_asymmetric_boxes | 4 | 0.515 | 1.078 | 5/5 | 5/5 | 5.275779813018744e-13 | 0 |
| five_scattered_quadrilaterals | 5 | 3.198 | 6.836 | 5/5 | 5/5 | 1.894395751378397e-09 | 0 |
| free_anchor | 2 | 0.169 | 0.610 | 5/5 | 5/5 | 6.503064753360377e-11 | 0 |
| common_point | 3 | 0.021 | 0.199 | 5/5 | 5/0 | 3.394802479182559e-14 | 0 |
| containment | 3 | 0.335 | 0.871 | 5/5 | 5/0 | 1.3144582311497288e-08 | 0 |
| nonconvex_u | 2 | 0.431 | 0.845 | 5/5 | 5/5 | 1.176836406102666e-13 | 0 |
| nonconvex_l | 4 | 0.937 | 0.304 | 5/5 | 5/5 | 6.643574579356937e-12 | 0 |
| seeded_boxes_6 | 6 | 2.518 | 11.394 | 5/5 | 5/5 | 6.124878382252064e-12 | 0 |
| seeded_concave_6 | 6 | 1.171 | 8.093 | 5/5 | 5/5 | 7.193392548288102e-10 | 0 |
| seeded_boxes_8 | 8 | 4.256 | 28.289 | 5/5 | 5/5 | 8.181189059541794e-11 | 0 |
| seeded_concave_8 | 8 | 4.169 | 3.379 | 5/5 | 5/5 | 3.069544618483633e-12 | 0 |
| seeded_boxes_10 | 10 | 11.268 | 48.657 | 5/5 | 5/5 | 5.968558980384842e-13 | 0 |
| seeded_concave_10 | 10 | 19.261 | 27.948 | 5/5 | 5/5 | 3.296918293926865e-12 | 0 |
| seeded_boxes_12 | 12 | 2.799 | 16.179 | 5/5 | 5/5 | 3.225864020350855e-12 | 0 |
| seeded_concave_12 | 12 | 11.932 | 53.049 | 5/5 | 5/5 | 2.212630079156952e-11 | 0 |
| german_1_n5 | 5 | 7.143 | 2.346 | 5/5 | 5/5 | 6.072298219805816e-10 | 0 |
| german_2_n5 | 5 | 2.292 | 0.287 | 5/5 | 5/0 | 1.972466634470038e-11 | 0 |
| german_3_n10 | 10 | 15.679 | 9.649 | 5/5 | 5/0 | 6.0254023992456496e-12 | 0 |
| german_4_n10 | 10 | 2.050 | 2.267 | 5/5 | 5/5 | 0.00011796010224429665 | 0 |
| german_5_n15 | 15 | 44.532 | 43.229 | 5/5 | 5/5 | 9.265903599953162e-09 | 0 |
| german_6_n15 | 15 | 107.746 | 47.326 | 5/5 | 5/5 | 1.9184653865522705e-11 | 0 |
| german_7_n20 | 20 | 85.806 | 141.671 | 5/5 | 5/0 | 2.802380549837835e-11 | 0 |
| german_8_n20 | 20 | 60.778 | 132.752 | 5/5 | 5/0 | 4.284061105863657e-09 | 0 |
| german_17_n5 | 5 | 1.418 | 0.986 | 5/5 | 5/5 | 1.9418905594648095e-08 | 0 |
| german_18_n5 | 5 | 5.449 | 4.622 | 5/5 | 5/5 | 1.0944347650365671e-09 | 0 |
| german_19_n10 | 10 | 5.525 | 7.220 | 5/5 | 5/5 | 6.409095476556104e-11 | 0 |
| german_20_n10 | 10 | 5.106 | 3.459 | 5/5 | 5/5 | 2.1677237782569136e-10 | 0 |
| german_22_n15 | 15 | 14.149 | 33.859 | 5/5 | 5/5 | 9.833911462919787e-11 | 0 |
| german_37_n15 | 15 | 5.841 | 43.942 | 5/5 | 5/5 | 2.874969595723087e-09 | 0 |
| german_23_n20 | 20 | 358.452 | 407.489 | 5/5 | 5/5 | 7.846665539545938e-10 | 0 |
| german_24_n20 | 20 | 320.294 | 341.449 | 5/5 | 5/0 | 1.4220802313502645e-08 | 0 |

Counts refer to repetitions. A numerical bound from the external solver is not an exact certificate. Raw tour validation and claimed optimality are reported separately. Timed-out searches remain in the table; their times are not times to optimality.
