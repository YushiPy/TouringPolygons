# Certified cycle comparison

Medians in milliseconds. See config.json for timing scope and solver parameters.

| Instance | Rational | Double | Gurobi total | Gurobi optimize | Exact / feasible / intervals | Rational recoveries |
|---|---:|---:|---:|---:|---|---:|
| two_oblique_triangles | 0.0177 | 0.0133 | 0.3212 | 0.2949 | True / True / True | 0 |
| three_asymmetric_boxes | 0.0655 | 0.0317 | 0.3367 | 0.3049 | True / True / True | 0 |
| four_asymmetric_boxes | 0.0935 | 0.0483 | 0.3962 | 0.3579 | True / True / True | 0 |
| five_scattered_quadrilaterals | 0.3186 | 0.3981 | 0.5253 | 0.4730 | True / True / True | 1 |
| overlap_zero | 0.0158 | 0.0111 | 0.2369 | 0.2110 | True / True / True | 0 |
| nested_positive | 0.0513 | 0.0234 | 0.3070 | 0.2768 | True / True / True | 0 |
| subunit_zero_link | 0.0629 | 0.0341 | 0.3574 | 0.3221 | True / True / True | 0 |
| overlap_positive | 0.0591 | 0.0311 | 0.3523 | 0.3171 | True / True / True | 0 |
| nonsmooth_anchor_orthic | 4.0060 | 12.4830 | 0.5988 | 0.5419 | True / True / True | 124 |
| nonrepresentable_common_point | 0.0202 | 17.2974 | 0.7124 | 0.6421 | True / True / True | 252 |
| ring_boxes_8 | 0.2740 | 0.2978 | 0.5777 | 0.5159 | True / True / True | 1 |
| ring_slanted_8 | 0.0928 | 0.0626 | 0.6457 | 0.5870 | True / True / True | 0 |
| ring_boxes_16 | 0.5417 | 0.5960 | 0.9593 | 0.8519 | True / True / True | 1 |
| ring_slanted_16 | 0.1745 | 0.1243 | 1.1169 | 1.0190 | True / True / True | 0 |
| ring_boxes_32 | 1.1002 | 1.2512 | 1.5471 | 1.3621 | True / True / True | 1 |
| ring_slanted_32 | 0.3332 | 0.2419 | 2.0635 | 1.8821 | True / True / True | 0 |
| ring_boxes_64 | 2.3430 | 2.7532 | 3.0245 | 2.6610 | True / True / True | 1 |
| ring_slanted_64 | 0.6518 | 0.4783 | 4.0295 | 3.6731 | True / True / True | 0 |
| many_edges_16 | 0.2092 | 0.1247 | 0.8998 | 0.8199 | True / True / True | 0 |
| many_edges_64 | 0.7355 | 0.4272 | 2.7280 | 2.4929 | True / True / True | 0 |

Gurobi objective/bound are numerical; independently certified intervals use exactly feasible rational contractions of its contacts. No coordinate equality is required.

Double uses the default exact recovery option. Recovery counts are per call; raw data distinguish feature, anchor and complete-cycle recovery. Validation and certificates use exact predicates in both modes.

These finite synthetic instances do not establish universal speed dominance or prove coverage of all intersection degeneracies.
