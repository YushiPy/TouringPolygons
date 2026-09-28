# Certified cycle comparison

Medians in milliseconds. See config.json for timing scope and solver parameters.

| Instance | Rational | Double | Gurobi total | Gurobi optimize | Exact / feasible / intervals | Rational recoveries |
|---|---:|---:|---:|---:|---|---:|
| two_oblique_triangles | 0.0171 | 0.0134 | 0.2993 | 0.2759 | True / True / True | 0 |
| three_asymmetric_boxes | 0.0671 | 0.0400 | 0.3515 | 0.3219 | True / True / True | 0 |
| four_asymmetric_boxes | 0.0858 | 0.0526 | 0.4085 | 0.3700 | True / True / True | 0 |
| five_scattered_quadrilaterals | 0.2883 | 0.3613 | 0.4455 | 0.4020 | True / True / True | 1 |
| overlap_zero | 0.0149 | 0.0108 | 0.2225 | 0.1981 | True / True / True | 0 |
| nested_positive | 0.0523 | 0.0358 | 0.3141 | 0.2811 | True / True / True | 0 |
| subunit_zero_link | 0.0622 | 0.0388 | 0.3543 | 0.3181 | True / True / True | 0 |
| overlap_positive | 0.0584 | 0.0381 | 0.3555 | 0.3159 | True / True / True | 0 |
| nonsmooth_anchor_orthic | 0.0915 | 0.1682 | 0.3423 | 0.3099 | True / True / True | 2 |
| nonrepresentable_common_point | 0.0167 | 0.0435 | 0.3564 | 0.3281 | True / True / True | 1 |
| ring_boxes_8 | 0.2767 | 0.3002 | 0.5419 | 0.4799 | True / True / True | 1 |
| ring_slanted_8 | 0.0909 | 0.0634 | 0.6335 | 0.5770 | True / True / True | 0 |
| ring_boxes_16 | 0.5243 | 0.5757 | 0.8983 | 0.7899 | True / True / True | 1 |
| ring_slanted_16 | 0.1730 | 0.1252 | 1.1159 | 1.0161 | True / True / True | 0 |
| ring_boxes_32 | 1.0918 | 1.2530 | 1.5913 | 1.3840 | True / True / True | 1 |
| ring_slanted_32 | 0.3340 | 0.2445 | 2.0842 | 1.8871 | True / True / True | 0 |
| ring_boxes_64 | 2.4042 | 2.8090 | 3.1105 | 2.7211 | True / True / True | 1 |
| ring_slanted_64 | 0.6485 | 0.4783 | 4.0884 | 3.7000 | True / True / True | 0 |
| many_edges_16 | 0.2069 | 0.1290 | 0.9034 | 0.8199 | True / True / True | 0 |
| many_edges_64 | 0.7320 | 0.4426 | 2.7512 | 2.5020 | True / True / True | 0 |

Gurobi objective/bound are numerical; independently certified intervals use exactly feasible rational contractions of its contacts. No coordinate equality is required.

Double uses the default exact recovery option. Recovery counts are per call; raw data distinguish feature, anchor and complete-cycle recovery. Validation and certificates use exact predicates in both modes.

These finite synthetic instances do not establish universal speed dominance or prove coverage of all intersection degeneracies.
