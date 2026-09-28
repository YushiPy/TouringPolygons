# Certified cycle comparison

Medians in milliseconds. See config.json for timing scope and solver parameters.

| Instance | Rational | Double | Gurobi total | Gurobi optimize | Exact / feasible / intervals | Rational recoveries |
|---|---:|---:|---:|---:|---|---:|
| two_oblique_triangles | 0.0186 | 0.0145 | 0.3481 | 0.3228 | True / True / True | 0 |
| three_asymmetric_boxes | 0.0716 | 0.0403 | 0.3458 | 0.3130 | True / True / True | 0 |
| four_asymmetric_boxes | 0.0871 | 0.0559 | 0.3900 | 0.3541 | True / True / True | 0 |
| five_scattered_quadrilaterals | 0.3319 | 0.3980 | 0.5253 | 0.4749 | True / True / True | 1 |
| overlap_zero | 0.0161 | 0.0115 | 0.2364 | 0.2110 | True / True / True | 0 |
| nested_positive | 0.0515 | 0.0362 | 0.2971 | 0.2670 | True / True / True | 0 |
| subunit_zero_link | 0.0626 | 0.0391 | 0.3502 | 0.3159 | True / True / True | 0 |
| overlap_positive | 0.0595 | 0.0383 | 0.3491 | 0.3140 | True / True / True | 0 |
| nonsmooth_anchor_orthic | 0.0920 | 0.1717 | 0.3387 | 0.3071 | True / True / True | 2 |
| nonrepresentable_common_point | 0.0168 | 0.0438 | 0.3548 | 0.3259 | True / True / True | 1 |
| ring_boxes_8 | 0.2805 | 0.3024 | 0.5139 | 0.4570 | True / True / True | 1 |
| ring_slanted_8 | 0.0912 | 0.0643 | 0.6338 | 0.5739 | True / True / True | 0 |
| ring_boxes_16 | 0.5268 | 0.5810 | 0.8613 | 0.7639 | True / True / True | 1 |
| ring_slanted_16 | 0.1722 | 0.1278 | 1.0808 | 0.9830 | True / True / True | 0 |
| ring_boxes_32 | 1.1000 | 1.2662 | 1.5722 | 1.3812 | True / True / True | 1 |
| ring_slanted_32 | 0.3469 | 0.2492 | 2.2250 | 2.0330 | True / True / True | 0 |
| ring_boxes_64 | 2.3264 | 2.7533 | 2.9771 | 2.5952 | True / True / True | 1 |
| ring_slanted_64 | 0.6557 | 0.4829 | 3.9975 | 3.6480 | True / True / True | 0 |
| many_edges_16 | 0.2121 | 0.1297 | 0.9123 | 0.8321 | True / True / True | 0 |
| many_edges_64 | 0.7353 | 0.4469 | 2.7143 | 2.4810 | True / True / True | 0 |

Gurobi objective/bound are numerical; independently certified intervals use exactly feasible rational contractions of its contacts. No coordinate equality is required.

Double uses the default exact recovery option. Recovery counts are per call; raw data distinguish feature, anchor and complete-cycle recovery. Validation and certificates use exact predicates in both modes.

These finite synthetic instances do not establish universal speed dominance or prove coverage of all intersection degeneracies.
