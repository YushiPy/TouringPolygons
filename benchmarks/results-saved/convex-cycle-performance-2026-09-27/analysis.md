# Certified cycle comparison

Medians in milliseconds. See config.json for timing scope and solver parameters.

| Instance | Rational | Double | Gurobi total | Gurobi optimize | Exact / feasible / intervals |
|---|---:|---:|---:|---:|---|
| two_oblique_triangles | 0.0177 | 0.0140 | 0.3302 | 0.3049 | True / True / True |
| three_asymmetric_boxes | 0.0687 | 0.0379 | 0.3266 | 0.2952 | True / True / True |
| four_asymmetric_boxes | 0.0842 | 0.0524 | 0.3600 | 0.3240 | True / True / True |
| five_scattered_quadrilaterals | 0.3266 | 0.4707 | 0.5770 | 0.5212 | True / True / True |
| overlap_zero | 0.0160 | 0.0179 | 0.2509 | 0.2241 | True / True / True |
| nested_positive | 0.0535 | 0.0364 | 0.3204 | 0.2890 | True / True / True |
| subunit_zero_link | 0.0757 | 0.0490 | 0.3661 | 0.3300 | True / True / True |
| overlap_positive | 0.0745 | 0.0503 | 0.3859 | 0.3471 | True / True / True |
| ring_boxes_8 | 0.2944 | 0.3233 | 0.6147 | 0.5500 | True / True / True |
| ring_slanted_8 | 0.0918 | 0.0648 | 0.6540 | 0.5920 | True / True / True |
| ring_boxes_16 | 0.5654 | 0.6346 | 0.9175 | 0.8049 | True / True / True |
| ring_slanted_16 | 0.1722 | 0.1252 | 1.1155 | 1.0021 | True / True / True |
| ring_boxes_32 | 1.1609 | 1.3559 | 1.6597 | 1.4608 | True / True / True |
| ring_slanted_32 | 0.3345 | 0.2477 | 2.1137 | 1.9171 | True / True / True |
| ring_boxes_64 | 2.5754 | 3.3726 | 3.0769 | 2.6770 | True / True / True |
| ring_slanted_64 | 0.6648 | 0.4872 | 4.2616 | 3.8440 | True / True / True |
| many_edges_16 | 0.2105 | 0.1303 | 0.9405 | 0.8509 | True / True / True |
| many_edges_64 | 0.7450 | 0.4439 | 2.7976 | 2.5589 | True / True / True |

Gurobi objective/bound are numerical; independently certified intervals use exactly feasible rational contractions of its contacts. No coordinate equality is required.

These finite synthetic instances do not establish universal speed dominance or complete support for all intersection degeneracies.
