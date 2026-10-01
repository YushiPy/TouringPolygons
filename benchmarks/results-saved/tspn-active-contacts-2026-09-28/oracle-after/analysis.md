# Certified cycle comparison

Medians in milliseconds. See config.json for timing scope and solver parameters.

| Instance | Rational | Double | Gurobi total | Gurobi optimize | Exact / feasible / intervals | Rational recoveries |
|---|---:|---:|---:|---:|---|---:|
| german_2_n5_0,4 | 0.3066 | 0.4247 | 0.6543 | 0.6039 | True / True / True | 1 |
| german_2_n5_0,2,4 | 2.4358 | 1.3598 | 0.6335 | 0.5829 | True / True / True | 2 |
| german_3_n10_0,9 | 0.1756 | 0.0796 | 0.5171 | 0.4790 | True / True / True | 0 |
| german_3_n10_0,4,9 | 0.2485 | 0.1279 | 0.6366 | 0.5898 | True / True / True | 0 |
| german_3_n10_0,2,4,9 | 0.9778 | 0.3818 | 0.7315 | 0.6762 | True / True / True | 0 |
| german_3_n10_0,4,9,2 | 0.9789 | 0.7683 | 0.7197 | 0.6649 | True / True / True | 1 |
| german_3_n10_0,4,9,5,2 | 1.2766 | 0.9839 | 0.8594 | 0.7939 | True / True / True | 1 |
| german_3_n10_0,4,8,9,5,2 | 5.4433 | 2.6663 | 1.0390 | 0.9601 | True / True / True | 2 |
| german_3_n10_0,4,9,8,5,2 | 3.0560 | 1.1905 | 0.9915 | 0.9170 | True / True / True | 1 |
| german_3_n10_0,3,4,9,8,5,2 | 19.7857 | 4.0373 | 1.2812 | 1.1830 | True / True / True | 2 |
| german_3_n10_0,4,3,9,8,5,2 | 14.3668 | 3.2652 | 1.1752 | 1.0839 | True / True / True | 2 |
| german_6_n15_0,13 | 0.3263 | 0.4300 | 0.5265 | 0.4871 | True / True / True | 1 |
| german_6_n15_0,8,13 | 1.0032 | 0.7185 | 0.6417 | 0.5941 | True / True / True | 1 |
| german_6_n15_0,7,8,13 | 0.9125 | 0.7675 | 0.7292 | 0.6731 | True / True / True | 1 |
| german_6_n15_0,8,13,7 | 0.5769 | 1.0052 | 0.7423 | 0.6871 | True / True / True | 1 |
| german_6_n15_0,10,7,8,13 | 4.8452 | 2.4010 | 0.9076 | 0.8361 | True / True / True | 2 |

Gurobi objective/bound are numerical; independently certified intervals use exactly feasible rational contractions of its contacts. No coordinate equality is required.

Double uses the default exact recovery option. Recovery counts are per call; raw data distinguish feature, anchor and complete-cycle recovery. Validation and certificates use exact predicates in both modes.

These finite synthetic instances do not establish universal speed dominance or prove coverage of all intersection degeneracies.
