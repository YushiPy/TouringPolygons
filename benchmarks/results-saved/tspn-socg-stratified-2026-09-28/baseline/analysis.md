# TSPN: maintained B&B versus Fekete SOCP B&B

Native-call medians in milliseconds; see config.json for matched gap, validation tolerance and timing scope.

| Instance | k | Ours ms | Fekete ms | Ours valid/closed | Fekete valid/closed | Objective spread | Interval separation |
|---|---:|---:|---:|---|---|---:|---:|
| osm_5-10_bangalore_n005_seed1851 | 5 | 5.259 | 2.736 | 1/1 | 1/1 | 6.072298219805816e-10 | 0 |
| osm_11-20_bangalore_n015_seed2483 | 15 | 28.852 | 32.626 | 1/1 | 1/1 | 9.265903599953162e-09 | 0 |
| osm_21-40_bangalore_n030_seed1038 | 30 | 238.116 | 292.464 | 1/1 | 1/0 | 5.182982931728475e-10 | 0 |
| osm_41-60_bangalore_n050_seed1049 | 50 | 2002.085 | 2014.870 | 1/0 | 1/0 | 21.768875785829778 | 0 |
| random_5-10_random_mixed_square_n005_seed5000 | 5 | 2.539 | 3.425 | 1/1 | 1/1 | 6.503597660412197e-10 | 0 |
| random_11-20_random_mixed_square_n015_seed15000 | 15 | 20.835 | 25.344 | 1/1 | 1/0 | 4.955182930643787e-09 | 0 |
| random_21-40_random_mixed_square_n030_seed30000 | 30 | 3565.695 | 539.504 | 1/0 | 1/0 | 23.503749948453304 | 0 |
| random_41-60_random_mixed_square_n050_seed50000 | 48 | error | 2011.553 | 0/0 | 1/0 | 0.0 | 0 |
| tessellation_5-10_euro-night-0000010 | 10 | 1877.206 | 9.105 | 1/1 | 1/1 | 2.353374111407902e-05 | 0 |
| tessellation_11-20_burma14 | 14 | 19.574 | 4.771 | 1/1 | 1/1 | 3.5562663924793014e-12 | 0 |
| tessellation_21-40_euro-night-0000025.instance | 25 | 2057.748 | 204.937 | 1/0 | 1/0 | 6977.8406051056445 | 0 |
| tessellation_41-60_att48 | 48 | 4357.920 | 965.748 | 1/0 | 1/0 | 13609.543205851594 | 0 |

Counts refer to repetitions. A numerical bound from the external solver is not an exact certificate. Raw tour validation and claimed optimality are reported separately. Timed-out searches remain in the table; their times are not times to optimality.
