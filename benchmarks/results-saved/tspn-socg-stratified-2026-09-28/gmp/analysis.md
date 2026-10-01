# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Timeouts O/F |
|---|---:|---|---:|---:|---:|---:|
| OSM | 5-10 | osm_5-10_bangalore_n005_seed1851 | 5 | 5.268 / 4 / 1 | 3.936 / 4 / 1 | 0 / 0 |
| OSM | 11-20 | osm_11-20_bangalore_n015_seed2483 | 15 | 19.364 / 20 / 1 | 36.019 / 51 / 1 | 0 / 0 |
| OSM | 21-40 | osm_21-40_bangalore_n030_seed1038 | 30 | 154.993 / 99 / 1 | 300.296 / 193 / 0 | 0 / 0 |
| OSM | 41-60 | osm_41-60_bangalore_n050_seed1049 | 50 | 2001.751 / 1011 / 0 | 2015.336 / 1479 / 0 | 0 / 0 |
| random | 5-10 | random_5-10_random_mixed_square_n005_seed5000 | 5 | 1.667 / 3 / 1 | 3.341 / 8 / 1 | 0 / 0 |
| random | 11-20 | random_11-20_random_mixed_square_n015_seed15000 | 15 | 14.435 / 15 / 1 | 26.300 / 26 / 0 | 0 / 0 |
| random | 21-40 | random_21-40_random_mixed_square_n030_seed30000 | 30 | 2304.254 / 48 / 0 | 539.765 / 635 / 0 | 0 / 0 |
| random | 41-60 | random_41-60_random_mixed_square_n050_seed50000 | 48 | — / — / 0 | 2011.453 / 948 / 0 | 1 / 0 |
| tessellation | 5-10 | tessellation_5-10_euro-night-0000010 | 10 | 1208.896 / 19 / 1 | 9.702 / 22 / 1 | 0 / 0 |
| tessellation | 11-20 | tessellation_11-20_burma14 | 14 | 12.465 / 7 / 1 | 4.735 / 8 / 1 | 0 / 0 |
| tessellation | 21-40 | tessellation_21-40_euro-night-0000025.instance | 25 | 2435.897 / 48 / 0 | 208.390 / 246 / 0 | 0 / 0 |
| tessellation | 41-60 | tessellation_41-60_att48 | 48 | 2924.490 / 87 / 0 | 979.303 / 1189 / 0 | 0 / 0 |

Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
