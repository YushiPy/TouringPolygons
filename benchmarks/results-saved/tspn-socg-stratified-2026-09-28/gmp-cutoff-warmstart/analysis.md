# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Timeouts O/F |
|---|---:|---|---:|---:|---:|---:|
| OSM | 5-10 | osm_5-10_bangalore_n005_seed1851 | 5 | 1.689 / 4 / 1 | 2.841 / 4 / 1 | 0 / 0 |
| OSM | 11-20 | osm_11-20_bangalore_n015_seed2483 | 15 | 12.791 / 20 / 1 | 28.089 / 51 / 1 | 0 / 0 |
| OSM | 21-40 | osm_21-40_bangalore_n030_seed1038 | 30 | 102.139 / 99 / 1 | 298.517 / 193 / 0 | 0 / 0 |
| OSM | 41-60 | osm_41-60_bangalore_n050_seed1049 | 50 | 2001.281 / 1514 / 0 | 2026.291 / 1503 / 0 | 0 / 0 |
| random | 5-10 | random_5-10_random_mixed_square_n005_seed5000 | 5 | 1.322 / 3 / 1 | 3.530 / 8 / 1 | 0 / 0 |
| random | 11-20 | random_11-20_random_mixed_square_n015_seed15000 | 15 | 11.503 / 15 / 1 | 26.666 / 26 / 0 | 0 / 0 |
| random | 21-40 | random_21-40_random_mixed_square_n030_seed30000 | 30 | 2282.042 / 48 / 0 | 550.968 / 635 / 0 | 0 / 0 |
| random | 41-60 | random_41-60_random_mixed_square_n050_seed50000 | 48 | — / — / 0 | 2010.546 / 958 / 0 | 1 / 0 |
| tessellation | 5-10 | tessellation_5-10_euro-night-0000010 | 10 | 1156.674 / 19 / 1 | 9.076 / 22 / 1 | 0 / 0 |
| tessellation | 11-20 | tessellation_11-20_burma14 | 14 | 10.711 / 7 / 1 | 4.865 / 8 / 1 | 0 / 0 |
| tessellation | 21-40 | tessellation_21-40_euro-night-0000025.instance | 25 | 1404.775 / 69 / 1 | 209.732 / 246 / 0 | 0 / 0 |
| tessellation | 41-60 | tessellation_41-60_att48 | 48 | 2887.479 / 87 / 0 | 959.555 / 1189 / 0 | 0 / 0 |

Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
