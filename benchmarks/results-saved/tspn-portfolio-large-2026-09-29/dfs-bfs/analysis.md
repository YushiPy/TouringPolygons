# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Valid O/F | Timeouts O/F | Objective spread | Interval separation | Matched speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OSM | 25-40 | large_osm_25-40_sao_paulo_n040_seed7537 | 39 | 1512.055 / 630 / 1 | 2842.936 / 3463 / 0 | 1 / 1 | 0 / 0 | 1.9948629415011965e-08 | 0 | None |
| OSM | 41-60 | large_osm_41-60_toronto_n050_seed4504 | 50 | 5000.083 / 2150 / 0 | 5006.595 / 4546 / 0 | 1 / 1 | 0 / 0 | 35.61820838023618 | 0 | None |
| random | 25-40 | large_random_25-40_random_mixed_square_n030_seed30019 | 30 | 257.481 / 133 / 1 | 515.836 / 612 / 1 | 1 / 1 | 0 / 0 | 2.644924279593397e-10 | 2.644497953951941e-10 | 2.0033950732239205 |
| random | 41-60 | large_random_41-60_random_mixed_square_n060_seed60012 | 59 | 5000.708 / 1963 / 0 | 5012.213 / 2940 / 0 | 1 / 1 | 0 / 0 | 2.127721342538848 | 0 | None |
| tessellation | 25-40 | large_tessellation_25-40_uniform-0000030-1 | 30 | 37.674 / 44 / 1 | 95.689 / 175 / 1 | 1 / 1 | 0 / 0 | 6.777554517611861e-09 | 0 | 2.539956113546298 |
| tessellation | 41-60 | large_tessellation_41-60_uniform-0000060-1 | 60 | 5017.624 / 592 / 0 | 5008.541 / 4805 / 0 | 1 / 1 | 0 / 0 | 395.8948323311997 | 0 | None |

Per-class and size-band aggregates are in strata.json. Speedups require all repetitions of both backends to pass validation and close the matched gap. Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
