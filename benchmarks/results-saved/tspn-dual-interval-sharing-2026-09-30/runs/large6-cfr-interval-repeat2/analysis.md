# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Valid O/F | Timeouts O/F | Objective spread | Interval separation | Matched speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OSM | 25-40 | large_osm_25-40_sao_paulo_n040_seed7537 | 39 | 562.716 / 580 / 1 | 2009.086 / 2302 / 0 | 1 / 1 | 0 / 0 | 1.9948629415011965e-08 | 0 | None |
| OSM | 41-60 | large_osm_41-60_toronto_n050_seed4504 | 50 | 2001.033 / 4500 / 0 | 2003.690 / 1413 / 0 | 1 / 1 | 0 / 0 | 21.955837238365802 | 0 | None |
| random | 25-40 | large_random_25-40_random_mixed_square_n030_seed30019 | 30 | 132.855 / 123 / 1 | 462.185 / 612 / 1 | 1 / 1 | 0 / 0 | 2.645208496687701e-10 | 2.637534635141492e-10 | 3.47887590103318 |
| random | 41-60 | large_random_41-60_random_mixed_square_n060_seed60012 | 59 | 2011.601 / 1800 / 0 | 2099.071 / 463 / 0 | 1 / 0 | 0 / 0 | 0.0 | 0 | None |
| tessellation | 25-40 | large_tessellation_25-40_uniform-0000030-1 | 30 | 19.065 / 44 / 1 | 87.841 / 175 / 1 | 1 / 1 | 0 / 0 | 6.777554517611861e-09 | 0 | 4.607540645259293 |
| tessellation | 41-60 | large_tessellation_41-60_uniform-0000060-1 | 60 | 2000.676 / 1239 / 0 | 2007.751 / 1387 / 0 | 1 / 1 | 0 / 0 | 389.47687569564914 | 0 | None |

Per-class and size-band aggregates are in strata.json. Speedups require all repetitions of both backends to pass validation and close the matched gap. Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
