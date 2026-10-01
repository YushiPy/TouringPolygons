# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Valid O/F | Timeouts O/F | Objective spread | Interval separation | Matched speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OSM | 25-40 | large_osm_25-40_sao_paulo_n040_seed7537 | 39 | 2001.788 / 383 / 0 | 2009.086 / 2302 / 0 | 1 / 1 | 0 / 0 | 1.9948515728174243e-08 | 0 | None |
| OSM | 41-60 | large_osm_41-60_toronto_n050_seed4504 | 50 | 2000.979 / 3314 / 0 | 2003.690 / 1413 / 0 | 1 / 1 | 0 / 0 | 15.694307768917838 | 0 | None |
| random | 25-40 | large_random_25-40_random_mixed_square_n030_seed30019 | 30 | 159.580 / 145 / 1 | 462.185 / 612 / 1 | 1 / 1 | 0 / 0 | 2.645208496687701e-10 | 2.6426505428389646e-10 | 2.8962531310405706 |
| random | 41-60 | large_random_41-60_random_mixed_square_n060_seed60012 | 59 | 2112.231 / 930 / 0 | 2099.071 / 463 / 0 | 1 / 0 | 0 / 0 | 0.0 | 0 | None |
| tessellation | 25-40 | large_tessellation_25-40_uniform-0000030-1 | 30 | 23.620 / 42 / 1 | 87.841 / 175 / 1 | 1 / 1 | 0 / 0 | 6.777554517611861e-09 | 0 | 3.7189619758783654 |
| tessellation | 41-60 | large_tessellation_41-60_uniform-0000060-1 | 60 | 2002.006 / 798 / 0 | 2007.751 / 1387 / 0 | 1 / 1 | 0 / 0 | 123.65070330464732 | 0 | None |

Per-class and size-band aggregates are in strata.json. Speedups require all repetitions of both backends to pass validation and close the matched gap. Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
