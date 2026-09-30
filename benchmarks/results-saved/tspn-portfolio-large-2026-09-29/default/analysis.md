# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Valid O/F | Timeouts O/F | Objective spread | Interval separation | Matched speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OSM | 25-40 | large_osm_25-40_sao_paulo_n040_seed7537 | 39 | 4977.343 / 747 / 1 | 2720.349 / 3463 / 0 | 1 / 1 | 0 / 0 | 1.9948515728174243e-08 | 0 | None |
| OSM | 41-60 | large_osm_41-60_toronto_n050_seed4504 | 50 | 5000.404 / 4407 / 0 | 5009.300 / 4992 / 0 | 1 / 1 | 0 / 0 | 3.5553085377714524 | 0 | None |
| random | 25-40 | large_random_25-40_random_mixed_square_n030_seed30019 | 30 | 233.704 / 145 / 1 | 508.539 / 612 / 1 | 1 / 1 | 0 / 0 | 2.644924279593397e-10 | 2.644497953951941e-10 | 2.175992226900524 |
| random | 41-60 | large_random_41-60_random_mixed_square_n060_seed60012 | 59 | 5003.147 / 1460 / 0 | 5020.138 / 3114 / 0 | 1 / 1 | 0 / 0 | 3.8096573702285355 | 0 | None |
| tessellation | 25-40 | large_tessellation_25-40_uniform-0000030-1 | 30 | 34.840 / 42 / 1 | 94.838 / 175 / 1 | 1 / 1 | 0 / 0 | 6.777554517611861e-09 | 0 | 2.722100906740856 |
| tessellation | 41-60 | large_tessellation_41-60_uniform-0000060-1 | 60 | 5000.379 / 830 / 0 | 5004.990 / 4748 / 0 | 1 / 1 | 0 / 0 | 272.24412913988635 | 0 | None |

Per-class and size-band aggregates are in strata.json. Speedups require all repetitions of both backends to pass validation and close the matched gap. Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
