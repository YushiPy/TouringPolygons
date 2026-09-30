# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Valid O/F | Timeouts O/F | Objective spread | Interval separation | Matched speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OSM | 25-40 | large_osm_25-40_sao_paulo_n040_seed7537 | 39 | 1680.259 / 868 / 1 | 2730.953 / 3463 / 0 | 1 / 1 | 0 / 0 | 1.9948629415011965e-08 | 0 | None |
| OSM | 41-60 | large_osm_41-60_toronto_n050_seed4504 | 50 | 5000.915 / 5734 / 0 | 5010.477 / 4967 / 0 | 1 / 1 | 0 / 0 | 3.5553085377714524 | 0 | None |
| random | 25-40 | large_random_25-40_random_mixed_square_n030_seed30019 | 30 | 266.200 / 267 / 1 | 508.590 / 612 / 1 | 1 / 1 | 0 / 0 | 2.644924279593397e-10 | 2.644497953951941e-10 | 1.9105538107589548 |
| random | 41-60 | large_random_41-60_random_mixed_square_n060_seed60012 | 59 | 5008.023 / 3029 / 0 | 5019.629 / 3147 / 0 | 1 / 1 | 0 / 0 | 2.127721342538848 | 0 | None |
| tessellation | 25-40 | large_tessellation_25-40_uniform-0000030-1 | 30 | 40.583 / 85 / 1 | 99.392 / 175 / 1 | 1 / 1 | 0 / 0 | 6.777554517611861e-09 | 0 | 2.4490839761224152 |
| tessellation | 41-60 | large_tessellation_41-60_uniform-0000060-1 | 60 | 5014.305 / 1371 / 0 | 5005.172 / 4658 / 0 | 1 / 1 | 0 / 0 | 392.21741806191494 | 0 | None |

Per-class and size-band aggregates are in strata.json. Speedups require all repetitions of both backends to pass validation and close the matched gap. Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
