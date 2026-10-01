# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Valid O/F | Timeouts O/F | Objective spread | Interval separation | Matched speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OSM | 5-10 | holdout_osm_5-10_jakarta_n005_seed7447 | 5 | 0.872 / 4 / 3 | 1.905 / 4 / 3 | 3 / 3 | 0 / 0 | 1.0218883517154609e-09 | 0 | 2.1857849154000575 |
| OSM | 11-20 | holdout_osm_11-20_lagos_n015_seed7082 | 15 | 3.837 / 9 / 3 | 9.217 / 13 / 3 | 3 / 3 | 0 / 0 | 4.384264684631489e-08 | 0 | 2.402558582040923 |
| random | 5-10 | holdout_random_5-10_random_mixed_square_n010_seed10013 | 10 | 3.919 / 16 / 3 | 13.827 / 24 / 3 | 3 / 3 | 0 / 0 | 1.7302568267041352e-09 | 0 | 3.5277729726916327 |
| random | 11-20 | holdout_random_11-20_random_mixed_square_n015_seed15016 | 15 | 43.323 / 32 / 3 | 87.578 / 80 / 3 | 3 / 3 | 0 / 0 | 1.2784084901795723e-09 | 0 | 2.021510880639524 |
| tessellation | 5-10 | holdout_tessellation_5-10_uniform-0000010-1 | 10 | 3.675 / 14 / 3 | 10.372 / 18 / 0 | 3 / 3 | 0 / 0 | 1.2755663192365319e-10 | 0 | None |
| tessellation | 11-20 | holdout_tessellation_11-20_us-night-0000020.instance | 20 | 104.504 / 126 / 3 | 79.724 / 130 / 3 | 3 / 3 | 0 / 0 | 1.7149432096630335e-08 | 0 | 0.7628724630845028 |

Per-class and size-band aggregates are in strata.json. Speedups require all repetitions of both backends to pass validation and close the matched gap. Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
