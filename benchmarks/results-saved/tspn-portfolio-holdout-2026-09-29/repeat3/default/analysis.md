# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Valid O/F | Timeouts O/F | Objective spread | Interval separation | Matched speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OSM | 5-10 | holdout_osm_5-10_jakarta_n005_seed7447 | 5 | 0.766 / 3 / 3 | 1.932 / 4 / 3 | 3 / 3 | 0 / 0 | 1.0218883517154609e-09 | 0 | 2.5233966699314396 |
| OSM | 11-20 | holdout_osm_11-20_lagos_n015_seed7082 | 15 | 5.567 / 8 / 3 | 9.258 / 13 / 3 | 3 / 3 | 0 / 0 | 4.384259000289603e-08 | 0 | 1.6631411120093416 |
| random | 5-10 | holdout_random_5-10_random_mixed_square_n010_seed10013 | 10 | 2.558 / 8 / 3 | 13.831 / 24 / 3 | 3 / 3 | 0 / 0 | 1.7302568267041352e-09 | 0 | 5.406926114151681 |
| random | 11-20 | holdout_random_11-20_random_mixed_square_n015_seed15016 | 15 | 38.543 / 17 / 3 | 86.930 / 80 / 3 | 3 / 3 | 0 / 0 | 1.2783800684701419e-09 | 0 | 2.2554228364001974 |
| tessellation | 5-10 | holdout_tessellation_5-10_uniform-0000010-1 | 10 | 2.918 / 8 / 3 | 10.275 / 18 / 0 | 3 / 3 | 0 / 0 | 1.2755663192365319e-10 | 0 | None |
| tessellation | 11-20 | holdout_tessellation_11-20_us-night-0000020.instance | 20 | 99.260 / 78 / 3 | 81.121 / 130 / 3 | 3 / 3 | 0 / 0 | 1.7149432096630335e-08 | 0 | 0.8172656017811897 |

Per-class and size-band aggregates are in strata.json. Speedups require all repetitions of both backends to pass validation and close the matched gap. Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
