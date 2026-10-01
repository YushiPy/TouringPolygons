# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Valid O/F | Timeouts O/F | Objective spread | Interval separation | Matched speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OSM | 5-10 | holdout_osm_5-10_jakarta_n005_seed7447 | 5 | 0.289 / 2 / 1 | 2.810 / 4 / 1 | 1 / 1 | 0 / 0 | 1.0218883517154609e-09 | 0 | 9.730103554649226 |
| OSM | 11-20 | holdout_osm_11-20_lagos_n015_seed7082 | 15 | 2.044 / 4 / 1 | 12.679 / 13 / 1 | 1 / 1 | 0 / 0 | 4.384264684631489e-08 | 0 | 6.202969999134068 |
| random | 5-10 | holdout_random_5-10_random_mixed_square_n010_seed10013 | 10 | 1.888 / 8 / 1 | 13.686 / 24 / 1 | 1 / 1 | 0 / 0 | 1.73024261584942e-09 | 0 | 7.248062270030271 |
| random | 11-20 | holdout_random_11-20_random_mixed_square_n015_seed15016 | 15 | 36.867 / 16 / 1 | 79.722 / 80 / 1 | 1 / 1 | 0 / 0 | 1.2784084901795723e-09 | 0 | 2.1624390174039605 |
| tessellation | 5-10 | holdout_tessellation_5-10_uniform-0000010-1 | 10 | 5.024 / 15 / 1 | 9.276 / 18 / 0 | 1 / 1 | 0 / 0 | 1.2755663192365319e-10 | 0 | None |
| tessellation | 11-20 | holdout_tessellation_11-20_us-night-0000020.instance | 20 | 68.757 / 84 / 1 | 75.735 / 130 / 1 | 1 / 1 | 0 / 0 | 1.7149432096630335e-08 | 0 | 1.1014839660278615 |

Per-class and size-band aggregates are in strata.json. Speedups require all repetitions of both backends to pass validation and close the matched gap. Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
