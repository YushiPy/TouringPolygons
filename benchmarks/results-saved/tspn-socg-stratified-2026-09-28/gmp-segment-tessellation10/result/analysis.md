# TSPN: maintained B&B versus Fekete SOCP B&B

Times include only completed solver calls; process-censored runs have no solution time. Native time-limit runs report elapsed solver time but do not imply gap closure. See config.json for matched formulation, gap, tolerances and strict Gurobi settings.

| Class | Band | Instance | k | Ours ms / calls / closed | Fekete ms / calls / closed | Timeouts O/F |
|---|---:|---|---:|---:|---:|---:|
| tessellation | 5-10 | tessellation_5-10_euro-night-0000010 | 10 | 1096.979 / 19 / 1 | 17.824 / 22 / 1 | 0 / 0 |

Calls are B&B relaxations. Gap-closed counts are separate from feasible-tour counts. Bounds from Fekete are numerical, not exact certificates. A time-limited solve can return a feasible tour without closing the requested gap; a process timeout is censored and has no solution time.
