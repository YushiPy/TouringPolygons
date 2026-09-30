# TSPN portfolio large holdout — 2026-09-29

Six cases were selected before timing, one for each class (`OSM`, `random`, `tessellation`) and actual simplified polygon-count band 25–40 and 41–60. Selection used seed `tpp-portfolio-large-2026-09-29-v1`, excluded all 18 prior screening/holdout cases, and excluded Bangalore OSM entries. Actual counts are 39, 50, 30, 59, 30 and 60. The three reserved candidates were not measured.

The four modes ran sequentially through `benchmarks/tpp.py tspn-benchmark` with the same frozen binaries: default, isolated `dfs-bfs`, independent two-worker race, and cooperative two-worker portfolio. Native and Fekete each received a nominal 5-second solve budget; the per-process timeout was 12 seconds. Native relative gap was `1e-6/(1+1e-6)` with zero absolute gap; Fekete used `1e-6`. Feasibility and independent validation tolerances were `1e-8` and `1e-7`. Fekete used one Gurobi thread; portfolio modes used two native search workers. No builds or CPU affinity pinning were used. Solver calls can overrun the nominal budget while an oracle call completes: observed runs were about 5.00–5.02 seconds, all below the process timeout.

Every tour passed the tolerance-based independent validator. Native closed the gap on three of six primary cases in each mode; Fekete closed it on two of six. Time-to-gap comparisons therefore include only the mutually closed random k=30 and tessellation k=30 cases. Native was faster on both in every mode; per-case times (native / Fekete, seconds) were:

| Mode | Random k=30 | Tessellation k=30 |
|---|---:|---:|
| Default | 0.234 / 0.509 | 0.035 / 0.095 |
| Isolated dfs-bfs | 0.257 / 0.516 | 0.038 / 0.096 |
| Independent race | 0.266 / 0.509 | 0.041 / 0.099 |
| Cooperative portfolio | 0.279 / 0.508 | 0.039 / 0.096 |

The three larger cases remained open at the nominal 5-second limit. Relative gap is `(UB - LB) / UB`; these are equal-budget screening results, not matched solved speedups. Actual elapsed times varied slightly around 5 seconds.

| Case | Default | Independent race | Cooperative | Fekete |
|---|---:|---:|---:|---:|
| OSM k=50 | 10.96% | 11.39% | 11.34% | 16.25% |
| Random k=59 | 3.77% | 5.33% | 4.06% | 10.74% |
| Tessellation k=60 | 13.78% | 15.02% | 15.02% | 7.88% |

The shared portfolio slightly improved the random k=59 gap over the independent race, while default was better on OSM k=50 and both portfolio modes matched on tessellation k=60. On those cases, per-worker calls (best-bound + dfs-bfs) were: OSM k=50, race 3,843 + 1,891 and cooperative 3,950 + 2,861; random k=59, race 1,374 + 1,655 and cooperative 1,407 + 1,700; tessellation k=60, race 797 + 574 and cooperative 795 + 632. Cooperative sharing recorded 18 incumbent publications and 16 imports across these three open cases. The two-worker totals are work done concurrently, so compare call counts alongside the per-worker times and bounds. These observations do not identify a hardware cause: cache, memory bandwidth, frequency and scheduler effects are shared.

The OSM k=39 case closed natively in 1.51–5.01 seconds, so it received three repetitions per mode with a nominal 10-second limit and a 15-second process cap. Native median times were default 4.998s, dfs-bfs 1.501s, independent race 1.654s and cooperative 1.705s. Fekete was valid but did not close the requested gap in any repetition (median 2.742s; relative gap `5.57e-6`), so it is excluded from matched speedups. Cooperative sharing recorded 12 publications and 8 imports across the three repetitions; maximum join tail was 1.22ms. The primary-screen maximum cooperative join tail was 17.4ms. Per-call oracle duration is not exposed by the current result telemetry.

`selection.json` and `instances.json` preserve inputs and the pre-timing policy; `reserve-selection.json` records the unused reserves. Each mode directory and `repeat3/` contain raw runs, bounds, validation, per-worker telemetry, config and summary. `provenance.json` records binary and source hashes, resource counts and host CPU information. The exact portfolio header is preserved in `source-overlay/` because it was untracked and does not appear in `source.patch`. No executables, builds, caches or license logs are stored in this campaign.
