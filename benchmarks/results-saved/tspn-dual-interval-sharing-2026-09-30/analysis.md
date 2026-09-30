# Dual screening, interval certificates, and portfolio bound sharing

## Scope and configuration

This follow-up measured three opt-in ideas against the same frozen candidate binary. Single-worker CFR means `cache + features + root`; comparisons add `dual-screen` or `interval`. The portfolio tests use two workers and CFR plus `memo`; comparisons add `share-bounds` or `interval`. Every flag is false unless requested. The V1 binary is `.build/tspn-dual-interval-sharing-2026-09-30/candidate-gmp-on/tpp-unordered-frozen` (SHA-256 `6e9724b988fd314c16e17a53a90f87b239e924a1d8419c6300311ec940b1ffcb`). A V2 binary adds only the lazy interned shared-bound index (`be321d7befc4839bd54f76f18acafacc50e57e2b0bd7483ae7e36c608ad28f2d`). Both candidates have source patches and untracked `cycle_interval.h` in `provenance/`.

The initial screen used six holdout inputs with actual region counts 5, 15, 10, 15, 10, and 20, and six larger inputs with actual counts 39, 50, 30, 59, 30, and 60. The input named `large_osm_25-40...n040` has 39 regions; the input named `large_random_41-60...n060` has 59. Single-worker and large6 portfolio screens used nominal 2 seconds per run and an 8-second firm process cap, one repetition. The OSM39 portfolio follow-up used 10 seconds, 15-second firm cap, and three repetitions.

The Fekete rows in every run were reused through `--reference-results`. The CLI validated input hashes and solver settings; no Fekete process was started in this campaign. Native comparisons use the same binary within each pair. Fekete timings remain an external reference and are not mixed into native speedup calculations.

## Correctness and objective audit

The saved raw files contain 96 native rows and 96 reused Fekete rows across 17 run directories and 12 distinct input hashes. All 96 native rows report a valid tour under the campaign validator. Sixty rows closed the gap; the other 36 ended at the requested time limit with a valid incumbent, and none were process timeouts.

`provenance/audit-objectives.py` recomputes cross-run objective consistency and interval containment from the raw files. All closed native objectives for a given input were identical across runs (maximum relative spread 0). Those known objectives fell inside every native interval for their input in 60/60 checks; maximum native relative interval width was `6.12e-15`. The same objectives fell inside the matched external Fekete intervals in 60/60 checks under the declared `validation_tolerance` cross-solver allowance (`1e-7 * max(1, |objective|)`). Fekete intervals remain external numerical bounds, not rational certificates; their maximum relative width across these references was 1.997%, including budget-limited rows. Full details are in `provenance/objective-interval-audit.json`.

All four affected suites passed with GMP on and off. V1 GMP-on passed TSPN (19 exhaustive cases; 798 option/call-cap comparisons; 38 combined concurrency checks), unordered (86 exhaustive-order cases; 344 interrupted-search checks), cycle (96 seeded cases plus boundary/contact regressions and 67 double comparisons), and cycle-certificate (1,040 zero-link ray pairs, 269 optimal). V2's new concurrent bound-index test passed in the GMP-on TSPN suite, and unordered tests passed. The V2 nonconvex tests were rebuilt with GMP off; TSPN, unordered, cycle, and cycle-certificate suites all passed. The configured solver gap target is approximately `1e-6`; feasibility and route validation tolerances are `1e-8` and `1e-7`, respectively. `time_limit` means a valid tour with an open proof gap, never a process timeout or an optimal result.

## Single-worker screen

| Input group / option | Native valid | Closed | Time limit | Closed-case paired speedup vs control | Counters |
|---|---:|---:|---:|---:|---|
| Small6 control (CFR) | 6/6 | 6 | 0 | — | 129 calls, 44 nodes |
| Small6 + dual-screen | 6/6 | 6 | 0 | 1.40x median, 6/6 wins | No child screens fired |
| Small6 + interval | 6/6 | 6 | 0 | 1.55x median, 6/6 wins | 173 interval uses |
| Large6 control (CFR) | 6/6 | 3 | 3 | — | 6,420 calls, 2,276 nodes |
| Large6 + dual-screen | 6/6 | 3 | 3 | 0.99x median, 0/3 wins | 63 child screens, 4 prunes |
| Large6 + interval | 6/6 | 3 | 3 | 1.24x median, 3/3 wins | 16,804 interval uses |

Small6 absolute times were 0.3–47.7 ms, so the one-repetition speedups are preliminary. In case order Jakarta5, Lagos15, random10, random15, tessellation10, US-night tessellation20, CFR control times were 1.07, 3.93, 2.96, 41.25, 4.37, and 47.67 ms; dual-screen was 0.30, 2.02, 1.83, 34.83, 4.26, and 46.30 ms; interval was 0.28, 1.51, 1.63, 32.21, 3.39, and 38.89 ms. Calls and nodes were unchanged across these variants.

The first large6 screen showed lower closed-case times with interval on OSM39 (0.752→0.550 s), random30 (0.155→0.125 s), and tessellation30 (0.0230→0.0187 s). Dual-screen had no closed-case benefit. Among the three open cases, control→interval relative gaps were Toronto50 10.25→7.88%, random59 3.92→1.34%, and tessellation60 12.57→9.88%. Dual-screen gaps were 10.29%, 5.33%, and 12.61%.

Two further large6 observations per CFR control and CFR+interval alternated run order. The repeated case medians and proof gaps were:

| Actual regions / class | Control median seconds (range) | Interval median seconds (range) | Paired closed speedups | Control open-gap median (range) | Interval open-gap median (range) |
|---|---:|---:|---:|---:|---:|
| 39 OSM | 0.792 (0.752–0.793) | 0.557 (0.550–0.563) | 1.37x, 1.41x, 1.42x | — | — |
| 50 Toronto | 2.001 (2.000–2.001) | 2.001 (2.000–2.001) | open in all runs | 10.60% (10.25–10.84) | 8.08% (7.88–8.11) |
| 30 random | 0.160 (0.155–0.161) | 0.131 (0.125–0.133) | 1.24x, 1.22x, 1.22x | — | — |
| 59 random | 2.000 (2.000–2.002) | 2.001 (2.000–2.012) | open in all runs | 5.33% (3.92–5.33) | 1.66% (1.34–1.71) |
| 30 tessellation | 0.0244 (0.0230–0.0257) | 0.0191 (0.0187–0.0193) | 1.23x, 1.35x, 1.26x | — | — |
| 60 tessellation | 2.001 (2.001–2.003) | 2.001 (2.000–2.003) | open in all runs | 12.75% (12.57–12.75) | 9.88% (9.88–10.31) |

Interval therefore won all nine paired, gap-closed large cases, with a median paired speedup of 1.263x; the three larger open cases also had lower gaps in every interval repeat. The measured benefit reflects the tested CFR workload and caps; it does not establish a universal speedup for every polygon class.

## Portfolio screens

On V1, CFR+memo and CFR+memo+share-bounds both returned six valid large6 tours, with three closed and three at the time limit. Sharing counters were 222,045 queries, 21,796 hits, 1,157 improvements, and 171 prunes. Closed times were OSM39 0.702→0.707 s, random30 0.169→0.175 s, and tessellation30 0.0236→0.0254 s. The open-case gaps changed Toronto50 11.12→11.63%, random59 5.33→5.33%, tessellation60 12.84→12.94%. This screen showed fewer calls but no wall-time gain.

V2's compact interned key index was then tested against its same-binary CFR+memo control. Sharing counters were 218,926 queries, 21,454 hits, 1,123 improvements, and 171 prunes. All six outputs were valid; three were closed and three were time-limited. OSM39 closed at 0.762→0.733 s, random30 at 0.184→0.184 s, and tessellation30 at 0.0243→0.0238 s. Toronto50's open gap was 11.63→11.75%; random59 5.33→5.33%; tessellation60 13.56→13.56%. This single bounded screen is diagnostic evidence for index overhead and reuse counts, not an aggregate speedup claim. Sharing remains opt-in.

For OSM39 at 10s/15s, the matched portfolio CFR+memo control times were 0.767, 0.754, and 0.747 s; adding interval gave 0.507, 0.517, and 0.505 s. All six tours were valid and gap-closed. Median time fell from 0.754 to 0.507 s; the median paired speedup was 1.481x (three of three wins). The Fekete timing rows for this protocol were reused identically from the matching 10s/15s reference; they were not rerun.

## Conclusion and limits

The interval certificate path is the only idea with consistent performance evidence here: it improved all nine repeated gap-closed large6 comparisons and all three repeated OSM39 portfolio comparisons, and reduced open-case gaps at the same 2-second budget. Dual-screen produced only four prunes in its one large6 screen and no closed-case speedup. Shared bounds achieved verified reuse and 171 prunes, but did not show a wall-time improvement; its modest interned-key refinement did not change that conclusion. Keep all options disabled by default pending wider validation.

All saved run data, configs, inputs, reused Fekete references, build/test outcomes, candidate patches, binary hashes, and analysis scripts are in this campaign. The full repository sanity script, dashboard browser suite, and WASM intersection checks were not run.
