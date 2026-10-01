# Final results — 2026-09-29

All runs use free cyclic order, closed polygon regions, no fixed point, target `UB <= (1+1e-6) LB`, feasibility tolerance `1e-8`, and independent validation tolerance `1e-7`. The primary screens used one repetition, a 2-second solver limit and 8-second process cap. Relative gaps below are `(UB-LB)/UB`; a valid tour does not imply that the requested gap closed. Native timing excludes process startup. Fekete reference rows were measured once per matching input/settings and reused with hash-checked `--reference-results`.

The full v1 idea matrix is in `matrix-comparison.tsv` and `matrix-aggregate.json`; v1 includes isolated `cache`, `dual`, `features`, `lazy`, `root`, `branch`, and all-six combined. Follow-up raw rows and per-case bounds remain in `runs/`. The shortlist is v3 `cache+features+root` (CFR), with `cache+features` (CF) as the rootless comparison.

## Shortlist screen

| Set | Configuration | Valid / gap closed | Median paired speedup vs frozen baseline |
|---|---|---:|---:|
| Small, 6 cases | v3 CF | 6/6, 6/6 | 1.65× |
| Small, 6 cases | v3 CFR | 6/6, 6/6 | 1.67× |
| Large, 6 cases | v3 CF | 6/6, 2/6 | 1.33× on 2 mutually closed cases |
| Large, 6 cases | v3 CFR | 6/6, 3/6 | 1.26× on 2 mutually closed cases |
| Large, 6 cases | v3 portfolio CF, two workers | 6/6, 3/6 | 1.00× on 2 mutually closed cases |

The large set is budget-limited: four of six frozen-baseline cases did not close the requested gap. CFR closed OSM k=39 in 0.802 seconds (580 calls), while the frozen baseline remained open at 2.001 seconds (377 calls). CFR also closed the random k=30 and tessellation k=30 cases. On its three open cases, CFR bounds were:

| Class / count | CFR seconds / calls | CFR gap | CFR UB / LB |
|---|---:|---:|---:|
| OSM k=50 | 2.000 / 3006 | 10.84% | 973.835 / 868.313 |
| Random k=59 | 2.002 / 1379 | 5.33% | 166.404 / 157.540 |
| Tessellation k=60 | 2.000 / 997 | 12.75% | 13879.101 / 12109.370 |

All six CFR large tours validated. The same-settings 2-second Fekete tessellation k=60 gap was 15.19%, also open. An earlier 5-second Fekete screen had 7.88%; that budget differs and is not a matched comparison. Portfolio CF also validated 6/6 and closed 3/6, but its k=50, k=59 and k=60 bounds were weaker than single-worker CFR. On OSM k=39 it took 0.924 seconds / 857 calls versus CFR’s 0.802 seconds / 580 calls.

## Three-repetition checks

On the small six-case holdout, v3 CFR validated and closed all 18/18 native runs. Frozen-baseline → CFR median times were OSM k=5: 0.740→0.299 ms; OSM k=15: 5.433→1.993 ms; random k=10: 2.429→1.863 ms; random k=15: 35.982→36.606 ms; tessellation k=10: 2.759→4.453 ms; tessellation k=20: 96.226→48.988 ms. The root option therefore is not uniformly faster; CF measured 1.616 ms on tessellation k=10. Full ranges, calls and validation rows are in `runs/small-repeat3-*`.

Across the five cases where both CFR and Fekete validated and closed all repetitions, CFR’s median matched speedup was **5.793746×**, with five wins. By class: OSM 2/2 wins, median 7.67×; random 2/2, median 4.68×; tessellation 1/2 matched, median 1.60×, one win. Fekete returned valid tours on tessellation k=10 but did not close the target gap, so that case is excluded.

On the OSM k=39 proof case at 10 seconds / 15-second process cap, frozen baseline, standalone CFR and portfolio CF each validated and closed 3/3 runs. Their medians were 4.696 s (4.676–4.725), 0.788 s (0.781–0.788), and 0.957 s (0.929–0.967), respectively. Standalone CFR was 5.96× faster than the frozen native baseline. Fekete was valid 3/3 but remained open at a 5.57e-6 median relative gap, so no matched Fekete speedup is reported.

## Limits and provenance

Lazy boundary solving timed out on large OSM k=39 at the 8-second cap on v1, v2 and v3. The bounded v1 sample localized 849/849 sampled frames to cycle relaxation / exact intersecting-boundary recovery, with 727 in `search_intersecting_boundaries` and repeated `DirectionalMaps::locate` → `build_vertex` recursion. `diagnostics/lazy-osm39-profile-summary.md` records only this function/count evidence. The v3 warm-recovery change did not remove the timeout. V2 and v3 CFR runs had essentially unchanged bounds and timings; no measurable warm-fix performance gain is claimed.

All optimization flags remain individually selectable. Large gap closure is limited: results do not support a blanket speedup claim across every class and size. The run summaries preserve objective spread and interval separation separately from tour validation and gap closure. Binary executables remain under ignored `.build/`; `provenance/v1`, `provenance/v2` and `provenance/v3` contain each frozen build’s hash and source snapshot, including the untracked portfolio header.
