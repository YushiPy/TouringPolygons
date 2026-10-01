# Held-Karp and learned branching screen

Native runs used 2 s nominal / 8 s firm, CFR (`cache + features + root`), one repetition and one thread. Fekete rows were reused from exact input/settings/binary-hash references. “Time limit” means a valid native run returning a feasible route with an open gap; process timeouts are counted separately. Gap is `(UB-LB)/UB`.

The formulation is a closed Euclidean tour of the supplied simple polygons,
with free visit order and no fixed point; overlaps are allowed and holes are
not represented. The existing comparison contract is unchanged: native relative
gap `9.99999000001e-7`, absolute gap zero, Fekete relative gap `1e-6`, native
feasibility tolerance `1e-8`, and independent validation tolerance `1e-7`.
Exact binary-rational containment/intersection predicates accompany numerical
distance diagnostics in that validator. The new one-tree bound uses exact
rational arithmetic and downward rounding, without a new acceptance epsilon.
Closing the full B&B's requested numerical gap is not a zero-error rational
optimality claim. All strict Gurobi parameters are recorded in each run config.

## V1 four-variant screen

| Data | Variant | Valid tours | Gap closed | Native time-limit | Process timeouts |
|---|---|---:|---:|---:|---:|
| small6 | cfr-baseline | 6/6 | 6/6 | 0/6 | 0/6 |
| small6 | cfr-one-tree | 6/6 | 6/6 | 0/6 | 0/6 |
| small6 | cfr-learn | 6/6 | 6/6 | 0/6 | 0/6 |
| small6 | cfr-one-tree-learn | 6/6 | 6/6 | 0/6 | 0/6 |
| large6 | cfr-baseline | 6/6 | 3/6 | 3/6 | 0/6 |
| large6 | cfr-one-tree | 6/6 | 3/6 | 3/6 | 0/6 |
| large6 | cfr-learn | 6/6 | 3/6 | 3/6 | 0/6 |
| large6 | cfr-one-tree-learn | 6/6 | 3/6 | 3/6 | 0/6 |

All V1 strategies closed 6/6 small cases. Each returned 6/6 valid large tours and closed 3/6 gaps; remaining rows stopped at the native time limit, not process timeout. Do not infer speedup from a mixed median across closed and time-limited rows. Raw rows contain per-case seconds, bounds, nodes, calls and counters.

V1 large-case times/calls/gaps in case order OSM40, Toronto50, random30, random60, tessellation30, tessellation60:
- `cfr-baseline`: large_osm_25-40_sao_paulo_n040_seed7537 0.775s/580 calls/gap 0.00%/HK 0.0ms/0 learned reorderings; large_osm_41-60_toronto_n050_seed4504 2.001s/3008 calls/gap 10.84%/HK 0.0ms/0 learned reorderings; large_random_25-40_random_mixed_square_n030_seed30019 0.158s/123 calls/gap 0.00%/HK 0.0ms/0 learned reorderings; large_random_41-60_random_mixed_square_n060_seed60012 2.042s/1385 calls/gap 5.33%/HK 0.0ms/0 learned reorderings; large_tessellation_25-40_uniform-0000030-1 0.024s/44 calls/gap 0.00%/HK 0.0ms/0 learned reorderings; large_tessellation_41-60_uniform-0000060-1 2.002s/1003 calls/gap 12.75%/HK 0.0ms/0 learned reorderings
- `cfr-one-tree`: large_osm_25-40_sao_paulo_n040_seed7537 0.881s/580 calls/gap 0.00%/HK 100.8ms/0 learned reorderings; large_osm_41-60_toronto_n050_seed4504 2.000s/3031 calls/gap 10.60%/HK 15.6ms/0 learned reorderings; large_random_25-40_random_mixed_square_n030_seed30019 0.199s/123 calls/gap 0.00%/HK 36.4ms/0 learned reorderings; large_random_41-60_random_mixed_square_n060_seed60012 2.015s/1373 calls/gap 5.33%/HK 131.3ms/0 learned reorderings; large_tessellation_25-40_uniform-0000030-1 0.031s/44 calls/gap 0.00%/HK 6.1ms/0 learned reorderings; large_tessellation_41-60_uniform-0000060-1 2.001s/993 calls/gap 12.75%/HK 24.7ms/0 learned reorderings
- `cfr-learn`: large_osm_25-40_sao_paulo_n040_seed7537 1.289s/1297 calls/gap 0.00%/HK 0.0ms/468 learned reorderings; large_osm_41-60_toronto_n050_seed4504 2.001s/3220 calls/gap 10.78%/HK 0.0ms/290 learned reorderings; large_random_25-40_random_mixed_square_n030_seed30019 0.182s/139 calls/gap 0.00%/HK 0.0ms/15 learned reorderings; large_random_41-60_random_mixed_square_n060_seed60012 2.013s/1502 calls/gap 4.73%/HK 0.0ms/210 learned reorderings; large_tessellation_25-40_uniform-0000030-1 0.024s/44 calls/gap 0.00%/HK 0.0ms/0 learned reorderings; large_tessellation_41-60_uniform-0000060-1 2.001s/610 calls/gap 17.18%/HK 0.0ms/34 learned reorderings
- `cfr-one-tree-learn`: large_osm_25-40_sao_paulo_n040_seed7537 1.857s/1297 calls/gap 0.00%/HK 579.2ms/468 learned reorderings; large_osm_41-60_toronto_n050_seed4504 2.001s/3193 calls/gap 10.79%/HK 15.5ms/290 learned reorderings; large_random_25-40_random_mixed_square_n030_seed30019 0.214s/139 calls/gap 0.00%/HK 36.8ms/15 learned reorderings; large_random_41-60_random_mixed_square_n060_seed60012 2.019s/1492 calls/gap 4.73%/HK 156.9ms/207 learned reorderings; large_tessellation_25-40_uniform-0000030-1 0.030s/44 calls/gap 0.00%/HK 5.6ms/0 learned reorderings; large_tessellation_41-60_uniform-0000060-1 2.000s/618 calls/gap 16.41%/HK 23.6ms/36 learned reorderings

V1 learned branching changed choices 1,017 times across the large cases. OSM40 changed from 310 nodes/580 calls/0.775 s under CFR to 784/1,297/1.289 s with learn. Large tessellation60 gap widened from 12.75% to 17.18%. The combined strategy did not remove this cost.

## V2 after nearest-neighbor/2-opt ascent target

V2 has final source hash in `provenance/candidate-v2-source-hashes.txt`. Its same-binary CFR control isolates the effect of new flags from source/build changes. V2 strategies returned 12/12 valid tours on the two screens; every large variant closed 3/6 gaps; three remaining cases were native time-limit rows and zero were process timeouts.

| Large case | CFR sec; nodes/calls; init LB/UB; final gap | +one-tree sec; calls; init LB; gap; HK ms/iterations/improvements | +one-tree+learn sec; calls; init LB; gap; HK ms/iterations/improvements; learned changes |
|---|---|---|---|
| large_osm_25-40_sao_paulo_n040_seed7537 | 0.792; 310/580; 0.00/605.85; 0.00% | 0.870; 580; 205.11; 0.00%; 105.8/704/1 | 1.870; 1297; 205.11; 0.00%; 620.3/4352/1; 468 |
| large_osm_41-60_toronto_n050_seed4504 | 2.000; 972/3085; 0.00/1047.43; 10.38% | 2.000; 3090; 390.48; 10.38%; 15.4/32/1 | 2.001; 3245; 390.48; 10.78%; 15.0/32/1; 292 |
| large_random_25-40_random_mixed_square_n030_seed30019 | 0.156; 76/123; 0.00/131.82; 0.00% | 0.194; 123; 46.76; 0.00%; 38.3/384/1 | 0.214; 139; 46.76; 0.00%; 38.0/384/1; 15 |
| large_random_41-60_random_mixed_square_n060_seed60012 | 2.001; 607/1401; 0.00/167.80; 4.33% | 2.009; 1375; 33.47; 5.33%; 146.0/384/1 | 2.018; 1492; 33.47; 4.73%; 177.2/480/1; 207 |
| large_tessellation_25-40_uniform-0000030-1 | 0.024; 25/44; 0.00/4836.59; 0.00% | 0.029; 44; 0.00; 0.00%; 5.5/32/0 | 0.030; 44; 0.00; 0.00%; 5.8/32/0; 0 |
| large_tessellation_41-60_uniform-0000060-1 | 2.001; 218/1055; 0.00/16896.51; 12.61% | 2.001; 1037; 0.00; 12.61%; 24.2/32/0 | 2.001; 638; 0.00; 16.41%; 24.2/32/0; 39 |

Across the six V2 large cases, one-tree made 4/6 root-bound improvements, using 1,568 HK iterations and about 335 ms total, but did not increase completed cases. Relative to same-binary CFR, open gaps were unchanged for Toronto50 and tessellation60; random60 widened from 4.33% to 5.33%. The new lower bound adds work without consistent proof progress. Combined with learning, OSM40 grows from 580 to 1,297 calls and 0.792 to 1.870 s; open gaps worsen on Toronto50 (10.38%→10.78%), random60 (4.33%→4.73%), and tessellation60 (12.61%→16.41%). Keep both options opt-in.

## Repeated OSM39 check

Same V2 executable and 10 s nominal / 15 s process cap were run three times, with `learn` added for the treatment. Fekete rows were reused from the matching three-repetition 10 s / 15 s config, not from the 2 s / 8 s holdout references.
- `osm39-v2-cfr-10s-repeat3`: median 0.751s; 3/3 valid, 3/3 closed; per-run calls [580, 580, 580]; learned-choice changes [0, 0, 0].
- `osm39-v2-cfr-learn-10s-repeat3`: median 1.235s; 3/3 valid, 3/3 closed; per-run calls [1297, 1297, 1297]; learned-choice changes [468, 468, 468].

Learn increased OSM39 calls from 580 to 1,297 and median elapsed time from about 0.751 s to 1.235 s. The calls identify a search-tree/ordering change; the convex-oracle implementation was unchanged. Fekete timings match the 10s/15s configuration and are reused, but do not enter the native baseline-versus-learn timing.

## Bound compatibility and validation

Every incumbent route was independently validated by the binary-rational geometry checker (`validation.valid=true`; closure and polygon distances are included in raw rows). `time_limit` indicates a valid incumbent with an open interval, not a process timeout or gap closure. The one-tree output is a lower bound in the same tour-length objective; alone it cannot certify optimality. The interpretation that neighboring/touching regions can have individually cheap but mutually incompatible pair contacts is a structural explanation, not something proven by this sample.

The audit covers 84 native rows: all passed independent route validation,
objective/upper-bound consistency, lower/upper-bound ordering and configured
gap-status checks. There were 63 gap-closed results, 21 native time limits and
zero process timeouts. Counts include repeated inputs and all preserved
versions; they do not represent 84 distinct benchmark instances.

The cross-run audit in `provenance/objective-interval-cross-run-audit.json`
compares 63 closed objectives (210 pairs) and 483 same-input intervals: no
contradiction within the reported comparison contract. Closed objectives agree
exactly across variants; the largest interval excess before applying that
contract is `2.84e-14`. Toronto50, random59 and tessellation60 have no closed
native reference in this campaign, so their open intervals cannot establish
their optimum. Contact-coordinate equality is never required.

## Provenance and scope

Candidate E baseline SHA `277452400f5c472919e3acbace5ac03a07abb6d5ec252ee5fd04f8c7d70fb3f2`; V1 SHA `c200491a7f34807f17f3898c182024f87a3464f3fa7bb8bbc24ddac1daf0f4dc`; V2 SHA `a12e2d4bdace1b0ff09d5d99c37237e838a5be7a520655fc2d19eb5c67777a2e`. Source diffs and per-file hashes are under `provenance/`. Fekete inputs/configs/commits/binary hashes are in `references/`. The unchanged untracked portfolio overlay is referenced in `provenance/source-overlay-reference.txt`. GMP ON/OFF tests passed on final source; Python CLI help lists both modes and the public CLI runs exercised each flag combination. Full repository sanity, dashboard/browser and WASM suites were not run. Keep both features opt-in; do not claim broad improvement from time-limited cases.
