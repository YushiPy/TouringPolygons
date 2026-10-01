# Filtered cutoff proposal screen

Filtered candidate binary SHA-256: `2ef10f6cfac6a0542ba2662f237e5e469bbe2ce0a88970247dc37873105b4851`. One repetition, one thread, 2 s nominal / 8 s firm; all rows use the same binary, with and without `bound-first`. Fekete timing was reused only from exact matching settings. The comparison contract remains the campaign contract: relative target `9.99999000001e-7`, absolute gap 0, feasibility tolerance `1e-8`, independent validation tolerance `1e-7`; report gap is `(UB-LB)/UB`.

All 12 native outcomes validated; small closed 6/6, large closed 3/6; process timeouts 0. Large native time-limit rows are feasible tours with open gaps.

## Per-case results

| Dataset / case | Control sec; calls/nodes; gap | Filtered bound-first sec; calls/nodes; gap | Closed in both? | Closed-case speed ratio C/BF | Open gap change (pp) | Certified cutoff skips | Initial-contact checks / accepts |
|---|---|---|---:|---:|---:|---:|---:|
| small6 / holdout_osm_5-10_jakarta_n005_seed7447 | 0.0010s; 2/1; closed | 0.0003s; 2/1; closed | yes | 3.083x | — | 0 | 0 / 0 |
| small6 / holdout_osm_11-20_lagos_n015_seed7082 | 0.0041s; 4/3; closed | 0.0021s; 4/3; closed | yes | 1.918x | — | 0 | 0 / 0 |
| small6 / holdout_random_5-10_random_mixed_square_n010_seed10013 | 0.0031s; 8/5; closed | 0.0021s; 8/5; closed | yes | 1.439x | — | 1 | 0 / 0 |
| small6 / holdout_random_11-20_random_mixed_square_n015_seed15016 | 0.0414s; 16/12; closed | 0.0372s; 16/12; closed | yes | 1.114x | — | 0 | 0 / 0 |
| small6 / holdout_tessellation_5-10_uniform-0000010-1 | 0.0045s; 15/4; closed | 0.0046s; 15/4; closed | yes | 0.995x | — | 6 | 0 / 0 |
| small6 / holdout_tessellation_11-20_us-night-0000020.instance | 0.0475s; 84/19; closed | 0.0522s; 84/19; closed | yes | 0.911x | — | 19 | 0 / 0 |
| large6 / large_osm_25-40_sao_paulo_n040_seed7537 | 0.7694s; 580/310; closed | 0.8441s; 580/310; closed | yes | 0.912x | — | 153 | 2 / 2 |
| large6 / large_osm_41-60_toronto_n050_seed4504 | 2.0005s; 3098/979; 10.3605% | 2.0003s; 2723/809; 11.6330% | no | — | +1.2725 | 98 | 0 / 0 |
| large6 / large_random_25-40_random_mixed_square_n030_seed30019 | 0.1561s; 123/76; closed | 0.1747s; 123/76; closed | yes | 0.894x | — | 11 | 4 / 4 |
| large6 / large_random_41-60_random_mixed_square_n060_seed60012 | 2.0006s; 1399/607; 4.3347% | 2.0302s; 1373/596; 5.3273% | no | — | +0.9926 | 75 | 2 / 2 |
| large6 / large_tessellation_25-40_uniform-0000030-1 | 0.0243s; 44/25; closed | 0.0260s; 44/25; closed | yes | 0.934x | — | 4 | 0 / 0 |
| large6 / large_tessellation_41-60_uniform-0000060-1 | 2.0009s; 1048/215; 12.6115% | 2.0030s; 942/181; 12.8366% | no | — | +0.2251 | 69 | 0 / 0 |

Across the 9 cases closed in both variants, the paired native time ratios (control/bound-first) are: holdout_osm_5-10_jakarta_n005_seed7447 3.083x, holdout_osm_11-20_lagos_n015_seed7082 1.918x, holdout_random_5-10_random_mixed_square_n010_seed10013 1.439x, holdout_random_11-20_random_mixed_square_n015_seed15016 1.114x, holdout_tessellation_5-10_uniform-0000010-1 0.995x, holdout_tessellation_11-20_us-night-0000020.instance 0.911x, large_osm_25-40_sao_paulo_n040_seed7537 0.912x, large_random_25-40_random_mixed_square_n030_seed30019 0.894x, large_tessellation_25-40_uniform-0000030-1 0.934x. These are single-run observations, not repeat-supported conclusions.

For the earlier unfiltered screen, the bound-first-only rows recorded 5,111 initial-contact checks, 465 accepts, and 384 cutoff skips across small6/large6. In this filtered screen, the two bound-first runs recorded 8 checks and 8 accepts total (2/2 on large OSM40, 4/4 on large random30, and zero on small6), while cutoff skips rose to 436. This shows the floating proposal filter suppressed nearly all added contact checks on these cases; it did not reduce the cutoff-skip count, and the screen does not establish overall speed benefit. Calls and nodes changed in some open cases; those did not close more gaps. Open gap changes are reported per case above, separated from closed-case timing.

Raw rows/configs live in the two `filtered-*` run directories. Per-case machine-readable pairs are in `runs/filtered-per-case.csv`.
