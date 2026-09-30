# Two-worker portfolio memo comparison

Same frozen shared-memo candidate binary, one thread per portfolio worker, default cooperative incumbent sharing, 2 s nominal / 8 s firm cap, one repetition. Control and treatment both used `cache + features + root`; memo was the only toggle. Fekete rows are reused for reference and are not a resource-matched comparison to two native workers.

| Large6 case | Control s; calls/nodes; gap | Memo s; calls/nodes; gap | Closed in both? | Memo queries / repeats / hits |
|---|---:|---:|---:|---:|
| large_osm_25-40_sao_paulo_n040_seed7537 | 0.8455; 1166/647; closed | 0.7153; 1198/660; closed | yes | 1198 / 372 / 372 |
| large_osm_41-60_toronto_n050_seed4504 | 2.0009; 5233/2735; 11.2149% | 2.0006; 5327/2794; 11.2717% | no | 5327 / 487 / 487 |
| large_random_25-40_random_mixed_square_n030_seed30019 | 0.1727; 243/150; closed | 0.1735; 252/153; closed | yes | 252 / 43 / 43 |
| large_random_41-60_random_mixed_square_n060_seed60012 | 2.0323; 2106/1128; 5.3273% | 2.0154; 2502/1469; 5.3273% | no | 2502 / 361 / 361 |
| large_tessellation_25-40_uniform-0000030-1 | 0.0353; 88/49; closed | 0.0245; 87/49; closed | yes | 87 / 20 / 20 |
| large_tessellation_41-60_uniform-0000060-1 | 2.0058; 1548/309; 12.8366% | 2.0070; 1547/310; 12.8366% | no | 1547 / 58 / 58 |

All six tours were independently valid in both runs. Three gaps closed in both; open-case gaps were Toronto 11.2149%→11.2717%, random60 5.3273%→5.3273%, and tessellation60 12.8366%→12.8366%. Across the six cases, memo had 1,341 hits from 10,913 requests; this is 2-worker work reuse and must not be read as single-core parity with Fekete.

## OSM39 repeated proof case

Same 39-region São Paulo proof case (`large_osm_25-40_sao_paulo_n040_seed7537`; source selection label OSM39), same frozen executable and CFR options, two portfolio workers, 10 s nominal / 15 s firm cap, three paired repetitions. Fekete timings were reused only from the exact 10 s/15 s reference; native control/treatment is the relevant comparison.

| Repeat | Control sec; calls | Memo sec; calls | Memo queries / hits | Valid / closed |
|---:|---:|---:|---:|---:|
| 1 | 0.8243; 1161 | 0.6922; 1199 | 1199 / 368 | True / True |
| 2 | 0.8301; 1175 | 0.7212; 1199 | 1199 / 378 | True / True |
| 3 | 0.8217; 1165 | 0.7023; 1199 | 1199 / 376 | True / True |

Both variants returned valid tours and closed the gap in all 3 repetitions, with identical recorded LB/UB (`560.832360047729` / `560.832360047729`). Control median 0.8243s (range 0.8217–0.8301); memo median 0.7023s (range 0.6922–0.7212), a 14.8% median reduction. Cache hits were [368, 378, 376], median 376 and mean 374.0 per run. Hit entries are copied under lock and independently certified; repeated requests avoided reconstructing/re-solving, not exact result verification.

Per-case rows are in `runs/portfolio-large6-per-case.csv`; OSM39 repeat details are in `runs/portfolio-osm39-repeat3.csv`.
