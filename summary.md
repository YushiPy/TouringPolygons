
## Benchmark Summary

CSV output: `results.csv`

Summary output: `summary.md`

| Metric | Value |
|---|---:|
| Total instances | 558 |
| Benchmarked instances | 558 |
| Benchmark runs | 558 |
| Repeat count | 1 |
| Worker threads | 12 |
| Fully solved runs | 551 |
| Capped by calls runs | 7 |
| Branch limited runs | 0 |
| Skipped by max polygons | 0 |
| Skipped empty | 0 |
| Skipped decomposition | 0 |
| Skipped no calls | 0 |
| Max observed branching | 82 |

| B&B Counter | Value |
|---|---:|
| Total convex calls | 13_733_199 |
| Incumbent solves | 558 |
| Bound solves | 13_379_520 |
| Leaf solves | 353_121 |
| Visited nodes | 3_868_009 |
| Pruned nodes | 9_511_963 |
| Best updates | 555 |

| Timing | Value |
|---|---:|
| Decomposition | 2.732432s (1.75%) |
| Approximation | 0.131547s (0.08%) |
| B&B | 153.006586s (98.16%) |
| Convex solver | 114.215363s (73.28%) of measured work |
| Measured work | 155.870565s (100.00%) |
| Wall-clock total | 17.628342s |
| Parallel speedup estimate | 8.84x |
| Mean seconds per call | 0.000008316734s |
| Checksum | 53014032366.363121032715 |

## Distributions

| Metric | Min | Median | P90 | P99 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Seconds per call | 0.000001 | 0.000005 | 0.000010 | 0.000024 | 0.000120 | 0.000006 |
| Calls | 2 | 101 | 15_028 | 1_000_000 | 1_000_000 | 24_611 |
| Best updates | 0 | 1 | 1 | 2 | 2 | 1 |
| Initial gap % | 0.000000 | 0.000577 | 0.006527 | 0.055362 | 0.163535 | 0.003378 |
| Incumbent gap % | 0.000000 | 0.000000 | 0.000000 | 0.002706 | 0.046131 | 0.000190 |
| Max branching | 1 | 5 | 9 | 20 | 82 | 5 |
| Decomposed pieces | 4 | 50 | 182 | 256 | 498 | 75 |

## Derived Metrics

| Metric | Value |
|---|---:|
| Overall prune rate | 71.09% |
| Calls per visited node | 3.550457 |
| Bound calls per leaf | 37.889335 |
| Solver time share | 73.28% |
| Mean failed-prune bound/incumbent | 0.998027 |
| Mean failed-prune incumbent-bound gap | 0.052477 |
| Mean failed-prune depth | 19.436 |

## Histograms

| Histogram | Buckets |
|---|---|
| Visited node depth | 0: 558, 1: 688, 2: 877, 3: 1_072, 4: 1_207, 5: 1_564, 6: 1_799, 7: 2_317, 8: 2_477, 9: 2_993, 10: 3_556, 11: 3_683, 12+: 3_845_218 |
| Bound solve depth | 0: 0, 1: 1_441, 2: 1_974, 3: 2_734, 4: 3_216, 5: 3_953, 6: 4_868, 7: 6_394, 8: 7_680, 9: 10_493, 10: 12_670, 11: 9_475, 12+: 13_314_629 |
| Leaf solve depth | 0: 0, 1: 0, 2: 0, 3: 0, 4: 15, 5: 114, 6: 1, 7: 0, 8: 0, 9: 4, 10: 582, 11: 0, 12+: 352_405 |
| Branching | 1: 227_778, 2: 465_305, 3-5: 2_330_492, 6-10: 469_469, 11+: 21_844 |

## Worst Runs

### By Runtime

| Case | Repeat | Value (Seconds) | Calls | Pieces | Max Branch | Initial Gap | Exhausted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 47 | 0 | 13.971405 | 1_000_000 | 252 | 7 | 0.000368% | false |
| 0 | 0 | 12.346767 | 1_000_000 | 241 | 10 | 0.000084% | false |
| 241 | 0 | 11.375823 | 1_000_000 | 220 | 6 | 0.000088% | false |
| 78 | 0 | 11.271137 | 1_000_000 | 202 | 7 | 0.000279% | false |
| 481 | 0 | 9.777186 | 1_000_000 | 239 | 7 | 0.000351% | false |
| 440 | 0 | 9.673756 | 1_000_000 | 224 | 7 | 0.000362% | false |
| 557 | 0 | 7.783815 | 1_000_000 | 201 | 7 | 0.000434% | false |
| 300 | 0 | 6.655976 | 555_019 | 256 | 12 | 0.000549% | true |
| 250 | 0 | 6.443898 | 474_356 | 253 | 8 | 0.000020% | true |
| 391 | 0 | 5.416750 | 380_507 | 244 | 33 | 0.000175% | true |

### By Convex Calls

| Case | Repeat | Value (Calls) | Calls | Pieces | Max Branch | Initial Gap | Exhausted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 78 | 0 | 1_000_000 | 1_000_000 | 202 | 7 | 0.000279% | false |
| 440 | 0 | 1_000_000 | 1_000_000 | 224 | 7 | 0.000362% | false |
| 557 | 0 | 1_000_000 | 1_000_000 | 201 | 7 | 0.000434% | false |
| 0 | 0 | 1_000_000 | 1_000_000 | 241 | 10 | 0.000084% | false |
| 241 | 0 | 1_000_000 | 1_000_000 | 220 | 6 | 0.000088% | false |
| 481 | 0 | 1_000_000 | 1_000_000 | 239 | 7 | 0.000351% | false |
| 47 | 0 | 1_000_000 | 1_000_000 | 252 | 7 | 0.000368% | false |
| 300 | 0 | 555_019 | 555_019 | 256 | 12 | 0.000549% | true |
| 554 | 0 | 507_702 | 507_702 | 167 | 6 | 0.000495% | true |
| 250 | 0 | 474_356 | 474_356 | 253 | 8 | 0.000020% | true |

### By Decomposed Pieces

| Case | Repeat | Value (Pieces) | Calls | Pieces | Max Branch | Initial Gap | Exhausted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 248 | 0 | 498 | 76_108 | 498 | 45 | 0.000000% | true |
| 50 | 0 | 344 | 191_775 | 344 | 12 | 0.000134% | true |
| 130 | 0 | 301 | 3_558 | 301 | 6 | 0.000004% | true |
| 129 | 0 | 275 | 19_214 | 275 | 15 | 0.000031% | true |
| 498 | 0 | 269 | 7_049 | 269 | 9 | 0.000010% | true |
| 300 | 0 | 256 | 555_019 | 256 | 12 | 0.000549% | true |
| 477 | 0 | 256 | 258_637 | 256 | 6 | 0.000076% | true |
| 250 | 0 | 253 | 474_356 | 253 | 8 | 0.000020% | true |
| 47 | 0 | 252 | 1_000_000 | 252 | 7 | 0.000368% | false |
| 402 | 0 | 252 | 10_776 | 252 | 8 | 0.000461% | true |

### By Max Branching

| Case | Repeat | Value (Max Branch) | Calls | Pieces | Max Branch | Initial Gap | Exhausted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 204 | 0 | 82 | 2_968 | 240 | 82 | 0.000403% | true |
| 248 | 0 | 45 | 76_108 | 498 | 45 | 0.000000% | true |
| 391 | 0 | 33 | 380_507 | 244 | 33 | 0.000175% | true |
| 121 | 0 | 31 | 33_878 | 164 | 31 | 0.000048% | true |
| 225 | 0 | 26 | 206_140 | 234 | 26 | 0.000064% | true |
| 57 | 0 | 20 | 1_155 | 116 | 20 | 0.001292% | true |
| 32 | 0 | 18 | 342 | 83 | 18 | 0.000048% | true |
| 273 | 0 | 17 | 335 | 65 | 17 | 0.001203% | true |
| 528 | 0 | 15 | 278 | 126 | 15 | 0.000238% | true |
| 129 | 0 | 15 | 19_214 | 275 | 15 | 0.000031% | true |

### By Initial Gap

| Case | Repeat | Value (Gap %) | Calls | Pieces | Max Branch | Initial Gap | Exhausted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 405 | 0 | 0.163535 | 12 | 10 | 1 | 0.163535% | true |
| 176 | 0 | 0.106338 | 12 | 10 | 1 | 0.106338% | true |
| 377 | 0 | 0.105530 | 10 | 8 | 2 | 0.105530% | true |
| 459 | 0 | 0.075286 | 8 | 6 | 2 | 0.075286% | true |
| 219 | 0 | 0.064138 | 7 | 5 | 1 | 0.064138% | true |
| 143 | 0 | 0.055362 | 36 | 22 | 5 | 0.055362% | true |
| 478 | 0 | 0.047776 | 7 | 5 | 1 | 0.047776% | true |
| 234 | 0 | 0.046131 | 7_654 | 63 | 11 | 0.046131% | true |
| 386 | 0 | 0.040763 | 12 | 10 | 1 | 0.040763% | true |
| 436 | 0 | 0.036816 | 13 | 11 | 2 | 0.036816% | true |

Tip: with summary output enabled, render it with `glow summary.md`.
