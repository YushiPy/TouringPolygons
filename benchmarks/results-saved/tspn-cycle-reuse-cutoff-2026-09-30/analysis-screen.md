# First screen — V2 CFR memo / bound-first

All runs used one thread, 2 s nominal / 8 s firm cap, one repetition. All four strategies share the same frozen candidate binary. Fekete rows were reused from exact input/settings references; no Fekete run was part of treatment timing. Each native case returned a valid tour; small6 closed 6/6 and large6 closed 3/6 for every variant. Large open cases ended at the native time limit, not process timeout.

| Dataset | Variant | valid / closed | native TL / process timeout | median sec (range) | calls | nodes | memo queries / repeated / hits | certified cutoff skips | initial contact checks / accepts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small6 | cfr-control | 6/6 / 6/6 | 0 / 0 | 0.0034 (0.0006–0.0499) | 129 | 44 | 0 / 0 / 0 | 0 | 0 / 0 |
| small6 | cfr-memo | 6/6 / 6/6 | 0 / 0 | 0.0034 (0.0003–0.0512) | 129 | 44 | 129 / 0 / 0 | 0 | 0 / 0 |
| small6 | cfr-bound-first | 6/6 / 6/6 | 0 / 0 | 0.0039 (0.0003–0.0686) | 129 | 44 | 0 / 0 / 0 | 25 | 123 / 24 |
| small6 | cfr-memo-bound-first | 6/6 / 6/6 | 0 / 0 | 0.0039 (0.0004–0.0667) | 129 | 44 | 129 / 0 / 0 | 25 | 123 / 24 |
| large6 | cfr-control | 6/6 / 3/6 | 3 / 0 | 1.4080 (0.0253–2.0104) | 5951 | 2082 | 0 / 0 / 0 | 0 | 0 / 0 |
| large6 | cfr-memo | 6/6 / 3/6 | 3 / 0 | 1.4390 (0.0249–2.0344) | 5995 | 2092 | 5995 / 0 / 0 | 0 | 0 / 0 |
| large6 | cfr-bound-first | 6/6 / 3/6 | 3 / 0 | 1.4831 (0.0313–2.0020) | 4993 | 1710 | 0 / 0 / 0 | 359 | 4988 / 441 |
| large6 | cfr-memo-bound-first | 6/6 / 3/6 | 3 / 0 | 1.4859 (0.0304–2.0013) | 5012 | 1726 | 5012 / 0 / 0 | 359 | 5007 / 442 |

Memo had zero repeated canonical keys and zero hits in both datasets; this screen provides no evidence of reusable identical relaxation states. `bound-first` recorded cutoff skips and accepted some inherited initial contacts, but did not reduce median wall time consistently; small6 median rose, and OSM40 showed no time improvement. Large open-case gaps were sometimes wider at the fixed time cap, so do not infer proof-progress gains from fewer nodes alone. Per-case values, including bounds, calls, nodes, and all new counters, are in `per-case-screen.csv`. Further repeats are deferred pending selection from this first screen.
