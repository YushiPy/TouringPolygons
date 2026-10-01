# Baseline convex hull relaxations

Native executable from commit 924e8b6, one warm-up and one timed repetition. Same normalized hull inputs as oracle-after. These are diagnostic measurements, not the repeated full-B&B comparison.

| Instance | Rational ms | Double ms | Gurobi ms | Rational candidates | Double candidates |
|---|---:|---:|---:|---:|---:|
| german_2_n5_0,4 | 0.2131 | 0.2700 | 0.4241 | 1 | 3 |
| german_2_n5_0,2,4 | 18.2317 | 22.0776 | 0.5169 | 3 | 68 |
| german_3_n10_0,9 | 0.1193 | 0.0496 | 0.3829 | 1 | 1 |
| german_3_n10_0,4,9 | 0.1697 | 0.0815 | 0.4383 | 1 | 1 |
| german_3_n10_0,2,4,9 | 0.8802 | 0.2532 | 0.4999 | 2 | 2 |
| german_3_n10_0,4,9,2 | 0.6774 | 0.5160 | 0.4959 | 2 | 4 |
| german_3_n10_0,4,9,5,2 | 0.8659 | 0.6653 | 0.5735 | 2 | 4 |
| german_3_n10_0,4,8,9,5,2 | 683.4253 | 925.1516 | 1.0983 | 228 | 200 |
| german_3_n10_0,4,9,8,5,2 | 2.6208 | 0.9002 | 0.7820 | 2 | 5 |
| german_3_n10_0,3,4,9,8,5,2 | 67.5209 | 1.9077 | 1.0155 | 4 | 6 |
| german_3_n10_0,4,3,9,8,5,2 | 17781.7775 | 17570.5863 | 1.1406 | 489 | 207 |
| german_6_n15_0,13 | 0.2228 | 0.2970 | 0.4172 | 1 | 3 |
| german_6_n15_0,8,13 | 0.7535 | 0.4786 | 0.5854 | 2 | 4 |
| german_6_n15_0,7,8,13 | 0.6820 | 0.5232 | 0.5554 | 2 | 4 |
| german_6_n15_0,8,13,7 | 0.3931 | 0.6505 | 0.5253 | 1 | 4 |
| german_6_n15_0,10,7,8,13 | 4399.9504 | 4505.7420 | 1.0489 | 482 | 143 |
