# Three-repetition holdout comparison

This is the follow-up run on the same six cases fixed before timing. Every case/backend/mode ran three times with a 1-second native limit and 8-second process cap. All four modes used the same frozen native executable and Fekete executable.

| Native mode | Valid / exact-cover / gap-closed native runs | Fekete valid / exact-cover / gap-closed runs | Matched cases | Native wins among matched cases | Median Fekete/native speedup | Max native time |
|---|---:|---:|---:|---:|---:|---:|
| Default | 18/18 / 6/18 / 18/18 | 18/18 / 9/18 / 15/18 | 5/6 | 4/5 | 2.26× | 99.6 ms |
| Isolated dfs-bfs | 18/18 / 6/18 / 18/18 | 18/18 / 9/18 / 15/18 | 5/6 | 4/5 | 3.16× | 137.9 ms |
| Independent race (no sharing) | 18/18 / 6/18 / 18/18 | 18/18 / 9/18 / 15/18 | 5/6 | 4/5 | 2.00× | 127.6 ms |
| Cooperative incumbent sharing | 18/18 / 6/18 / 18/18 | 18/18 / 9/18 / 15/18 | 5/6 | 4/5 | 2.19× | 104.6 ms |

Speedups use case-level median times and count only cases where all three runs from both backends pass tolerance validation and close the matched gap. Four of five matched cases favor the native solver in every mode; the 20-polygon tessellation case favors Fekete. The 10-polygon tessellation case has valid tours from both solvers, but Fekete does not close the requested gap in any of its three repetitions, so it is excluded from speedup counts. All runs returned valid tours; `exactly_covers` is a stricter exact-contact indicator, reported separately.

| Portfolio mode | Publications / imports | Median proof time | Median / maximum join tail |
|---|---:|---:|---:|
| Independent race (no sharing) | 0 / 0 | 4.557 ms | 0.241 / 3.157 ms |
| Cooperative incumbent sharing | 52 / 24 | 3.848 ms | 0.187 / 0.990 ms |

The shared portfolio had a 2.19× median matched speedup over Fekete, compared with 2.00× for the independent race and 3.16× for isolated `dfs-bfs`. The paired cooperative/default runtime ratio is listed per case in `aggregate.json`; its median is 0.88×. This small holdout does not show a consistent cooperative speed advantage over the independent race or isolated strategy. The largest native time observed was 137.9 ms, well below the 1-second solver budget; no process timeouts or runner errors occurred.

All six instances had a common intersection among all reported lower and upper bounds across modes and repetitions: True. The per-case intersections and all per-run bounds are in `aggregate.json` and the mode `raw.jsonl` files.
