# Free-order TPP follow-up

Last updated: 2026-09-11

This file is the starting point for resuming the comparison between our
free-order TPP solver and the German `tspn-comparison` implementation.

## Current conclusion

For the adapted fixed-endpoint, free-order TPP, our standalone solver is
clearly stronger under strict correctness requirements. The improvement comes
from both the specialized TPP oracle and our branch-and-bound search.

This is not yet a claim about the Germans' original problem. Their instances
are closed-cycle TSPN instances, and the raw archive can contain holes. Our
experiment used their 558 simplified geometries and added deterministic
exterior start and target points. We should describe the result as an
**adapted endpoint-path comparison**, not a direct reproduction of their paper.

## Main results

All runs used one solver thread per instance, four concurrent instance
processes, a 3-second soft limit, `eps=1e-9`, and independent trajectory
validation at `1e-7`.

| Configuration | Proven optimal | Strictly valid | Sum of solver times | Median time | Total calls | Median calls |
|---|---:|---:|---:|---:|---:|---:|
| Our standalone TPP | **525/558** | **558/558** | **187.125 s** | **0.00516 s** | 5,281,888 | **155.5** |
| German search + SOCP | 49/558 | 107/558 | 640.335 s | 0.26132 s | **1,354,407** on 557 cases | 725.0 |
| German search + TPP oracle | 393/558 | 504/558 | 666.541 s | 0.09293 s | 5,221,455 | 277.0 |

Key interpretations:

- Swapping SOCP for TPP inside the German search increases proofs from 49 to
  393 and cuts median time from 0.261 s to 0.093 s. The oracle is a major gain.
- Using our standalone search increases proofs again, from 393 to 525, and cuts
  median time to 0.005 s. Our search is also materially stronger.
- Neither German configuration proves an instance that our solver does not.
- Our solver is faster than each German configuration on 556/558 instances.
- Standalone TPP returns a strictly valid incumbent on every instance.
- When both TPP-based configurations prove optimality, their objectives agree
  to about `2e-10` relative error. This is strong integration evidence.
- Calls are directly comparable only between the two German-search rows. The
  standalone solver explores a different search tree.

Proofs by instance size:

| Polygons | Cases | Standalone TPP | German + SOCP | German + TPP |
|---:|---:|---:|---:|---:|
| 4-10 | 128 | 128 | 5 | 128 |
| 11-20 | 136 | 136 | 13 | 131 |
| 21-30 | 75 | 74 | 16 | 63 |
| 31-40 | 74 | 73 | 13 | 51 |
| 41-50 | 75 | 71 | 2 | 16 |
| 51-60 | 70 | 43 | 0 | 4 |

## Where the German implementation is better

- On 13 difficult instances where standalone TPP timed out without proving
  optimality, German search + TPP found a meaningfully better valid incumbent.
  The improvements range from about 0.075% to 2.19%. Their initialization,
  branching, or incumbent heuristics are worth studying.
- SOCP has smoother tail behavior inside the German search. It is faster on
  197/558 paired cases and has a slightly lower summed time than TPP there,
  despite being much worse in median time and proof count.
- SOCP returns an approximate incumbent on 557/558 instances. Only 107 satisfy
  our `1e-7` validator, but all 557 satisfy the German repository's native
  `1e-3` tolerance. This may be useful when loose approximate output matters.
- Their native solver supports closed tours and instances with holes. Our
  specialized solver currently has a narrower problem domain.

## Prioritized improvements

### Completed: deadline-aware certified refinement

Implemented on 2026-09-11. The certified convex oracle now accepts a remaining
time budget, checks it inside long-double and extended-precision refinement,
and returns the best feasible upper bound and certified lower bound obtained so
far without claiming a closed gap. The standalone solver passes its remaining
global time to every oracle call. The German-search integration uses the same
branch-and-bound deadline and reports interrupted oracle calls.

Full 558-instance standalone rerun:

- proofs: 525 to 526;
- strict-valid paths: 558/558 in both runs;
- summed solver time: 187.13 s to 184.58 s;
- median time: 5.16 ms to 4.60 ms;
- maximum time: 3.001 s in both runs;
- no lost proofs or meaningfully worse incumbents.

Raw result: `tspn-comparison/results/socg-full/ours-deadline.jsonl`.

The small speed and proof changes may include parallel scheduling noise. The
important standalone result is that nine oracle calls were safely interrupted
at the deadline and all 558 returned paths still passed independent validation.

Full 558-instance German search + TPP rerun:

- proofs: 393 to 394;
- strict-valid incumbents: 504/558 in both runs;
- summed solver time: 666.54 s to 566.86 s, a 14.95% reduction;
- median time: 92.93 ms to 89.04 ms;
- maximum time: 25.97 s to 3.068 s;
- runs exceeding 4 seconds: 20 to 0;
- no lost proofs, valid incumbents, or meaningfully worse incumbents.

Raw result:
`tspn-comparison/results/socg-deadline-full/20260911-040646/`.

The 12 worst historical tail cases fell from 100.65 s to 36.03 s in total.
This completes the primary work in item 2 below. Small post-deadline node
cleanup still allows roughly 30-70 ms of overrun in the German search.

### 1. Import the useful German search heuristics

Explored on 2026-09-11: an optional bidirectional version of our existing
nearest-neighbor/contact/2-opt initializer now runs from both endpoints and
keeps the better initial path. It is available through
`--bidirectional-initial` and is disabled by default because the full-corpus
result was mixed:

- proof count remained 526/558;
- summed time was unchanged within noise, 184.58 s versus 184.60 s;
- total calls fell from 5,374,437 to 5,287,246;
- four final timeout incumbents improved by 1.13% to 1.45%;
- four final timeout incumbents worsened by 0.10% to 0.89%;
- all 558 paths remained strictly valid.

Raw bidirectional result:
`tspn-comparison/results/socg-full/ours-bidirectional.jsonl`.

On cases 65 and 556, the reverse start reproduced the better German incumbent
to numerical precision. This identifies endpoint direction as one source of
their incumbent advantage. It does not yet justify enabling the heuristic by
default because changing the initial upper bound changes which nodes are
visited before the time limit.

- Identify the initial orders, node priorities, and incumbent updates used on
  the 13 cases where German search + TPP has the better incumbent.
- Add those ideas as optional heuristics in the standalone solver.
- Consider a short portfolio phase that preserves candidates from both search
  trajectories instead of replacing the one-way trajectory.
- A/B test them specifically on the 33 standalone timeout cases, then rerun the
  full suite to check for regressions.

### 2. Further reduce deadline latency

- Cache repeated fallback/refinement work.
- Profile the fallback triggers and separate numerical conditioning problems
  from geometric certificate repair.
- Add deadline checks within long German node/callback processing if a harder
  wall-time guarantee than the current approximately 70 ms overrun is needed.

The original 25.97-second overrun has been removed by the completed deadline
work above.

### 3. Strengthen the large-instance search

The remaining weakness is concentrated at 51-60 polygons, where standalone
TPP proves 43/70 cases. For the 33 unproven cases, the median remaining relative
gap is about 2.77% and the maximum is about 8.09%.

Investigate:

- stronger lower bounds and earlier bound reuse;
- branch and node ordering;
- better initial tours and incumbent polishing;
- symmetry or dominance reductions;
- profiles split by time in the oracle, queue growth, and node expansion.

### 4. Consider an optional portfolio fallback

Keep the dependency-light TPP oracle as the default. Optionally use a cheap
approximate convex backend only when a call is predicted to require expensive
certification. Do not make Gurobi a required dependency.

### 5. Support the original German problem directly

- Implement a cyclic TPP oracle, eliminating the artificial endpoints.
- Add polygon-with-holes support, or a rigorously equivalent representation.
- Then rerun both `instances_socg_simplified.zip` and `instances_socg.zip` under
  the original closed-tour objective.

## Publication-level checks, if needed

The current evidence is sufficient for engineering conclusions. Before making
a formal performance claim:

- repeat runs sequentially and with several randomized instance orderings;
- report medians or confidence intervals across repetitions;
- report timeouts as censored observations, not ordinary completed runtimes;
- freeze the code revisions, generated suite, hardware, tolerances, and seeds;
- preserve both strict `1e-7` validation and the German native `1e-3` metric.

## Independent generated canon campaign

Completed on 2026-09-11. The tracked generator
`benchmarks/scripts/generate_free_order_canon.py` creates an independent suite
from `packages/instance-generation`, instead of reusing the German corpus. The
default São Paulo campaign contains 120 inspectable diagnostic cases and 540
held-out cases, balanced across 40, 60, and 80 polygons. Its ten profiles cover
density, clearance, sparsity, polygon complexity, convex/nonconvex mixtures,
endpoint direction, geographic placement, and numerical scale.

The generated campaign is at
`benchmarks/campaigns/free-order-canon-v1/`. All 660 cases are unique,
pairwise-disjoint within each instance, and round-trip through the binary
format. The 66 endpoint pairs have identical geometry with exchanged
endpoints. The 66 scale pairs have identical geometry modulo their `1e12`
scale ratio. Combined suite hashes:

- diagnostic: `8c09a258fd0e70b34b2b8796e5dd78aadd61a92ec2edb3814d949e2bf0e5af4a`;
- held-out: `6883b9a821186bd798cac3d7120d69b5dcd8a0b0ba02b3e0dc291449ff012fb8`.

Only the diagnostic split has been run. With the current default solver and a
3-second limit, all 120 incumbents passed strict validation and 22 were proven
optimal. This suite is intentionally much harder than the German-derived
development set: 98/120 reached the time limit.

The paired scale test is reassuring for the improved solver. Both the `1e-6`
and `1e6` versions proved 4/12 cases, with median call counts of 195,036 and
198,519. Small scale caused all six observed certified fallback calls, however,
so the pair remains a useful numerical regression test even though it did not
produce a material search-performance difference.

The endpoint-direction pair also showed no proof-count difference on this
suite: neither orientation proved any of its 12 deliberately difficult cases.
The default orientation used a median 249,372 calls and the reverse used
257,917. This remains useful for studying incumbent quality, but does not by
itself justify enabling bidirectional initialization.

Do not inspect case-level results from `free-order-canon-v1.bin` during tuning.
Use `free-order-dev-v1.bin`, then run the frozen 540-case canon only at a
decision checkpoint. Regeneration and benchmark commands are documented in
`benchmarks/README.md`.

## Files needed to resume

- Full explanation and reproduction commands:
  [`docs/free-order-oracle-comparison.md`](docs/free-order-oracle-comparison.md)
- Compact generated summary:
  [`tspn-comparison/results/socg-full/comparison.md`](tspn-comparison/results/socg-full/comparison.md)
- Standalone raw results:
  `tspn-comparison/results/socg-full/ours.jsonl`
- German + SOCP raw results:
  `tspn-comparison/results/socg-full/socp/20260910-210701/`
- German + TPP raw results:
  `tspn-comparison/results/socg-full/tpp/20260910-211002/`
- Source corpus:
  `tspn-comparison/solver/instances/instances_socg_simplified.zip`
- Converted suite:
  `tspn-comparison/instances/socg-simplified-bbox-path-v1.bin`
- Conversion and benchmark scripts:
  `benchmarks/scripts/convert_tspn_native_instances.py`,
  `benchmarks/scripts/tspn_run_comparison.py`, and
  `benchmarks/scripts/unordered_benchmark.py`
- Reapplicable German integration patches:
  `benchmarks/patches/tspn-bnb2-tpp-oracle.patch` followed by
  `benchmarks/patches/tspn-bnb2-tpp-deadline.patch`

Some generated suites and raw result directories are intentionally ignored by
Git. Back them up separately before cleaning the working tree or moving to a
different machine.
